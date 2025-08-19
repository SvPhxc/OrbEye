#!/usr/bin/env python3
"""
LiDAR Target Tracker
Continuously tracks a target by scanning in a circle around the current position,
finding the highest strength viable point, and moving to it.
Uses ClutterFilter to reject static background objects.
Controlled by debug_mode flag and saves best points to satellite_points.
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math


class ClutterFilter:
    """
    A robust filter that rejects static background objects by first finding clutter
    in the same direction (az/el) and then checking if the new point is
    significantly closer than the known background object.
    """

    def __init__(self, background_file="background_scan.npy", angular_tolerance=1, distance_margin_cm=70.0):
        """
        Initializes the filter with a 2D (azimuth, elevation) background map.

        Args:
            background_file (str): Path to the background scan data.
            angular_tolerance (float): The maximum angle (in degrees) to consider a point
                                       as being in the "same direction" as a background point.
            distance_margin_cm (float): A new point must be at least this much closer
                                        than a background object in the same direction to be
                                        considered a valid target.
        """
        self.angular_tolerance = angular_tolerance
        self.distance_margin_cm = distance_margin_cm
        self.background_tree = None
        self.background_data = None

        try:
            # Load background data [azimuth, elevation, distance_cm, strength]
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")

            # Build k-d tree on directional coordinates ONLY [az, el] for fast angular search.
            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords)
            print("[ClutterFilter] 2D (directional) k-d tree built successfully.")

        except FileNotFoundError:
            print(
                f"[ClutterFilter] WARNING: Background file '{background_file}' not found. Running without clutter filtering.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR loading background data: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """
        Checks if a measurement is a valid target or background clutter.

        Returns:
            bool: True if the target is valid, False if it's likely clutter.
        """
        if self.background_tree is None:
            return True  # No background data, so all targets are considered valid.

        query_point = np.array([azimuth, elevation])
        try:
            # Query the 2D tree to find the angularly closest background point.
            angular_dist, idx = self.background_tree.query(query_point, k=1)

            # 1. Is there a background object in this direction?
            if angular_dist < self.angular_tolerance:
                # 2. Get the distance of that background object.
                bg_distance = self.background_data[idx, 2]

                # 3. Is the new measurement significantly IN FRONT of the background?
                if distance < (bg_distance - self.distance_margin_cm):
                    return True  # Yes, it's a valid target in front of the background.
                else:
                    # No, it's at or behind the background object. It's clutter.
                    return False

            # If no background points are nearby in this direction, it's a valid target.
            return True

        except Exception as e:
            print(f"[ClutterFilter] Query error: {e}")
            return True  # Fail-safe: if the query fails, accept the measurement.


class TargetTracker:
    """
    Tracks a target by continuously scanning around the current position
    and moving to the point with highest strength.
    Controlled by debug_mode flag and saves best points to satellite_points.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        """
        Initialize the tracker with shared data and clutter filter.

        Args:
            shared_data: Shared memory dictionary from the hardware controller
            background_file: Path to background scan data for clutter filtering
        """
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file=background_file)

        # Tracking parameters
        self.scan_radius_az = 10.0  # Degrees to scan left/right from center
        self.scan_radius_el = 10.0  # Degrees to scan up/down from center
        self.scan_points = 16  # Number of points to scan in the circle
        self.min_strength_threshold = 100  # Minimum strength to consider a valid target
        self.sample_time = 0.05  # Time to sample at each scan point (seconds)
        self.samples_per_point = 2  # Number of samples to average at each point

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.lost_target_count = 0
        self.max_lost_count = 5  # Number of scans without target before expanding search

        # Performance tracking
        self.last_scan_time = 0
        self.target_history = []  # Store last N target positions for smoothing
        self.history_size = 5

        print("[Tracker] Target tracker initialized")
        print(f"[Tracker] Scan radius: ±{self.scan_radius_az}° azimuth, ±{self.scan_radius_el}° elevation")
        print(f"[Tracker] Scan points: {self.scan_points} points per circle")
        print("[Tracker] Waiting for debug_mode to be enabled...")

    def update_satellite_points(self, azimuth, elevation, distance, strength):
        """
        Update the satellite_points array with the best target data.
        Array format: [azimuth, elevation, distance, strength, timestamp]

        Args:
            azimuth: Target azimuth in degrees
            elevation: Target elevation in degrees
            distance: Target distance in cm
            strength: Target signal strength
        """
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()

            print(f"[Tracker] Updated satellite_points: az={azimuth:.1f}°, el={elevation:.1f}°, "
                  f"dist={distance:.0f}cm, str={strength:.0f}")
        except Exception as e:
            print(f"[Tracker] Error updating satellite_points: {e}")

    def clear_satellite_points(self):
        """
        Clear the satellite_points array (set to zeros).
        """
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
            print("[Tracker] Cleared satellite_points")
        except Exception as e:
            print(f"[Tracker] Error clearing satellite_points: {e}")

    def scan_circle(self, center_az, center_el):
        """
        Scan in a circle around the given center point.

        Args:
            center_az: Center azimuth in degrees
            center_el: Center elevation in degrees

        Returns:
            List of (azimuth, elevation, distance, strength) tuples for valid targets
        """
        scan_results = []

        print(f"[Tracker] Scanning circle around ({center_az:.1f}°, {center_el:.1f}°)")

        for i in range(self.scan_points):
            # Check for shutdown or debug_mode disabled
            if self.shared_data["shutdown"].value or not self.shared_data["debug_mode"].value:
                print("[Tracker] Scan interrupted")
                return scan_results

            # Calculate scan point on circle
            angle = (2 * math.pi * i) / self.scan_points

            # Calculate position with elliptical pattern (different radius for az/el)
            scan_az = center_az + self.scan_radius_az * math.cos(angle)
            scan_el = center_el + self.scan_radius_el * math.sin(angle)

            # Clamp elevation to valid range
            scan_el = max(0, min(90, scan_el))

            # Wrap azimuth to 0-360
            scan_az = scan_az % 360
            if scan_az < 0:
                scan_az += 360

            # Move to scan point
            self._move_to_position(scan_az, scan_el)

            # Wait for movement to complete
            time.sleep(0.005)  # Small delay for movement

            # Collect samples at this position
            samples = self._collect_samples()

            # Process samples through clutter filter
            for az, el, dist, strength in samples:
                if self.clutter_filter.is_valid_target(az, el, dist, strength):
                    if strength >= self.min_strength_threshold:
                        scan_results.append((az, el, dist, strength))
                        print(f"[Tracker]   Valid target at ({az:.1f}°, {el:.1f}°): "
                              f"dist={dist:.0f}cm, str={strength}")

        return scan_results

    def _move_to_position(self, azimuth, elevation):
        """
        Move the scanner to the specified position.

        Args:
            azimuth: Target azimuth in degrees
            elevation: Target elevation in degrees
        """
        # Check for shutdown
        if self.shared_data["shutdown"].value:
            return

        # Set target positions
        self.shared_data["target_azimuth"].value = azimuth
        self.shared_data["target_elevation"].value = elevation

        # Trigger movement
        self.shared_data["go_to_target"].value = True

        # Wait for movement to start
        timeout = time.time() + 2.0  # 2 second timeout
        while self.shared_data["go_to_target"].value and time.time() < timeout:
            if self.shared_data["shutdown"].value:
                break
            time.sleep(0.001)

    def _collect_samples(self):
        """
        Collect LiDAR samples at the current position.

        Returns:
            List of (azimuth, elevation, distance, strength) tuples
        """
        samples = []

        for _ in range(self.samples_per_point):
            # Check for shutdown
            if self.shared_data["shutdown"].value:
                break

            # Get current position
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            # Get LiDAR data
            with self.shared_data["lidar_data"].get_lock():
                dist = self.shared_data["lidar_data"][0]
                strength = self.shared_data["lidar_data"][1]

            if dist > 0:  # Valid measurement
                samples.append((current_az, current_el, dist, strength))

            time.sleep(self.sample_time)

        return samples

    def find_best_target(self, scan_results):
        """
        Find the best target from scan results based on strength.

        Args:
            scan_results: List of (azimuth, elevation, distance, strength) tuples

        Returns:
            Best target as (azimuth, elevation, distance, strength) or None
        """
        if not scan_results:
            return None

        # Sort by strength (highest first)
        sorted_results = sorted(scan_results, key=lambda x: x[3], reverse=True)

        # Return the strongest target
        best = sorted_results[0]
        print(f"[Tracker] Best target: ({best[0]:.1f}°, {best[1]:.1f}°) "
              f"dist={best[2]:.0f}cm, str={best[3]}")

        # Save to satellite_points
        self.update_satellite_points(best[0], best[1], best[2], best[3])

        return best

    def smooth_target_position(self, new_target):
        """
        Smooth target position using history to reduce jitter.

        Args:
            new_target: New target position (az, el, dist, strength)

        Returns:
            Smoothed position (az, el)
        """
        if new_target:
            self.target_history.append((new_target[0], new_target[1]))

            # Keep history size limited
            if len(self.target_history) > self.history_size:
                self.target_history.pop(0)

            # Calculate average position
            if len(self.target_history) > 1:
                avg_az = sum(h[0] for h in self.target_history) / len(self.target_history)
                avg_el = sum(h[1] for h in self.target_history) / len(self.target_history)
                return avg_az, avg_el

        return new_target[0], new_target[1] if new_target else (None, None)

    def expand_search(self):
        """
        Expand search radius when target is lost.
        """
        self.scan_radius_az = min(self.scan_radius_az * 1.5, 15.0)  # Max 45 degrees
        self.scan_radius_el = min(self.scan_radius_el * 1.5, 15.0)  # Max 30 degrees
        print(f"[Tracker] Expanding search radius to ±{self.scan_radius_az:.1f}° az, "
              f"±{self.scan_radius_el:.1f}° el")

    def reset_search_radius(self):
        """
        Reset search radius to default when target is found.
        """
        self.scan_radius_az = 10.0
        self.scan_radius_el = 10.0

    def run(self):
        """
        Main tracking loop - monitors debug_mode flag and tracks when enabled.
        """
        print("[Tracker] Tracker process started")

        try:
            while not self.shared_data["shutdown"].value:
                # Check if tracking should be active
                if self.shared_data["debug_mode"].value:
                    # Debug mode enabled - start tracking
                    if self.current_target_az is None:
                        # Initialize position
                        self.current_target_az = self.shared_data["stepper_degrees"].value
                        self.current_target_el = self.shared_data["servo_degrees"].value

                        # If still not set, use defaults
                        if self.current_target_az == 0 and self.current_target_el == 0:
                            self.current_target_az = 180.0
                            self.current_target_el = 45.0

                        print(f"[Tracker] Starting tracking at ({self.current_target_az:.1f}°, "
                              f"{self.current_target_el:.1f}°)")

                    # Perform one tracking cycle
                    scan_start = time.time()

                    # Scan around current position
                    scan_results = self.scan_circle(self.current_target_az, self.current_target_el)

                    # Find best target
                    best_target = self.find_best_target(scan_results)

                    if best_target:
                        # Target found - reset lost count
                        self.lost_target_count = 0
                        self.reset_search_radius()

                        # Smooth the target position
                        smooth_az, smooth_el = self.smooth_target_position(best_target)

                        # Update current target position
                        self.current_target_az = smooth_az
                        self.current_target_el = smooth_el

                        print(f"[Tracker] Moving to target at ({smooth_az:.1f}°, {smooth_el:.1f}°)")

                        # Move to new target position
                        self._move_to_position(smooth_az, smooth_el)

                    else:
                        # No target found
                        self.lost_target_count += 1
                        print(f"[Tracker] No target found (lost count: {self.lost_target_count})")

                        # Clear satellite_points when no target
                        self.clear_satellite_points()

                        if self.lost_target_count >= self.max_lost_count:
                            # Expand search area
                            self.expand_search()
                            self.lost_target_count = 0  # Reset count after expanding

                    # Calculate and display scan time
                    scan_time = time.time() - scan_start
                    print(f"[Tracker] Scan completed in {scan_time:.2f}s")

                else:
                    # Debug mode disabled - wait and clear data
                    if self.current_target_az is not None:
                        print("[Tracker] Debug mode disabled, stopping tracking")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.target_history = []
                        self.lost_target_count = 0
                        self.reset_search_radius()
                        self.clear_satellite_points()

                    time.sleep(0.1)  # Wait while disabled

                # Small delay between cycles
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("[Tracker] Tracking interrupted by user")
        except Exception as e:
            print(f"[Tracker] Tracking error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Tracker] Tracker process shutting down")
            self.clear_satellite_points()


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """
    Main function to run the target tracker as a process.

    Args:
        shared_data: Shared memory dictionary from hardware controller
        background_file: Path to background scan data
    """
    print("[Tracker] Initializing tracker process...")
    tracker = TargetTracker(shared_data, background_file)

    try:
        # Run the tracker
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
    finally:
        print("[Tracker] Process ended")


