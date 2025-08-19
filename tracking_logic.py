#!/usr/bin/env python3
"""
Robust LiDAR Target Tracker with Acquisition and Demo Modes
Key features:
- Initial acquisition scan triggered by acquire_points flag
- Demo mode for tracking orbiting drone (2m away, 20s orbit)
- Waits for target_reached to ensure accurate positioning
- Verifies LiDAR data matches scan position
- Fixed angle wraparound at 0°/360° boundary
- Respects 1000Hz LiDAR polling limit
- Predictive tracking for smooth following

Demo Mode:
- Triggered by 'demo' flag
- Tracks drone orbiting at ~2m distance
- Uses 'heading' for initial direction (optional)
- Uses 'inclination' for orbit plane (-1 if unknown)
- Automatically determines orbit parameters from motion
- Predictive tracking based on 20-second orbit period
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math
from collections import deque


class ClutterFilter:
    """
    Optimized clutter filter with caching for better performance.
    """

    def __init__(self, background_file="background_scan.npy", angular_tolerance=1, distance_margin_cm=70.0):
        self.angular_tolerance = angular_tolerance
        self.distance_margin_cm = distance_margin_cm
        self.background_tree = None
        self.background_data = None
        self._query_cache = {}
        self._cache_size = 75000

        try:
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")

            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords, leafsize=16)
            self.bg_distances = self.background_data[:, 2]
            print("[ClutterFilter] K-d tree built successfully.")

        except FileNotFoundError:
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """Check if target is valid with caching."""
        if self.background_tree is None:
            return True

        cache_key = (int(azimuth * 10), int(elevation * 10))

        if cache_key in self._query_cache:
            bg_distance = self._query_cache[cache_key]
        else:
            query_point = np.array([azimuth, elevation])
            try:
                angular_dist, idx = self.background_tree.query(query_point, k=1)

                if angular_dist < self.angular_tolerance:
                    bg_distance = self.bg_distances[idx]
                else:
                    bg_distance = float('inf')

                if len(self._query_cache) < self._cache_size:
                    self._query_cache[cache_key] = bg_distance

            except Exception as e:
                print(f"[ClutterFilter] Query error: {e}")
                return True

        return distance < (bg_distance - self.distance_margin_cm)


class AngleHandler:
    """Robust angle handling with 0/360 wraparound support."""

    @staticmethod
    def normalize(angle):
        """Normalize angle to [0, 360) range."""
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle

    @staticmethod
    def difference(angle1, angle2):
        """Calculate shortest angular difference between two angles."""
        diff = (angle2 - angle1 + 180) % 360 - 180
        return diff

    @staticmethod
    def shortest_path(current, target):
        """Calculate the target angle that requires minimum rotation."""
        diff = AngleHandler.difference(current, target)
        return AngleHandler.normalize(current + diff)

    @staticmethod
    def circular_mean(angles):
        """Compute mean of angles, properly handling wraparound."""
        if not angles:
            return None
        x_sum = sum(math.cos(math.radians(a)) for a in angles)
        y_sum = sum(math.sin(math.radians(a)) for a in angles)
        mean_angle = math.degrees(math.atan2(y_sum, x_sum))
        return AngleHandler.normalize(mean_angle)

    @staticmethod
    def is_near_boundary(angle, threshold=15):
        """Check if angle is near the 0/360 boundary."""
        norm_angle = AngleHandler.normalize(angle)
        return norm_angle < threshold or norm_angle > (360 - threshold)


class TargetTracker:
    """
    Robust tracker with acquisition mode, position verification, and demo mode.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file=background_file)
        self.angle_handler = AngleHandler()

        # Demo mode parameters
        self.demo_mode = False
        self.demo_heading = 0.0  # Current heading in orbit
        self.demo_inclination = -1  # -1 means unknown
        self.demo_orbit_time = 20.0  # seconds for full orbit
        self.demo_angular_velocity = 360.0 / self.demo_orbit_time  # 18 degrees/second
        self.demo_radius = 200.0  # 2 meters in cm
        self.demo_center_el = 45.0  # Default elevation center
        self.demo_last_update = None
        self.demo_orbit_points = []  # Points to determine orbit plane
        self.demo_orbit_determined = False

        # LiDAR rate limiting (1000Hz max)
        self.lidar_min_interval = 0.002  # 1ms minimum between reads
        self.last_lidar_read = 0

        # Acquisition mode parameters
        self.acquisition_radius_az = 30.0  # Wide initial search
        self.acquisition_radius_el = 20.0
        self.acquisition_points = 12  # More points for initial search
        self.acquisition_active = False

        # Normal tracking parameters (slower for reliability)
        self.scan_radius_az = 10.0
        self.scan_radius_el = 6.0
        self.scan_points = 8
        self.min_strength_threshold = 600
        self.high_confidence_threshold = 4000  # Higher strength means more confidence

        # Movement parameters (slower for accuracy)
        self.movement_timeout = 1.0  # Longer timeout for reliability
        self.position_tolerance = 0  # Tighter tolerance
        self.position_verify_delay = 0.015  # Time to wait after reaching position

        # Position verification
        self.max_position_error = 0.5  # Maximum acceptable position error

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.tracking_confidence = 0.0
        self.consecutive_good_tracks = 0

        # History tracking
        self.target_history = deque(maxlen=4)
        self.position_history = deque(maxlen=5)
        self.lost_target_count = 0
        self.max_lost_count = 5

        # Performance monitoring
        self.cycle_count = 0
        self.successful_reads = 0
        self.failed_reads = 0

        print("[Tracker] Robust tracker with acquisition mode initialized")
        print(f"[Tracker] Acquisition scan: ±{self.acquisition_radius_az}° with {self.acquisition_points} points")

    def wait_for_target_reached(self, timeout=1.0):
        """Wait for the system to reach the target position."""
        start_time = time.time()

        # First wait for movement to start (go_to_target to be processed)
        while self.shared_data["go_to_target"].value and time.time() - start_time < 0.1:
            time.sleep(0.001)

        # Now wait for target_reached flag
        while time.time() - start_time < timeout:
            if self.shared_data["shutdown"].value:
                return False

            if self.shared_data["target_reached"].value:
                # Clear the flag
                self.shared_data["target_reached"].value = False
                # Small delay to let system stabilize
                time.sleep(self.position_verify_delay)
                return True

            time.sleep(0.001)

        return False  # Timeout

    def move_to_position_verified(self, azimuth, elevation):
        """Move to position and verify we actually reached it."""
        if self.shared_data["shutdown"].value:
            return False

        # Get current position
        current_az = self.shared_data["stepper_degrees"].value

        # Use shortest path for azimuth
        target_az = self.angle_handler.shortest_path(current_az, azimuth)

        # Set targets
        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = elevation

        # Trigger movement
        self.shared_data["go_to_target"].value = True

        # Wait for target to be reached
        if not self.wait_for_target_reached(self.movement_timeout):
            self.failed_reads += 1
            print(f"[Tracker] Warning: Failed to reach position ({target_az:.1f}°, {elevation:.1f}°)")
            return False

        # Verify position
        actual_az = self.shared_data["stepper_degrees"].value
        actual_el = self.shared_data["servo_degrees"].value

        az_error = abs(self.angle_handler.difference(actual_az, target_az))
        el_error = abs(actual_el - elevation)

        if az_error > self.max_position_error or el_error > self.max_position_error:
            print(f"[Tracker] Position error too large: az_err={az_error:.1f}°, el_err={el_error:.1f}°")
            self.failed_reads += 1
            return False

        self.successful_reads += 1
        return True

    def read_lidar_at_position(self):
        """Read LiDAR data after confirming position."""
        # Ensure we don't exceed 1000Hz
        elapsed = time.time() - self.last_lidar_read
        if elapsed < self.lidar_min_interval:
            time.sleep(self.lidar_min_interval - elapsed)

        # Get current actual position
        actual_az = self.shared_data["stepper_degrees"].value
        actual_el = self.shared_data["servo_degrees"].value

        # Read LiDAR data
        with self.shared_data["lidar_data"].get_lock():
            distance = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]

        self.last_lidar_read = time.time()

        return actual_az, actual_el, distance, strength

    def acquisition_scan(self):
        """Perform wide initial scan to find any target."""
        print("[Tracker] Starting acquisition scan...")
        best_target = None
        best_strength = 0

        # Scan in a grid pattern
        for el_offset in [0, -10, 10, -20, 20]:
            elevation = np.clip(45 + el_offset, 10, 80)

            for i in range(self.acquisition_points):
                if self.shared_data["shutdown"].value:
                    return None

                # Calculate azimuth positions
                azimuth = (360.0 * i) / self.acquisition_points

                # Move to position and wait for confirmation
                if not self.move_to_position_verified(azimuth, elevation):
                    continue

                # Read LiDAR at confirmed position
                actual_az, actual_el, distance, strength = self.read_lidar_at_position()

                if distance > 0:
                    # Check if valid target
                    if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                        if strength >= self.min_strength_threshold:
                            print(f"[Acquisition] Found: ({actual_az:.1f}°, {actual_el:.1f}°) "
                                  f"dist={distance:.0f}cm, str={strength:.0f}")

                            if strength > best_strength:
                                best_target = (actual_az, actual_el, distance, strength)
                                best_strength = strength

                            # If we find a really good target, stop searching
                            if strength > 200:
                                print(f"[Acquisition] Strong target found, ending scan")
                                return best_target

        if best_target:
            print(f"[Acquisition] Best target: ({best_target[0]:.1f}°, {best_target[1]:.1f}°) "
                  f"str={best_target[3]:.0f}")
        else:
            print("[Acquisition] No target found")

        return best_target

    def tracking_scan(self, center_az, center_el):
        """Perform tracking scan with position verification."""
        scan_results = []
        scan_start = time.time()

        # Adjust scan density based on confidence
        if self.tracking_confidence > 0.7:
            points_to_scan = max(3, self.scan_points - 2)
            radius_az = self.scan_radius_az * 0.7
            radius_el = self.scan_radius_el * 0.7
        else:
            points_to_scan = self.scan_points
            radius_az = self.scan_radius_az
            radius_el = self.scan_radius_el

        # Always scan center point first
        if self.move_to_position_verified(center_az, center_el):
            az, el, dist, strength = self.read_lidar_at_position()
            if dist > 0 and self.clutter_filter.is_valid_target(az, el, dist, strength):
                if strength >= self.min_strength_threshold:
                    scan_results.append((az, el, dist, strength))

        # Scan surrounding points
        for i in range(points_to_scan):
            if self.shared_data["shutdown"].value or not self.shared_data["debug_mode"].value:
                break

            # Time limit for scanning
            if time.time() - scan_start > 0.8:
                break

            angle = (2 * math.pi * i) / points_to_scan
            scan_az = center_az + radius_az * math.cos(angle)
            scan_el = center_el + radius_el * math.sin(angle)
            scan_el = np.clip(scan_el, 0, 90)
            scan_az = self.angle_handler.normalize(scan_az)

            # Move and verify position
            if not self.move_to_position_verified(scan_az, scan_el):
                continue

            # Read at verified position
            actual_az, actual_el, distance, strength = self.read_lidar_at_position()

            if distance > 0:
                if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                    if strength >= self.min_strength_threshold:
                        scan_results.append((actual_az, actual_el, distance, strength))

        return scan_results

    def find_best_target(self, scan_results):
        """Find the best target with consistency checking."""
        if not scan_results:
            return None

        # Sort by strength
        scan_results.sort(key=lambda x: x[3], reverse=True)

        # If we have history, prefer targets close to previous position
        if self.target_history and len(self.target_history) > 1:
            last_az, last_el = self.target_history[-1]

            # Score targets based on strength and proximity to last position
            def score_target(target):
                az, el, dist, strength = target
                az_diff = abs(self.angle_handler.difference(last_az, az))
                el_diff = abs(last_el - el)
                position_error = math.sqrt(az_diff ** 2 + el_diff ** 2)

                # Penalize targets that are too far from last position
                if position_error > 20:  # More than 20 degrees away
                    return strength * 0.3
                elif position_error > 10:
                    return strength * 0.7
                else:
                    return strength

            best = max(scan_results, key=score_target)
        else:
            # No history, just use strongest
            best = scan_results[0]

        return best

    def smooth_position(self, new_az, new_el):
        """Smooth position with proper angle wraparound handling."""
        self.position_history.append((new_az, new_el))

        if len(self.position_history) < 2:
            return new_az, new_el

        # Check if angles are near boundary
        az_values = [p[0] for p in self.position_history]
        near_boundary = any(self.angle_handler.is_near_boundary(az) for az in az_values)

        if near_boundary:
            # Use circular mean for angles near boundary
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            # Weighted average with more weight on recent values
            weights = np.exp(np.linspace(-2, 0, len(self.position_history)))
            weights /= weights.sum()
            smooth_az = sum(az * w for (az, _), w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        # Weighted average for elevation
        el_values = [p[1] for p in self.position_history]
        smooth_el = np.average(el_values, weights=weights)

        return smooth_az, smooth_el

    def update_tracking_confidence(self, found_target, target_strength=0):
        """Update confidence based on tracking success."""
        if found_target:
            if target_strength > self.high_confidence_threshold:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
                self.consecutive_good_tracks += 1
            else:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.1)
                self.consecutive_good_tracks = 0
        else:
            self.tracking_confidence = max(0.0, self.tracking_confidence - 0.3)
            self.consecutive_good_tracks = 0

    def update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update satellite points in shared memory."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()

            print(f"[Tracker] Target locked: ({azimuth:.1f}°, {elevation:.1f}°) "
                  f"dist={distance:.0f}cm, str={strength:.0f}, conf={self.tracking_confidence:.2f}")
        except Exception as e:
            print(f"[Tracker] Error updating satellite_points: {e}")

    def clear_satellite_points(self):
        """Clear satellite points."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
        except Exception as e:
            print(f"[Tracker] Error clearing satellite_points: {e}")

    def demo_determine_orbit_plane(self):
        """Determine the orbit plane from collected points."""
        if len(self.demo_orbit_points) < 3:
            return False

        # Calculate the plane of motion from points
        points = self.demo_orbit_points[-3:]  # Use last 3 points

        # Calculate elevation changes
        el_changes = []
        az_changes = []
        for i in range(len(points) - 1):
            az_diff = self.angle_handler.difference(points[i][0], points[i + 1][0])
            el_diff = points[i + 1][1] - points[i][1]
            az_changes.append(az_diff)
            el_changes.append(el_diff)

        # Determine inclination (elevation change per degree of azimuth)
        avg_az_change = np.mean(np.abs(az_changes))
        avg_el_change = np.mean(el_changes)

        if avg_az_change > 0:
            self.demo_inclination = avg_el_change / avg_az_change
        else:
            self.demo_inclination = 0.0

        # Determine direction (clockwise or counter-clockwise)
        if np.mean(az_changes) > 0:
            self.demo_angular_velocity = abs(self.demo_angular_velocity)  # Moving right (CW)
        else:
            self.demo_angular_velocity = -abs(self.demo_angular_velocity)  # Moving left (CCW)

        # Set center elevation based on current points
        self.demo_center_el = np.mean([p[1] for p in points])

        print(f"[Demo] Orbit determined: inclination={self.demo_inclination:.2f}°/°, "
              f"direction={'CW' if self.demo_angular_velocity > 0 else 'CCW'}, "
              f"center_el={self.demo_center_el:.1f}°")

        self.demo_orbit_determined = True
        return True

    def demo_predict_position(self, current_time):
        """Predict drone position based on orbital motion."""
        if self.demo_last_update is None:
            return self.demo_heading, self.demo_center_el

        # Calculate time elapsed
        dt = current_time - self.demo_last_update

        # Predict new heading
        predicted_heading = self.demo_heading + self.demo_angular_velocity * dt
        predicted_heading = self.angle_handler.normalize(predicted_heading)

        # Predict elevation based on inclination
        if self.demo_inclination != -1 and self.demo_inclination != 0:
            # Calculate elevation change based on heading change
            heading_change = self.demo_angular_velocity * dt
            el_change = heading_change * self.demo_inclination
            predicted_el = self.demo_center_el + el_change

            # Add sinusoidal variation for inclined orbit
            predicted_el += 5 * math.sin(math.radians(predicted_heading))
        else:
            predicted_el = self.demo_center_el

        predicted_el = np.clip(predicted_el, 10, 80)

        return predicted_heading, predicted_el

    def demo_track_orbit(self):
        """Track drone in orbital motion with prediction."""
        current_time = time.time()

        # Predict where drone should be
        predicted_az, predicted_el = self.demo_predict_position(current_time)

        print(f"[Demo] Predicted position: ({predicted_az:.1f}°, {predicted_el:.1f}°)")

        # Use tighter scan for predictive tracking
        old_radius_az = self.scan_radius_az
        old_radius_el = self.scan_radius_el
        self.scan_radius_az = 5.0  # Tight scan around predicted position
        self.scan_radius_el = 5.0

        # Scan around predicted position
        scan_results = self.tracking_scan(predicted_az, predicted_el)

        # Restore scan radius
        self.scan_radius_az = old_radius_az
        self.scan_radius_el = old_radius_el

        if scan_results:
            # Find best target
            best = self.find_best_target(scan_results)

            if best:
                actual_az, actual_el, distance, strength = best

                # Calculate time delta
                dt = current_time - self.demo_last_update if self.demo_last_update else 0.05

                # Update heading and timing
                self.demo_heading = actual_az
                self.demo_last_update = current_time

                # Add to orbit points if still determining plane
                if not self.demo_orbit_determined:
                    self.demo_orbit_points.append((actual_az, actual_el))

                    # Try to determine orbit after collecting enough points
                    if len(self.demo_orbit_points) >= 3:
                        self.demo_determine_orbit_plane()

                # Update satellite points
                self.update_satellite_points(actual_az, actual_el, distance, strength)

                # Calculate actual angular velocity for adjustment
                if len(self.demo_orbit_points) > 1 and dt > 0:
                    prev_az = self.demo_orbit_points[-2][0]
                    actual_velocity = self.angle_handler.difference(prev_az, actual_az) / dt

                    # Adjust predicted velocity with smoothing
                    self.demo_angular_velocity = 0.7 * self.demo_angular_velocity + 0.3 * actual_velocity

                print(f"[Demo] Tracking: heading={actual_az:.1f}°, el={actual_el:.1f}°, "
                      f"dist={distance:.0f}cm, velocity={self.demo_angular_velocity:.1f}°/s")

                return True

        return False

    def demo_acquisition(self):
        """Special acquisition for demo mode - look for orbiting drone."""
        print("[Demo] Starting demo acquisition for orbiting drone...")

        # If heading is provided, start there
        if "heading" in self.shared_data and self.shared_data["heading"].value >= 0:
            start_heading = self.shared_data["heading"].value
            print(f"[Demo] Starting with provided heading: {start_heading:.1f}°")
        else:
            start_heading = None

        # Check if inclination is provided
        if "inclination" in self.shared_data:
            provided_inclination = self.shared_data["inclination"].value
            if provided_inclination != -1:
                self.demo_inclination = provided_inclination
                print(f"[Demo] Using provided inclination: {provided_inclination:.2f}°")

        # Scan pattern optimized for finding orbiting target
        scan_elevations = [45, 35, 55, 25, 65]  # Start at likely elevations

        if start_heading is not None:
            # Scan around provided heading
            scan_azimuths = [
                start_heading,
                start_heading + 10, start_heading - 10,
                start_heading + 20, start_heading - 20,
                start_heading + 30, start_heading - 30
            ]
        else:
            # Full circle scan
            scan_azimuths = np.linspace(0, 350, 12)

        best_target = None
        best_strength = 0

        for el in scan_elevations:
            for az in scan_azimuths:
                if self.shared_data["shutdown"].value:
                    return None

                az = self.angle_handler.normalize(az)

                # Move and verify
                if not self.move_to_position_verified(az, el):
                    continue

                # Read LiDAR
                actual_az, actual_el, distance, strength = self.read_lidar_at_position()

                # Check if this could be our drone (around 2m away)
                if 150 < distance < 250:  # 1.5m to 2.5m range
                    if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                        if strength >= self.min_strength_threshold:
                            print(f"[Demo] Found potential drone: ({actual_az:.1f}°, {actual_el:.1f}°) "
                                  f"dist={distance:.0f}cm, str={strength:.0f}")

                            if strength > best_strength:
                                best_target = (actual_az, actual_el, distance, strength)
                                best_strength = strength

                            # If very strong signal, likely our drone
                            if strength > 150:
                                break

            if best_target and best_strength > 150:
                break

        if best_target:
            # Initialize demo tracking
            self.demo_heading = best_target[0]
            self.demo_center_el = best_target[1]
            self.demo_last_update = time.time()
            self.demo_orbit_points = [(best_target[0], best_target[1])]

            print(f"[Demo] Acquisition successful: drone at ({best_target[0]:.1f}°, {best_target[1]:.1f}°)")

            # If inclination unknown, we need to track a few more points
            if self.demo_inclination == -1:
                print("[Demo] Inclination unknown, will determine from motion")
                self.demo_orbit_determined = False
            else:
                self.demo_orbit_determined = True

            return best_target

        print("[Demo] No drone found in expected range")
        return None

    def run(self):
        """Main tracking loop with acquisition mode and demo mode."""
        print("[Tracker] Starting robust tracking with acquisition and demo modes")

        try:
            while not self.shared_data["shutdown"].value:
                # Check for demo mode
                if "demo" in self.shared_data and self.shared_data["demo"].value:
                    if not self.demo_mode:
                        print("[Demo] Demo mode activated - tracking orbiting drone")
                        self.demo_mode = True

                        # Perform initial acquisition for demo
                        target = self.demo_acquisition()

                        if target:
                            self.current_target_az = target[0]
                            self.current_target_el = target[1]
                            self.tracking_confidence = 0.8  # High confidence for demo
                            self.update_satellite_points(target[0], target[1], target[2], target[3])
                        else:
                            print("[Demo] Failed to acquire drone")
                            self.demo_mode = False
                            self.shared_data["demo"].value = False
                            continue

                    # Run demo tracking
                    if self.demo_mode:
                        cycle_start = time.time()

                        # Track the orbiting drone
                        if self.demo_track_orbit():
                            # Successfully tracking
                            self.lost_target_count = 0
                        else:
                            # Lost drone
                            self.lost_target_count += 1
                            print(f"[Demo] Drone lost ({self.lost_target_count}/3)")

                            if self.lost_target_count >= 3:
                                print("[Demo] Drone lost, attempting re-acquisition")
                                target = self.demo_acquisition()

                                if target:
                                    self.lost_target_count = 0
                                    self.demo_heading = target[0]
                                    self.demo_last_update = time.time()
                                    print("[Demo] Re-acquired drone")
                                else:
                                    print("[Demo] Re-acquisition failed, exiting demo mode")
                                    self.demo_mode = False
                                    self.shared_data["demo"].value = False

                        # Maintain demo loop timing
                        cycle_time = time.time() - cycle_start
                        if cycle_time < 0.05:  # 20Hz update rate for smooth tracking
                            time.sleep(0.05 - cycle_time)

                        continue

                else:
                    if self.demo_mode:
                        print("[Demo] Demo mode deactivated")
                        self.demo_mode = False
                        self.demo_orbit_points = []
                        self.demo_orbit_determined = False

                # Check for acquisition trigger
                if self.shared_data["acquire_points"].value:
                    print("[Tracker] Acquisition triggered")
                    self.shared_data["acquire_points"].value = False

                    # Perform acquisition scan
                    target = self.acquisition_scan()

                    if target:
                        # Initialize tracking at found target
                        self.current_target_az = target[0]
                        self.current_target_el = target[1]
                        self.target_history.clear()
                        self.target_history.append((target[0], target[1]))
                        self.position_history.clear()
                        self.tracking_confidence = 0.5
                        self.update_satellite_points(target[0], target[1], target[2], target[3])

                        # Enable debug mode to start tracking
                        self.shared_data["debug_mode"].value = True
                        print(f"[Tracker] Acquisition successful, starting tracking at "
                              f"({target[0]:.1f}°, {target[1]:.1f}°)")
                    else:
                        print("[Tracker] Acquisition failed - no target found")
                        self.clear_satellite_points()

                    continue

                # Normal tracking mode
                if not self.shared_data["debug_mode"].value:
                    if self.current_target_az is not None:
                        print("[Tracker] Tracking disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.clear_satellite_points()
                        self.tracking_confidence = 0.0
                    time.sleep(0.1)
                    continue

                # Initialize tracking if needed
                if self.current_target_az is None:
                    print("[Tracker] Tracking enabled, waiting for acquisition...")
                    time.sleep(0.1)
                    continue

                cycle_start = time.time()

                # Perform tracking scan
                scan_results = self.tracking_scan(self.current_target_az, self.current_target_el)

                # Find best target
                best_target = self.find_best_target(scan_results)

                if best_target:
                    # Target found
                    self.lost_target_count = 0

                    # Update confidence
                    self.update_tracking_confidence(True, best_target[3])

                    # Add to history
                    self.target_history.append((best_target[0], best_target[1]))

                    # Smooth position
                    smooth_az, smooth_el = self.smooth_position(best_target[0], best_target[1])

                    # Update target position
                    self.current_target_az = smooth_az
                    self.current_target_el = smooth_el

                    # Update satellite points
                    self.update_satellite_points(smooth_az, smooth_el,
                                                 best_target[2], best_target[3])

                    # Reset scan radius if we have good confidence
                    if self.tracking_confidence > 0.5:
                        self.scan_radius_az = max(8.0, self.scan_radius_az * 0.9)
                        self.scan_radius_el = max(8.0, self.scan_radius_el * 0.9)

                else:
                    # Target lost
                    self.lost_target_count += 1
                    self.update_tracking_confidence(False)

                    print(f"[Tracker] Target lost ({self.lost_target_count}/{self.max_lost_count}), "
                          f"confidence={self.tracking_confidence:.2f}")

                    # Clear satellite points if confidence is too low
                    if self.tracking_confidence < 0.2:
                        self.clear_satellite_points()

                    # Expand search area
                    if self.lost_target_count >= self.max_lost_count:
                        self.scan_radius_az = min(self.scan_radius_az * 1.5, 30.0)
                        self.scan_radius_el = min(self.scan_radius_el * 1.5, 25.0)
                        self.lost_target_count = 0
                        print(f"[Tracker] Expanding search to ±{self.scan_radius_az:.1f}°")

                        # If we've lost it for too long, might need re-acquisition
                        if self.tracking_confidence < 0.1:
                            print("[Tracker] Lost target, need re-acquisition")
                            self.current_target_az = None
                            self.current_target_el = None

                # Performance monitoring
                cycle_time = time.time() - cycle_start
                self.cycle_count += 1

                if self.cycle_count % 20 == 0:
                    success_rate = (self.successful_reads / max(1, self.successful_reads + self.failed_reads)) * 100
                    print(f"[Tracker] Stats: {cycle_time * 1000:.0f}ms cycle, "
                          f"{success_rate:.0f}% position success")

                # Minimum cycle time to avoid overwhelming system
                if cycle_time < 0.05:  # Minimum 50ms per cycle
                    time.sleep(0.05 - cycle_time)

        except KeyboardInterrupt:
            print("[Tracker] Interrupted")
        except Exception as e:
            print(f"[Tracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Tracker] Shutting down")
            self.clear_satellite_points()


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """
    Run the robust tracker process with acquisition and demo modes.

    Shared data flags expected:
    - acquire_points: Trigger acquisition scan
    - debug_mode: Enable/disable tracking
    - demo: Enable demo mode for tracking orbiting drone
    - heading: Initial heading for demo mode (optional)
    - inclination: Orbit inclination (-1 if unknown)
    - target_reached: Set by motion system when position reached
    - shutdown: Stop the tracker
    """
    print("[Tracker] Initializing robust tracker with acquisition and demo...")

    # Ensure demo flags exist
    if "demo" not in shared_data:
        print("[Tracker] Warning: 'demo' flag not in shared_data")
    if "heading" not in shared_data:
        print("[Tracker] Warning: 'heading' value not in shared_data")
    if "inclination" not in shared_data:
        print("[Tracker] Warning: 'inclination' value not in shared_data")

    tracker = TargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
    finally:
        print("[Tracker] Process ended")