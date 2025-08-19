#!/usr/bin/env python3
"""
Optimized LiDAR Target Tracker with Rate Limiting and Point Acquisition Mode
Key improvements:
- Added a point acquisition mode triggered by a flag.
- Implemented a wide scan to find the first target.
- Engages tracker after initial point acquisition.
- Uses a "target_reached" flag to signal successful acquisition.
- Fixed angle wraparound issues at 0°/360° boundary.
- Respects 1000Hz LiDAR polling limit.
- Prevents getting stuck on high strength targets.
- Adaptive scanning without over-optimization.
- Predictive movement with proper timing.
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
        self._cache_size = 1500

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

        # Fast cache key using integer math
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
        # Convert to unit vectors for proper averaging
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
    Optimized tracker with proper rate limiting and angle handling.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file=background_file)
        self.angle_handler = AngleHandler()

        # LiDAR rate limiting (1000Hz max)
        self.lidar_min_interval = 0.001  # 1ms minimum between reads
        self.last_lidar_read = 0

        # Balanced optimization parameters
        self.scan_radius_az = 8.0  # Reasonable scan radius
        self.scan_radius_el = 8.0
        self.scan_points = 5  # Balanced number of scan points
        self.min_strength_threshold = 80

        # Prevent getting stuck on high strength targets
        self.max_strength_lock = 300  # Don't lock on targets above this
        self.strength_decay_factor = 0.95  # Decay factor for high strength memory
        self.last_high_strength = 0

        # Movement parameters
        self.movement_timeout = 0.5  # Increased timeout for acquisition
        self.position_tolerance = 2.5  # Degrees
        self.min_movement_delay = 0.002  # Minimum delay for movement

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.target_velocity_az = 0.0
        self.target_velocity_el = 0.0
        self.last_update_time = None

        # History tracking
        self.target_history = deque(maxlen=3)
        self.position_history = deque(maxlen=4)
        self.lost_target_count = 0
        self.max_lost_count = 3

        # Performance monitoring
        self.cycle_count = 0
        self.total_cycle_time = 0
        self.lidar_reads = 0

        print("[Tracker] Rate-limited tracker initialized")
        print(f"[Tracker] LiDAR polling: max 1000Hz (1ms intervals)")
        print(f"[Tracker] Scan pattern: {self.scan_points} points, ±{self.scan_radius_az}°")

    def wait_for_lidar_ready(self):
        """Ensure we don't exceed 1000Hz LiDAR polling rate."""
        elapsed = time.time() - self.last_lidar_read
        if elapsed < self.lidar_min_interval:
            time.sleep(self.lidar_min_interval - elapsed)
        self.last_lidar_read = time.time()

    def read_lidar_safe(self):
        """Read LiDAR data with rate limiting."""
        self.wait_for_lidar_ready()

        with self.shared_data["lidar_data"].get_lock():
            distance = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]

        self.lidar_reads += 1

        # Apply strength decay to prevent getting stuck
        if strength > self.max_strength_lock:
            if strength > self.last_high_strength:
                self.last_high_strength = strength
            else:
                # Decay the memory of high strength to prevent sticking
                self.last_high_strength *= self.strength_decay_factor
                strength = min(strength, self.last_high_strength)

        return distance, strength

    def move_to_position(self, azimuth, elevation, wait=True):
        """Move to position with proper angle handling."""
        if self.shared_data["shutdown"].value:
            return False

        # Get current position
        current_az = self.shared_data["stepper_degrees"].value

        # Use shortest path for azimuth
        target_az = self.angle_handler.shortest_path(current_az, azimuth)

        # Set targets
        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = elevation
        self.shared_data["go_to_target"].value = True

        if not wait:
            time.sleep(self.min_movement_delay)
            return True

        # Wait for movement with timeout
        start_time = time.time()
        while time.time() - start_time < self.movement_timeout:
            if self.shared_data["shutdown"].value:
                return False

            # The motor controller should set go_to_target to False when done
            if not self.shared_data["go_to_target"].value:
                return True

            time.sleep(0.01)

        # Timeout occurred
        print("[Tracker] Warning: move_to_position timed out.")
        return False

    def scan_pattern(self, center_az, center_el):
        """Scan in a pattern around center point with rate limiting."""
        scan_results = []
        scan_start = time.time()

        # Generate scan points in a logical order
        scan_sequence = []
        scan_sequence.append((center_az, center_el))  # Add center point first

        # Add circle points
        for i in range(self.scan_points):
            angle = (2 * math.pi * i) / self.scan_points
            scan_az = center_az + self.scan_radius_az * math.cos(angle)
            scan_el = center_el + self.scan_radius_el * math.sin(angle)
            scan_el = np.clip(scan_el, 0, 90)
            scan_az = self.angle_handler.normalize(scan_az)
            scan_sequence.append((scan_az, scan_el))

        # Scan each point
        for scan_az, scan_el in scan_sequence:
            if self.shared_data["shutdown"].value:
                break

            # Don't spend too long scanning
            if time.time() - scan_start > 0.4:
                break

            # Move to scan point
            self.move_to_position(scan_az, scan_el, wait=False)
            time.sleep(0.01)  # Small delay for movement to start

            # Read LiDAR with rate limiting
            distance, strength = self.read_lidar_safe()

            if distance > 0:
                actual_az = self.shared_data["stepper_degrees"].value
                actual_el = self.shared_data["servo_degrees"].value

                if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                    if strength >= self.min_strength_threshold:
                        scan_results.append((actual_az, actual_el, distance, strength))

        return scan_results

    def find_best_target(self, scan_results):
        """Find the best target from scan results."""
        if not scan_results:
            return None

        def score_target(target):
            az, el, dist, strength = target
            score = strength
            if strength > 400:  # Penalize extremely high strength to avoid getting stuck
                score *= 0.9
            return score

        best = max(scan_results, key=score_target)
        return best

    def smooth_position(self, new_az, new_el):
        """Smooth position with proper angle wraparound handling."""
        self.position_history.append((new_az, new_el))

        if len(self.position_history) < 2:
            return new_az, new_el

        az_values = [p[0] for p in self.position_history]
        near_boundary = any(self.angle_handler.is_near_boundary(az) for az in az_values)

        if near_boundary:
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            weights = np.linspace(0.3, 1.0, len(self.position_history))
            weights /= weights.sum()
            smooth_az = sum(az * w for (az, _), w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        el_values = [p[1] for p in self.position_history]
        smooth_el = np.average(el_values, weights=weights)

        return smooth_az, smooth_el

    def update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update satellite points in shared memory."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()
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

    def predict_position(self):
        """Simple position prediction based on velocity."""
        if self.last_update_time is None:
            return self.current_target_az, self.current_target_el

        dt = min(time.time() - self.last_update_time, 0.1)  # Cap prediction time
        pred_az = self.current_target_az + self.target_velocity_az * dt * 0.5
        pred_el = self.current_target_el + self.target_velocity_el * dt * 0.5

        pred_az = self.angle_handler.normalize(pred_az)
        pred_el = np.clip(pred_el, 0, 90)

        return pred_az, pred_el

    def update_velocity(self, new_az, new_el):
        """Update velocity estimates."""
        if self.last_update_time is not None:
            dt = time.time() - self.last_update_time
            if dt > 0 and dt < 1.0:
                az_diff = self.angle_handler.difference(self.current_target_az, new_az)
                new_vel_az = az_diff / dt
                new_vel_el = (new_el - self.current_target_el) / dt

                alpha = 0.3
                self.target_velocity_az = alpha * new_vel_az + (1 - alpha) * self.target_velocity_az
                self.target_velocity_el = alpha * new_vel_el + (1 - alpha) * self.target_velocity_el

                max_vel = 50.0  # degrees/second
                self.target_velocity_az = np.clip(self.target_velocity_az, -max_vel, max_vel)
                self.target_velocity_el = np.clip(self.target_velocity_el, -max_vel, max_vel)

    # ### NEW METHOD for Point Acquisition ###
    def acquire_first_point(self):
        """Perform a wide scan to find the first valid target."""
        print("[Tracker] Starting point acquisition scan...")
        all_scan_results = []

        # Define a wide scan area (e.g., 180 degrees)
        # Scan at a few different elevation levels
        for el in [30, 45, 60]:
            if self.shared_data["shutdown"].value: return False

            # Scan across a wide azimuth range
            for az in range(90, 271, 15):  # Scan from 90 to 270 degrees
                if self.shared_data["shutdown"].value: return False

                if self.move_to_position(az, el, wait=True):
                    distance, strength = self.read_lidar_safe()

                    if distance > 0 and self.clutter_filter.is_valid_target(az, el, distance, strength):
                        if strength >= self.min_strength_threshold:
                            print(f"[Acquire] Found potential target at ({az}°, {el}°), str={strength}")
                            all_scan_results.append((az, el, distance, strength))
                else:
                    print(f"[Acquire] Failed to move to ({az}, {el})")

        if not all_scan_results:
            print("[Acquire] No valid targets found in wide scan.")
            self.shared_data["acquire_points"].value = False  # Reset flag
            return False

        # Find the best target from the wide scan
        initial_target = self.find_best_target(all_scan_results)
        if not initial_target:
            self.shared_data["acquire_points"].value = False  # Reset flag
            return False

        print(
            f"[Acquire] Best initial target at ({initial_target[0]:.1f}°, {initial_target[1]:.1f}°). Engaging tracker.")

        # Move to the best initial target's position to start a fine scan
        self.move_to_position(initial_target[0], initial_target[1], wait=True)

        # Now perform a local, fine-grained scan to lock on
        fine_scan_results = self.scan_pattern(initial_target[0], initial_target[1])
        final_target = self.find_best_target(fine_scan_results)

        if final_target:
            print(f"[Acquire] Lock confirmed at ({final_target[0]:.1f}°, {final_target[1]:.1f}°)")
            # Set this as the current tracked target
            self.current_target_az, self.current_target_el = self.smooth_position(final_target[0], final_target[1])
            self.last_update_time = time.time()
            self.update_satellite_points(self.current_target_az, self.current_target_el, final_target[2],
                                         final_target[3])

            # ### SET TARGET REACHED FLAG ###
            self.shared_data["target_reached"].value = True

            # Switch off the acquisition flag, as the job is done
            self.shared_data["acquire_points"].value = False
            return True
        else:
            print("[Acquire] Failed to confirm target with fine scan.")
            self.shared_data["acquire_points"].value = False  # Reset flag
            return False

    def run(self):
        """Main tracking loop with state management for different modes."""
        print("[Tracker] Starting tracking loop")

        try:
            while not self.shared_data["shutdown"].value:

                # ### NEW: Point Acquisition Mode ###
                if self.shared_data["acquire_points"].value:
                    # Clear any previous tracking state
                    self.current_target_az = None
                    self.current_target_el = None
                    self.clear_satellite_points()
                    self.shared_data["target_reached"].value = False

                    # Run the acquisition process
                    self.acquire_first_point()

                    # After attempting acquisition, wait for the flag to be cleared or changed
                    while self.shared_data["acquire_points"].value and not self.shared_data["shutdown"].value:
                        time.sleep(0.1)
                    continue

                # ### Existing: Continuous Tracking Mode ###
                if not self.shared_data["debug_mode"].value:
                    if self.current_target_az is not None:
                        print("[Tracker] Continuous tracking disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.clear_satellite_points()
                    time.sleep(0.05)
                    continue

                if self.current_target_az is None:
                    # Initialize tracking at the current position or a default start
                    self.current_target_az = self.shared_data["stepper_degrees"].value
                    self.current_target_el = self.shared_data["servo_degrees"].value
                    self.last_update_time = time.time()
                    if self.current_target_az == 0 and self.current_target_el == 0:
                        self.current_target_az = 180.0
                        self.current_target_el = 45.0
                    print(
                        f"[Tracker] Starting continuous tracking at ({self.current_target_az:.1f}°, {self.current_target_el:.1f}°)")

                cycle_start = time.time()

                pred_az, pred_el = self.predict_position()
                scan_results = self.scan_pattern(pred_az, pred_el)
                best_target = self.find_best_target(scan_results)

                if best_target:
                    self.lost_target_count = 0
                    if self.scan_radius_az > 10:
                        self.scan_radius_az = 8.0
                        self.scan_radius_el = 8.0

                    smooth_az, smooth_el = self.smooth_position(best_target[0], best_target[1])
                    self.update_velocity(smooth_az, smooth_el)
                    self.current_target_az = smooth_az
                    self.current_target_el = smooth_el
                    self.last_update_time = time.time()

                    self.update_satellite_points(smooth_az, smooth_el, best_target[2], best_target[3])
                    self.move_to_position(smooth_az, smooth_el, wait=False)
                else:
                    self.lost_target_count += 1
                    print(f"[Tracker] Target lost ({self.lost_target_count}/{self.max_lost_count})")
                    self.clear_satellite_points()

                    if self.lost_target_count >= self.max_lost_count:
                        self.scan_radius_az = min(self.scan_radius_az * 1.5, 25.0)
                        self.scan_radius_el = min(self.scan_radius_el * 1.5, 20.0)
                        self.lost_target_count = 0
                        print(f"[Tracker] Expanding search to ±{self.scan_radius_az:.1f}°")

                cycle_time = time.time() - cycle_start
                self.cycle_count += 1
                self.total_cycle_time += cycle_time

                if self.cycle_count % 50 == 0:
                    avg_cycle = self.total_cycle_time / self.cycle_count
                    lidar_rate = self.lidar_reads / self.total_cycle_time
                    print(f"[Tracker] Stats: {avg_cycle * 1000:.1f}ms/cycle, {lidar_rate:.0f}Hz LiDAR rate")

                if cycle_time < 0.02:
                    time.sleep(0.02 - cycle_time)

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
    """Run the rate-limited tracker process."""
    print("[Tracker] Initializing rate-limited tracker...")
    tracker = TargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
    finally:
        print("[Tracker] Process ended")