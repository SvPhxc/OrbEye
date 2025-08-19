#!/usr/bin/env python3
"""
Optimized LiDAR Target Tracker with Rate Limiting
Key improvements:
- Fixed angle wraparound issues at 0°/360° boundary
- Respects 1000Hz LiDAR polling limit
- Prevents getting stuck on high strength targets
- Adaptive scanning without over-optimization
- Predictive movement with proper timing
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
        self.movement_timeout = 0.3  # Reasonable timeout
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
        timeout = time.time() + self.movement_timeout
        while self.shared_data["go_to_target"].value and time.time() < timeout:
            if self.shared_data["shutdown"].value:
                return False

            # Check if close enough
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            az_diff = abs(self.angle_handler.difference(current_az, target_az))
            el_diff = abs(current_el - elevation)

            if az_diff < self.position_tolerance and el_diff < self.position_tolerance:
                return True

            time.sleep(0.002)

        return True

    def scan_pattern(self, center_az, center_el):
        """Scan in a pattern around center point with rate limiting."""
        scan_results = []
        scan_start = time.time()

        print(f"[Tracker] Scanning around ({center_az:.1f}°, {center_el:.1f}°)")

        # Generate scan points in a logical order
        scan_sequence = []

        # Add center point first
        scan_sequence.append((center_az, center_el))

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
            if self.shared_data["shutdown"].value or not self.shared_data["debug_mode"].value:
                break

            # Don't spend too long scanning
            if time.time() - scan_start > 0.4:
                break

            # Move to scan point
            self.move_to_position(scan_az, scan_el, wait=False)

            # Small delay for movement to start
            time.sleep(0.003)

            # Read LiDAR with rate limiting
            distance, strength = self.read_lidar_safe()

            if distance > 0:
                # Get actual position
                actual_az = self.shared_data["stepper_degrees"].value
                actual_el = self.shared_data["servo_degrees"].value

                # Check if valid target
                if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                    if strength >= self.min_strength_threshold:
                        scan_results.append((actual_az, actual_el, distance, strength))

                        # Don't terminate early on high strength to avoid getting stuck
                        # Just continue scanning normally

        return scan_results

    def find_best_target(self, scan_results):
        """Find the best target from scan results."""
        if not scan_results:
            return None

        # Sort by strength but apply penalties for being too strong (might be stuck)
        def score_target(target):
            az, el, dist, strength = target
            score = strength

            # Penalize extremely high strength slightly to avoid getting stuck
            if strength > 400:
                score *= 0.9

            return score

        best = max(scan_results, key=score_target)

        print(f"[Tracker] Best target: ({best[0]:.1f}°, {best[1]:.1f}°) "
              f"dist={best[2]:.0f}cm, str={best[3]:.0f}")

        return best

    def smooth_position(self, new_az, new_el):
        """Smooth position with proper angle wraparound handling."""
        self.position_history.append((new_az, new_el))

        if len(self.position_history) < 2:
            return new_az, new_el

        # Check if any angles are near the boundary
        az_values = [p[0] for p in self.position_history]
        near_boundary = any(self.angle_handler.is_near_boundary(az) for az in az_values)

        if near_boundary:
            # Use circular mean for angles near boundary
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            # Simple weighted average away from boundary
            weights = np.linspace(0.3, 1.0, len(self.position_history))
            weights /= weights.sum()
            smooth_az = sum(az * w for (az, _), w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        # Simple average for elevation
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

        # Apply velocity with damping
        pred_az = self.current_target_az + self.target_velocity_az * dt * 0.5
        pred_el = self.current_target_el + self.target_velocity_el * dt * 0.5

        pred_az = self.angle_handler.normalize(pred_az)
        pred_el = np.clip(pred_el, 0, 90)

        return pred_az, pred_el

    def update_velocity(self, new_az, new_el):
        """Update velocity estimates."""
        if self.last_update_time is not None:
            dt = time.time() - self.last_update_time
            if dt > 0 and dt < 1.0:  # Ignore large time gaps
                # Use angle difference for azimuth velocity
                az_diff = self.angle_handler.difference(self.current_target_az, new_az)
                new_vel_az = az_diff / dt
                new_vel_el = (new_el - self.current_target_el) / dt

                # Smooth velocity with exponential moving average
                alpha = 0.3
                self.target_velocity_az = alpha * new_vel_az + (1 - alpha) * self.target_velocity_az
                self.target_velocity_el = alpha * new_vel_el + (1 - alpha) * self.target_velocity_el

                # Limit maximum velocity
                max_vel = 50.0  # degrees/second
                self.target_velocity_az = np.clip(self.target_velocity_az, -max_vel, max_vel)
                self.target_velocity_el = np.clip(self.target_velocity_el, -max_vel, max_vel)

    def run(self):
        """Main tracking loop with proper rate limiting."""
        print("[Tracker] Starting rate-limited tracking")

        try:
            while not self.shared_data["shutdown"].value:
                if not self.shared_data["debug_mode"].value:
                    # Not in debug mode
                    if self.current_target_az is not None:
                        print("[Tracker] Debug mode disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.clear_satellite_points()
                    time.sleep(0.05)
                    continue

                # Initialize if needed
                if self.current_target_az is None:
                    self.current_target_az = self.shared_data["stepper_degrees"].value
                    self.current_target_el = self.shared_data["servo_degrees"].value
                    self.last_update_time = time.time()

                    if self.current_target_az == 0 and self.current_target_el == 0:
                        self.current_target_az = 180.0
                        self.current_target_el = 45.0

                    print(f"[Tracker] Starting at ({self.current_target_az:.1f}°, "
                          f"{self.current_target_el:.1f}°)")

                cycle_start = time.time()

                # Predict next position
                pred_az, pred_el = self.predict_position()

                # Scan around predicted position
                scan_results = self.scan_pattern(pred_az, pred_el)

                # Find best target
                best_target = self.find_best_target(scan_results)

                if best_target:
                    # Target found
                    self.lost_target_count = 0

                    # Reset scan radius
                    if self.scan_radius_az > 10:
                        self.scan_radius_az = 8.0
                        self.scan_radius_el = 8.0

                    # Smooth position
                    smooth_az, smooth_el = self.smooth_position(best_target[0], best_target[1])

                    # Update velocity
                    self.update_velocity(smooth_az, smooth_el)

                    # Update target position
                    self.current_target_az = smooth_az
                    self.current_target_el = smooth_el
                    self.last_update_time = time.time()

                    # Update satellite points
                    self.update_satellite_points(smooth_az, smooth_el,
                                                 best_target[2], best_target[3])

                    # Move to smoothed position
                    self.move_to_position(smooth_az, smooth_el, wait=False)

                else:
                    # Target lost
                    self.lost_target_count += 1
                    print(f"[Tracker] Target lost ({self.lost_target_count}/{self.max_lost_count})")

                    self.clear_satellite_points()

                    # Expand search area if consistently lost
                    if self.lost_target_count >= self.max_lost_count:
                        self.scan_radius_az = min(self.scan_radius_az * 1.5, 25.0)
                        self.scan_radius_el = min(self.scan_radius_el * 1.5, 20.0)
                        self.lost_target_count = 0
                        print(f"[Tracker] Expanding search to ±{self.scan_radius_az:.1f}°")

                # Performance monitoring
                cycle_time = time.time() - cycle_start
                self.cycle_count += 1
                self.total_cycle_time += cycle_time

                if self.cycle_count % 50 == 0:
                    avg_cycle = self.total_cycle_time / self.cycle_count
                    lidar_rate = self.lidar_reads / self.total_cycle_time
                    print(f"[Tracker] Stats: {avg_cycle * 1000:.1f}ms/cycle, "
                          f"{lidar_rate:.0f}Hz LiDAR rate")

                # Ensure minimum cycle time to prevent overwhelming the system
                if cycle_time < 0.02:  # Minimum 20ms per cycle
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