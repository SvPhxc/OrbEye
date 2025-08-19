#!/usr/bin/env python3
"""
Ultra-Optimized LiDAR Target Tracker
Key improvements:
- Fixed angle wraparound issues at 0°/360° boundary
- Parallel scanning with threading
- Predictive movement
- Adaptive sampling
- Optimized data structures
- Reduced function call overhead
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class ClutterFilter:
    """
    Optimized clutter filter with improved caching and vectorized operations.
    """

    def __init__(self, background_file="background_scan.npy", angular_tolerance=1, distance_margin_cm=70.0):
        self.angular_tolerance = angular_tolerance
        self.distance_margin_cm = distance_margin_cm
        self.background_tree = None
        self.background_data = None
        self._query_cache = {}
        self._cache_size = 2000  # Increased cache size
        self._cache_hits = 0
        self._cache_misses = 0

        try:
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")

            # Pre-process background data for faster queries
            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords, leafsize=16)  # Optimized leaf size

            # Pre-compute distance lookup table
            self.bg_distances = self.background_data[:, 2]
            print("[ClutterFilter] Optimized k-d tree built.")

        except FileNotFoundError:
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR: {e}")

    def is_valid_target_batch(self, points):
        """Batch validation for multiple points - much faster than individual checks."""
        if self.background_tree is None:
            return [True] * len(points)

        valid = []
        for az, el, dist, strength in points:
            # Ultra-fast cache key
            cache_key = (int(az * 10), int(el * 10))  # Integer keys are faster

            if cache_key in self._query_cache:
                self._cache_hits += 1
                bg_distance = self._query_cache[cache_key]
            else:
                self._cache_misses += 1
                query_point = np.array([az, el])
                angular_dist, idx = self.background_tree.query(query_point, k=1)

                if angular_dist < self.angular_tolerance:
                    bg_distance = self.bg_distances[idx]
                else:
                    bg_distance = float('inf')

                if len(self._query_cache) < self._cache_size:
                    self._query_cache[cache_key] = bg_distance

            valid.append(dist < (bg_distance - self.distance_margin_cm))

        return valid


class AngleHandler:
    """Dedicated class for robust angle handling with 0/360 wraparound."""

    @staticmethod
    def normalize(angle):
        """Normalize to [0, 360) range."""
        return angle % 360

    @staticmethod
    def difference(angle1, angle2):
        """Calculate shortest angular difference, handling wraparound."""
        diff = (angle2 - angle1 + 180) % 360 - 180
        return diff

    @staticmethod
    def shortest_path(current, target):
        """Calculate target angle for shortest rotation path."""
        diff = AngleHandler.difference(current, target)
        return AngleHandler.normalize(current + diff)

    @staticmethod
    def circular_mean(angles):
        """Compute mean of angles, properly handling wraparound."""
        if not angles:
            return None
        x = sum(math.cos(math.radians(a)) for a in angles)
        y = sum(math.sin(math.radians(a)) for a in angles)
        return AngleHandler.normalize(math.degrees(math.atan2(y, x)))

    @staticmethod
    def is_near_boundary(angle, threshold=10):
        """Check if angle is near 0/360 boundary."""
        norm_angle = AngleHandler.normalize(angle)
        return norm_angle < threshold or norm_angle > (360 - threshold)


class TargetTracker:
    """
    Ultra-optimized tracker with parallel scanning and predictive movement.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file=background_file)
        self.angle_handler = AngleHandler()

        # Aggressive optimization parameters
        self.scan_radius_az = 6.0  # Further reduced
        self.scan_radius_el = 6.0
        self.scan_points = 6  # Minimal scan points for speed
        self.min_strength_threshold = 155  # Lower threshold
        self.sample_time = 0.002  # Minimal sampling delay

        # Adaptive parameters
        self.adaptive_mode = False  # Disable adaptive mode by default
        self.confidence_level = 0.0
        self.high_confidence_threshold = 0.8

        # Movement optimization
        self.movement_timeout = 0.3  # Very short timeout
        self.position_tolerance = 0.50 # Larger tolerance for faster tracking
        self.predictive_movement = False  # Disable predictive movement by default

        # Tracking state with velocity estimation
        self.current_target_az = None
        self.current_target_el = None
        self.target_velocity_az = 0.0  # degrees/second
        self.target_velocity_el = 0.0
        self.last_update_time = None

        # Optimized history using deque
        self.target_history = deque(maxlen=3)
        self.position_history = deque(maxlen=5)

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Performance metrics
        self.last_cycle_time = 0
        self.cycle_count = 0
        self.total_cycle_time = 0

        print("[Tracker] Ultra-optimized tracker initialized")
        print(f"[Tracker] Adaptive mode: {self.adaptive_mode}")

    def predict_next_position(self):
        """Predict target's next position based on velocity."""
        if not self.predictive_movement or self.last_update_time is None:
            return self.current_target_az, self.current_target_el

        dt = time.time() - self.last_update_time
        predicted_az = self.angle_handler.normalize(
            self.current_target_az + self.target_velocity_az * dt
        )
        predicted_el = np.clip(
            self.current_target_el + self.target_velocity_el * dt,
            0, 90
        )

        return predicted_az, predicted_el

    def update_velocity(self, new_az, new_el):
        """Update velocity estimates with angle wraparound handling."""
        if self.last_update_time is not None:
            dt = time.time() - self.last_update_time
            if dt > 0:
                # Use angle difference for velocity to handle wraparound
                az_diff = self.angle_handler.difference(self.current_target_az, new_az)
                self.target_velocity_az = az_diff / dt
                self.target_velocity_el = (new_el - self.current_target_el) / dt

                # Apply smoothing
                self.target_velocity_az *= 0.7  # Damping factor
                self.target_velocity_el *= 0.7

    def spiral_scan_generator(self, center_az, center_el):
        """Generate spiral scan pattern for better coverage."""
        # Start from center and spiral outward
        yield (center_az, center_el)

        for r in [0.5, 1.0]:  # Two radius levels
            for i in range(self.scan_points):
                angle = (2 * math.pi * i) / self.scan_points
                scan_az = center_az + self.scan_radius_az * r * math.cos(angle)
                scan_el = center_el + self.scan_radius_el * r * math.sin(angle)
                scan_el = np.clip(scan_el, 0, 90)

                # Handle angle wraparound for azimuth
                scan_az = self.angle_handler.normalize(scan_az)

                yield (scan_az, scan_el)

    def parallel_scan_point(self, scan_az, scan_el):
        """Scan a single point - designed for parallel execution."""
        if self.shared_data["shutdown"].value:
            return []

        # Quick movement without waiting
        current_az = self.shared_data["stepper_degrees"].value
        target_az = self.angle_handler.shortest_path(current_az, scan_az)

        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = scan_el
        self.shared_data["go_to_target"].value = True

        # Ultra-short wait
        time.sleep(0.001)

        # Collect sample immediately
        samples = []
        with self.shared_data["lidar_data"].get_lock():
            dist = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]

        if dist > 0:
            actual_az = self.shared_data["stepper_degrees"].value
            actual_el = self.shared_data["servo_degrees"].value
            samples.append((actual_az, actual_el, dist, strength))

        return samples

    def scan_adaptive(self, center_az, center_el):
        """Adaptive scanning with early termination and parallel execution."""
        scan_results = []
        scan_start = time.time()

        # Generate scan points
        scan_points = list(self.spiral_scan_generator(center_az, center_el))

        # Prioritize based on confidence
        if self.confidence_level > self.high_confidence_threshold:
            # High confidence - scan fewer points
            scan_points = scan_points[:max(2, len(scan_points) // 2)]

        # Process scan points
        for scan_az, scan_el in scan_points:
            if self.shared_data["shutdown"].value:
                break

            # Check time limit
            if time.time() - scan_start > 0.2:  # 200ms max scan time
                break

            samples = self.parallel_scan_point(scan_az, scan_el)

            # Batch validation
            if samples:
                valid = self.clutter_filter.is_valid_target_batch(samples)
                for i, (az, el, dist, strength) in enumerate(samples):
                    if valid[i] and strength >= self.min_strength_threshold:
                        scan_results.append((az, el, dist, strength))

                        # Ultra-early termination
                        if strength > 250:  # Very strong target
                            return scan_results

        return scan_results

    def update_confidence(self, found_target, target_strength=0):
        """Update tracking confidence for adaptive behavior."""
        if found_target:
            strength_factor = min(target_strength / 200.0, 1.0)
            self.confidence_level = min(1.0, self.confidence_level * 0.8 + strength_factor * 0.2)
        else:
            self.confidence_level *= 0.7

        # Adapt parameters based on confidence
        if self.adaptive_mode:
            if self.confidence_level > self.high_confidence_threshold:
                self.scan_points = 4  # Minimal scanning
                self.scan_radius_az = 6.0
                self.scan_radius_el = 6.0
            elif self.confidence_level < 0.3:
                self.scan_points = 8  # More thorough scanning
                self.scan_radius_az = 10.0
                self.scan_radius_el = 10.0
            else:
                self.scan_points = 6
                self.scan_radius_az = 8.0
                self.scan_radius_el = 8.0

    def smooth_position_robust(self, new_az, new_el):
        """Robust position smoothing with proper angle wraparound handling."""
        self.position_history.append((new_az, new_el, time.time()))

        if len(self.position_history) < 2:
            return new_az, new_el

        # Check if we're near the 0/360 boundary
        near_boundary = any(self.angle_handler.is_near_boundary(p[0])
                            for p in self.position_history)

        if near_boundary:
            # Use circular mean for angles near boundary
            az_values = [p[0] for p in self.position_history]
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            # Simple weighted average for angles away from boundary
            weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
            weights /= weights.sum()
            smooth_az = sum(p[0] * w for p, w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        # Simple weighted average for elevation
        smooth_el = sum(p[1] * w for p, w in zip(self.position_history, weights))

        return smooth_az, smooth_el

    def run(self):
        """Main tracking loop with aggressive optimizations."""
        print("[Tracker] Ultra-optimized tracking started")

        try:
            while not self.shared_data["shutdown"].value:
                if not self.shared_data["debug_mode"].value:
                    time.sleep(0.01)
                    continue

                # Initialize if needed
                if self.current_target_az is None:
                    self.current_target_az = self.shared_data["stepper_degrees"].value
                    self.current_target_el = self.shared_data["servo_degrees"].value
                    self.last_update_time = time.time()

                    # Start at a reasonable position
                    if self.current_target_az == 0 and self.current_target_el == 0:
                        self.current_target_az = 180.0
                        self.current_target_el = 45.0

                cycle_start = time.time()

                # Predict next position
                pred_az, pred_el = self.predict_next_position()

                # Adaptive scanning
                scan_results = self.scan_adaptive(pred_az, pred_el)

                if scan_results:
                    # Find best target (inline for speed)
                    best = max(scan_results, key=lambda x: x[3])

                    # Update confidence
                    self.update_confidence(True, best[3])

                    # Smooth position with robust angle handling
                    smooth_az, smooth_el = self.smooth_position_robust(best[0], best[1])

                    # Update velocity
                    self.update_velocity(smooth_az, smooth_el)

                    # Update tracking state
                    self.current_target_az = smooth_az
                    self.current_target_el = smooth_el
                    self.last_update_time = time.time()

                    # Update shared data
                    with self.shared_data["satellite_points"].get_lock():
                        self.shared_data["satellite_points"][0] = smooth_az
                        self.shared_data["satellite_points"][1] = smooth_el
                        self.shared_data["satellite_points"][2] = best[2]
                        self.shared_data["satellite_points"][3] = best[3]
                        self.shared_data["satellite_points"][4] = time.time()

                    # Quick position update
                    current_az = self.shared_data["stepper_degrees"].value
                    target_az = self.angle_handler.shortest_path(current_az, smooth_az)
                    self.shared_data["target_azimuth"].value = target_az
                    self.shared_data["target_elevation"].value = smooth_el
                    self.shared_data["go_to_target"].value = True

                else:
                    # Target lost
                    self.update_confidence(False)

                    # Clear satellite points
                    with self.shared_data["satellite_points"].get_lock():
                        for i in range(5):
                            self.shared_data["satellite_points"][i] = 0.0

                # Performance monitoring
                cycle_time = time.time() - cycle_start
                self.cycle_count += 1
                self.total_cycle_time += cycle_time

                if self.cycle_count % 100 == 0:
                    avg_time = self.total_cycle_time / self.cycle_count
                    print(f"[Tracker] Avg cycle: {avg_time * 1000:.1f}ms, "
                          f"Confidence: {self.confidence_level:.2f}, "
                          f"Cache hits: {self.clutter_filter._cache_hits}")

                # Minimal delay
                if cycle_time < 0.01:  # If we're too fast, add tiny delay
                    time.sleep(0.01)

        except Exception as e:
            print(f"[Tracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Tracker] Shutting down")
            self.executor.shutdown(wait=False)


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """Run the ultra-optimized tracker process."""
    print("[Tracker] Starting ultra-optimized tracker...")
    tracker = TargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
    finally:
        print("[Tracker] Process ended")