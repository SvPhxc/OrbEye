#!/usr/bin/env python3
"""
Refined High-Performance Target Tracker
Focus on smooth, reactive tracking without over-engineering:
- Smooth movement with proper damping
- Conservative prediction only when tracking
- Search mode vs tracking mode separation
- Reduced data noise and jitter
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math
import queue
from collections import deque


class SmoothMovementController:
    """
    Controls smooth movement with proper damping to prevent spasming.
    """

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.movement_damping = 0.3  # Smooth out rapid movements
        self.last_target_az = None
        self.last_target_el = None
        self.movement_threshold = 1.0  # Only move if change is significant

    def smooth_move_to(self, target_az, target_el, force_move=False):
        """
        Move smoothly to target with damping to prevent spasming.
        """
        if self.shared_data["shutdown"].value:
            return

        current_az = self.shared_data["stepper_degrees"].value
        current_el = self.shared_data["servo_degrees"].value

        # Calculate shortest path for azimuth
        target_az = self._calculate_shortest_path(current_az, target_az)

        # Apply damping if we have previous targets
        if self.last_target_az is not None and not force_move:
            # Smooth the movement using damping
            damped_az = self.last_target_az + self.movement_damping * (target_az - self.last_target_az)
            damped_el = self.last_target_el + self.movement_damping * (target_el - self.last_target_el)

            # Check if movement is significant enough
            az_change = abs(self._angle_difference(current_az, damped_az))
            el_change = abs(current_el - damped_el)

            if az_change < self.movement_threshold and el_change < self.movement_threshold and not force_move:
                return  # Skip tiny movements to prevent spasming

            target_az, target_el = damped_az, damped_el

        # Store for next damping calculation
        self.last_target_az = target_az
        self.last_target_el = target_el

        # Execute movement
        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = elevation
        self.shared_data["go_to_target"].value = True

        # Wait for movement with reasonable timeout
        timeout = time.time() + 0.4
        while self.shared_data["go_to_target"].value and time.time() < timeout:
            if self.shared_data["shutdown"].value:
                break
            time.sleep(0.001)

    def _calculate_shortest_path(self, current_az, target_az):
        """Calculate shortest azimuth path."""
        current_az = current_az % 360
        target_az = target_az % 360
        diff = self._angle_difference(current_az, target_az)
        return (current_az + diff) % 360

    def _angle_difference(self, angle1, angle2):
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


class ReactivePredictor:
    """
    Conservative predictor that only makes small adjustments when tracking.
    """

    def __init__(self, max_history=5):
        self.position_history = deque(maxlen=max_history)
        self.min_samples = 3
        self.max_prediction_time = 0.05  # Very short prediction

    def add_position(self, azimuth, elevation):
        """Add position with timestamp."""
        self.position_history.append({
            'timestamp': time.time(),
            'azimuth': azimuth,
            'elevation': elevation
        })

    def get_reactive_adjustment(self):
        """
        Get small reactive adjustment based on recent movement.
        Returns small adjustment angles, not full prediction.
        """
        if len(self.position_history) < self.min_samples:
            return 0.0, 0.0, 0.0  # No adjustment

        # Look at very recent movement only
        recent = list(self.position_history)[-3:]

        if len(recent) < 2:
            return 0.0, 0.0, 0.0

        # Calculate simple velocity from last two points
        dt = recent[-1]['timestamp'] - recent[-2]['timestamp']
        if dt <= 0 or dt > 0.1:  # Ignore if too old or too fast
            return 0.0, 0.0, 0.0

        az_vel = self._angle_difference(recent[-2]['azimuth'], recent[-1]['azimuth']) / dt
        el_vel = (recent[-1]['elevation'] - recent[-2]['elevation']) / dt

        # Very conservative adjustment - only small corrections
        max_adjustment = 2.0  # Max 2 degrees adjustment

        az_adjustment = max(-max_adjustment, min(max_adjustment, az_vel * self.max_prediction_time))
        el_adjustment = max(-max_adjustment, min(max_adjustment, el_vel * self.max_prediction_time))

        # Calculate confidence based on consistency
        if len(recent) >= 3:
            dt2 = recent[-2]['timestamp'] - recent[-3]['timestamp']
            if dt2 > 0:
                az_vel2 = self._angle_difference(recent[-3]['azimuth'], recent[-2]['azimuth']) / dt2
                el_vel2 = (recent[-2]['elevation'] - recent[-3]['elevation']) / dt2

                # Check consistency
                az_consistency = 1.0 - min(1.0, abs(az_vel - az_vel2) / 10.0)
                el_consistency = 1.0 - min(1.0, abs(el_vel - el_vel2) / 10.0)
                confidence = (az_consistency + el_consistency) / 2.0
            else:
                confidence = 0.5
        else:
            confidence = 0.3

        # Only return adjustment if we're confident
        if confidence < 0.4:
            return 0.0, 0.0, confidence

        return az_adjustment, el_adjustment, confidence

    def _angle_difference(self, angle1, angle2):
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


class RefinedTargetTracker:
    """
    Refined tracker focused on smooth, reactive performance.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data

        # Initialize components
        self.movement_controller = SmoothMovementController(shared_data)
        self.predictor = ReactivePredictor()

        # Load clutter filter
        self.clutter_filter = self._init_clutter_filter(background_file)

        # Tracking modes
        self.tracking_mode = "search"  # "search" or "tracking"
        self.target_found = False

        # Search parameters
        self.search_scan_points = 12
        self.search_radius_az = 15.0
        self.search_radius_el = 10.0

        # Tracking parameters
        self.track_scan_points = 6
        self.track_radius_az = 6.0
        self.track_radius_el = 6.0

        # Thresholds
        self.min_strength_threshold = 80
        self.strong_target_threshold = 150
        self.sample_time = 0.002

        # State
        self.current_target_az = None
        self.current_target_el = None
        self.lost_target_count = 0
        self.max_lost_count = 3
        self.tracking_confidence = 0.0

        print("[RefinedTracker] Initialized with smooth movement control")

    def _init_clutter_filter(self, background_file):
        """Initialize simple clutter filter."""
        try:
            background_data = np.load(background_file)
            coords = background_data[:, [0, 1]]
            background_tree = cKDTree(coords)

            def is_valid_target(azimuth, elevation, distance, strength):
                try:
                    query_point = np.array([azimuth, elevation])
                    angular_dist, idx = background_tree.query(query_point, k=1)

                    if angular_dist < 1.0:
                        bg_distance = background_data[idx, 2]
                        return distance < (bg_distance - 60.0)
                    return True
                except:
                    return True

            print(f"[RefinedTracker] Loaded clutter filter")
            return is_valid_target

        except:
            print("[RefinedTracker] No clutter filter available")
            return lambda az, el, dist, strength: True

    def search_for_target(self):
        """
        Initial search mode - wider scan to find targets.
        Only runs when no target is being tracked.
        """
        print("[RefinedTracker] Searching for target...")

        # Start from current position
        center_az = self.shared_data["stepper_degrees"].value or 180.0
        center_el = self.shared_data["servo_degrees"].value or 45.0

        scan_results = []

        for i in range(self.search_scan_points):
            if (self.shared_data["shutdown"].value or
                    not self.shared_data["debug_mode"].value):
                break

            # Calculate scan position
            angle = (2 * math.pi * i) / self.search_scan_points
            scan_az = center_az + self.search_radius_az * math.cos(angle)
            scan_el = center_el + self.search_radius_el * math.sin(angle)
            scan_el = max(0, min(90, scan_el))

            # Move to position
            self.movement_controller.smooth_move_to(scan_az, scan_el, force_move=True)

            # Take sample
            sample = self._get_sample()
            if sample:
                az, el, dist, strength = sample
                if (self.clutter_filter(az, el, dist, strength) and
                        strength >= self.min_strength_threshold):
                    scan_results.append(sample)

                    # If we find a strong target, start tracking immediately
                    if strength >= self.strong_target_threshold:
                        print(f"[RefinedTracker] Strong target found during search: {strength}")
                        break

        return scan_results

    def track_target(self, center_az, center_el):
        """
        Precise tracking mode - small adjustments around known target.
        """
        scan_results = []

        # Get reactive adjustment
        az_adj, el_adj, confidence = self.predictor.get_reactive_adjustment()

        # Apply small predictive adjustment if confident
        if confidence > 0.5:
            center_az += az_adj
            center_el += el_adj
            center_el = max(0, min(90, center_el))
            print(f"[RefinedTracker] Applying reactive adjustment: {az_adj:.1f}°, {el_adj:.1f}°")

        # Small, precise scan around target
        for i in range(self.track_scan_points):
            if (self.shared_data["shutdown"].value or
                    not self.shared_data["debug_mode"].value):
                break

            angle = (2 * math.pi * i) / self.track_scan_points
            scan_az = center_az + self.track_radius_az * math.cos(angle)
            scan_el = center_el + self.track_radius_el * math.sin(angle)
            scan_el = max(0, min(90, scan_el))

            # Smooth movement in tracking mode
            self.movement_controller.smooth_move_to(scan_az, scan_el)

            # Take sample
            sample = self._get_sample()
            if sample:
                az, el, dist, strength = sample
                if (self.clutter_filter(az, el, dist, strength) and
                        strength >= self.min_strength_threshold):
                    scan_results.append(sample)

                    # Early exit for strong targets
                    if strength >= self.strong_target_threshold:
                        break

        return scan_results

    def _get_sample(self):
        """Get a single LiDAR sample."""
        time.sleep(self.sample_time)  # Brief settling time

        try:
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            with self.shared_data["lidar_data"].get_lock():
                dist = self.shared_data["lidar_data"][0]
                strength = self.shared_data["lidar_data"][1]

            if dist > 0:
                return (current_az, current_el, dist, strength)
        except:
            pass
        return None

    def find_best_target(self, scan_results):
        """Find best target with simple scoring."""
        if not scan_results:
            return None

        # Simple: just pick strongest target
        best = max(scan_results, key=lambda x: x[3])

        print(f"[RefinedTracker] Best target: ({best[0]:.1f}°, {best[1]:.1f}°) "
              f"dist={best[2]:.0f}cm, str={best[3]}")

        # Update satellite points
        self._update_satellite_points(best[0], best[1], best[2], best[3])

        return best

    def _update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update satellite points."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()
        except Exception as e:
            print(f"[RefinedTracker] Error updating satellite_points: {e}")

    def _clear_satellite_points(self):
        """Clear satellite points."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
        except:
            pass

    def run(self):
        """
        Main tracking loop with proper search/track mode separation.
        """
        print("[RefinedTracker] Starting refined tracker")

        try:
            while not self.shared_data["shutdown"].value:
                if self.shared_data["debug_mode"].value:
                    cycle_start = time.time()

                    if self.tracking_mode == "search":
                        # Search mode - look for targets
                        scan_results = self.search_for_target()
                        best_target = self.find_best_target(scan_results)

                        if best_target:
                            # Found target - switch to tracking mode
                            self.tracking_mode = "tracking"
                            self.target_found = True
                            self.current_target_az = best_target[0]
                            self.current_target_el = best_target[1]
                            self.lost_target_count = 0

                            # Add to predictor
                            self.predictor.add_position(best_target[0], best_target[1])

                            print(f"[RefinedTracker] Target acquired! Switching to tracking mode")

                            # Move to target smoothly
                            self.movement_controller.smooth_move_to(
                                self.current_target_az, self.current_target_el, force_move=True)
                        else:
                            print("[RefinedTracker] No target found in search")
                            time.sleep(0.01)  # Pause between search attempts

                    elif self.tracking_mode == "tracking":
                        # Tracking mode - precise tracking around known target
                        scan_results = self.track_target(self.current_target_az, self.current_target_el)
                        best_target = self.find_best_target(scan_results)

                        if best_target:
                            # Continue tracking
                            self.lost_target_count = 0
                            self.current_target_az = best_target[0]
                            self.current_target_el = best_target[1]

                            # Add to predictor for reactive adjustments
                            self.predictor.add_position(best_target[0], best_target[1])

                            # Move to target with smooth control
                            self.movement_controller.smooth_move_to(
                                self.current_target_az, self.current_target_el)

                        else:
                            # Lost target
                            self.lost_target_count += 1
                            print(f"[RefinedTracker] Target lost ({self.lost_target_count}/{self.max_lost_count})")

                            self._clear_satellite_points()

                            if self.lost_target_count >= self.max_lost_count:
                                # Switch back to search mode
                                print("[RefinedTracker] Switching back to search mode")
                                self.tracking_mode = "search"
                                self.target_found = False
                                self.lost_target_count = 0
                                self.predictor = ReactivePredictor()  # Reset predictor

                    # Performance monitoring
                    cycle_time = time.time() - cycle_start
                    if cycle_time > 0.5:
                        print(f"[RefinedTracker] Slow cycle: {cycle_time:.2f}s")

                else:
                    # Debug mode disabled
                    if self.target_found:
                        print("[RefinedTracker] Debug mode disabled")
                        self.tracking_mode = "search"
                        self.target_found = False
                        self.current_target_az = None
                        self.current_target_el = None
                        self.lost_target_count = 0
                        self.predictor = ReactivePredictor()
                        self._clear_satellite_points()

                    time.sleep(0.05)

                # Small delay
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("[RefinedTracker] Interrupted")
        except Exception as e:
            print(f"[RefinedTracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[RefinedTracker] Shutting down")


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """
    Run the refined target tracker process.
    """
    print("[RefinedTracker] Initializing refined tracker process...")
    tracker = RefinedTargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[RefinedTracker] Process error: {e}")
    finally:
        print("[RefinedTracker] Refined process ended")