#!/usr/bin/env python3
"""
Advanced High-Performance LiDAR Target Tracker
Implements all performance optimizations:
- Async LiDAR data processing
- Predictive tracking with velocity estimation
- Dynamic scan patterns based on distance/confidence
- Adaptive movement precision
- Advanced caching and optimization
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math
import queue
from collections import deque
import statistics


class AsyncLidarProcessor:
    """
    Asynchronous LiDAR data processor that runs in a separate thread
    to collect and buffer samples while the main tracker is moving.
    """

    def __init__(self, shared_data, buffer_size=50):
        self.shared_data = shared_data
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue(maxsize=buffer_size)

    def start(self):
        """Start the async processor thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("[AsyncLidar] Started async LiDAR processor")

    def stop(self):
        """Stop the async processor thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("[AsyncLidar] Stopped async processor")

    def _process_loop(self):
        """Main processing loop - runs in separate thread."""
        while self.running and not self.shared_data["shutdown"].value:
            try:
                # Get current position and LiDAR data
                current_az = self.shared_data["stepper_degrees"].value
                current_el = self.shared_data["servo_degrees"].value

                with self.shared_data["lidar_data"].get_lock():
                    dist = self.shared_data["lidar_data"][0]
                    strength = self.shared_data["lidar_data"][1]

                if dist > 0:  # Valid measurement
                    sample = {
                        'timestamp': time.time(),
                        'azimuth': current_az,
                        'elevation': current_el,
                        'distance': dist,
                        'strength': strength
                    }

                    # Add to buffer
                    self.data_buffer.append(sample)

                    # Also add to queue for immediate access
                    try:
                        self.data_queue.put_nowait(sample)
                    except queue.Full:
                        # Remove oldest if queue is full
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(sample)
                        except queue.Empty:
                            pass

                time.sleep(0.0005)  # Very fast sampling

            except Exception as e:
                print(f"[AsyncLidar] Processing error: {e}")
                time.sleep(0.001)

    def get_recent_samples(self, max_age=0.1, position_tolerance=3.0):
        """
        Get recent samples near a specific position.

        Args:
            max_age: Maximum age of samples in seconds
            position_tolerance: Maximum angular distance from target position

        Returns:
            List of recent valid samples
        """
        current_time = time.time()
        recent_samples = []

        # Get samples from queue first (most recent)
        while not self.data_queue.empty():
            try:
                sample = self.data_queue.get_nowait()
                age = current_time - sample['timestamp']
                if age <= max_age:
                    recent_samples.append(sample)
            except queue.Empty:
                break

        # Also check buffer for additional samples
        for sample in list(self.data_buffer):
            age = current_time - sample['timestamp']
            if age <= max_age and sample not in recent_samples:
                recent_samples.append(sample)

        return recent_samples

    def get_samples_at_position(self, target_az, target_el, tolerance=2.0, max_age=0.05):
        """Get samples near a specific position."""
        recent_samples = self.get_recent_samples(max_age)
        position_samples = []

        for sample in recent_samples:
            az_diff = abs(self._angle_difference(sample['azimuth'], target_az))
            el_diff = abs(sample['elevation'] - target_el)

            if az_diff <= tolerance and el_diff <= tolerance:
                position_samples.append(sample)

        return position_samples

    def _angle_difference(self, angle1, angle2):
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


class VelocityPredictor:
    """
    Predicts target movement based on position history.
    """

    def __init__(self, history_size=10, min_samples=3):
        self.history_size = history_size
        self.min_samples = min_samples
        self.position_history = deque(maxlen=history_size)

    def add_position(self, azimuth, elevation, timestamp=None):
        """Add a new position measurement."""
        if timestamp is None:
            timestamp = time.time()

        self.position_history.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'elevation': elevation
        })

    def predict_position(self, prediction_time=0.1):
        """
        Predict target position after prediction_time seconds.

        Args:
            prediction_time: Time in seconds to predict ahead

        Returns:
            (predicted_az, predicted_el, confidence) or (None, None, 0)
        """
        if len(self.position_history) < self.min_samples:
            return None, None, 0.0

        # Calculate velocities from recent history
        recent_history = list(self.position_history)[-self.min_samples:]

        if len(recent_history) < 2:
            return None, None, 0.0

        # Calculate average velocity
        az_velocities = []
        el_velocities = []

        for i in range(1, len(recent_history)):
            dt = recent_history[i]['timestamp'] - recent_history[i - 1]['timestamp']
            if dt > 0:
                # Handle azimuth wraparound
                az_diff = self._angle_difference(
                    recent_history[i - 1]['azimuth'],
                    recent_history[i]['azimuth']
                )
                az_vel = az_diff / dt

                el_diff = recent_history[i]['elevation'] - recent_history[i - 1]['elevation']
                el_vel = el_diff / dt

                az_velocities.append(az_vel)
                el_velocities.append(el_vel)

        if not az_velocities:
            return None, None, 0.0

        # Average velocity
        avg_az_vel = statistics.mean(az_velocities)
        avg_el_vel = statistics.mean(el_velocities)

        # Calculate velocity consistency (confidence)
        if len(az_velocities) > 1:
            az_std = statistics.stdev(az_velocities)
            el_std = statistics.stdev(el_velocities)

            # Lower standard deviation = higher confidence
            confidence = 1.0 / (1.0 + math.sqrt(az_std ** 2 + el_std ** 2) / 10.0)
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = 0.5

        # Predict future position
        last_pos = recent_history[-1]
        predicted_az = last_pos['azimuth'] + avg_az_vel * prediction_time
        predicted_el = last_pos['elevation'] + avg_el_vel * prediction_time

        # Normalize angles
        predicted_az = predicted_az % 360
        predicted_el = max(0, min(90, predicted_el))

        return predicted_az, predicted_el, confidence

    def _angle_difference(self, angle1, angle2):
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


class AdaptiveScanPattern:
    """
    Dynamic scan patterns that adapt based on target distance, confidence, and movement.
    """

    def __init__(self):
        # Pattern definitions: (points, az_radius, el_radius, description)
        self.patterns = {
            'micro': (4, 3.0, 3.0, "Ultra-fast micro scan"),
            'small': (6, 5.0, 5.0, "Small precision scan"),
            'medium': (8, 8.0, 8.0, "Medium coverage scan"),
            'large': (12, 12.0, 12.0, "Large area scan"),
            'search': (16, 20.0, 15.0, "Wide search pattern")
        }

    def select_pattern(self, distance, confidence, velocity_magnitude, lost_count):
        """
        Select optimal scan pattern based on tracking conditions.

        Args:
            distance: Target distance in cm
            confidence: Tracking confidence (0-1)
            velocity_magnitude: Target velocity magnitude
            lost_count: Number of consecutive lost target counts

        Returns:
            (scan_points, az_radius, el_radius, pattern_name)
        """
        # Default to medium
        pattern_name = 'medium'

        # Close targets with high confidence -> micro scan
        if distance < 200 and confidence > 0.8 and velocity_magnitude < 5.0:
            pattern_name = 'micro'

        # Close targets with good confidence -> small scan
        elif distance < 300 and confidence > 0.6:
            pattern_name = 'small'

        # Far targets or low confidence -> larger scan
        elif distance > 500 or confidence < 0.4:
            pattern_name = 'large'

        # Lost target -> search pattern
        elif lost_count > 0:
            pattern_name = 'search'

        # Fast moving targets -> larger coverage
        elif velocity_magnitude > 10.0:
            pattern_name = 'large'

        pattern = self.patterns[pattern_name]
        return pattern[0], pattern[1], pattern[2], pattern_name


class AdvancedTargetTracker:
    """
    Advanced target tracker with all performance optimizations.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data

        # Initialize components
        self.async_lidar = AsyncLidarProcessor(shared_data)
        self.velocity_predictor = VelocityPredictor()
        self.scan_pattern = AdaptiveScanPattern()

        # Load clutter filter
        self.clutter_filter = self._init_clutter_filter(background_file)

        # Adaptive parameters
        self.min_strength_threshold = 60  # Lower threshold for distant targets
        self.good_target_threshold = 150  # Early termination threshold
        self.movement_step_size = 2.0  # Larger steps for faster movement
        self.position_tolerance = 3.0  # Relaxed positioning tolerance

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.current_distance = None
        self.tracking_confidence = 0.0
        self.lost_target_count = 0
        self.max_lost_count = 2  # Quick response to lost targets

        # Performance optimization
        self.last_scan_time = 0
        self.target_history = []
        self.cycle_times = deque(maxlen=10)  # Track performance

        print("[AdvancedTracker] Initialized with all optimizations")

    def _init_clutter_filter(self, background_file):
        """Initialize optimized clutter filter."""
        try:
            background_data = np.load(background_file)
            coords = background_data[:, [0, 1]]
            background_tree = cKDTree(coords)

            def is_valid_target(azimuth, elevation, distance, strength):
                try:
                    query_point = np.array([azimuth, elevation])
                    angular_dist, idx = background_tree.query(query_point, k=1)

                    if angular_dist < 1.0:  # 1 degree tolerance
                        bg_distance = background_data[idx, 2]
                        return distance < (bg_distance - 50.0)  # 50cm margin
                    return True
                except:
                    return True

            print(f"[AdvancedTracker] Loaded clutter filter with {len(background_data)} points")
            return is_valid_target

        except:
            print("[AdvancedTracker] No clutter filter available")
            return lambda az, el, dist, strength: True

    def start(self):
        """Start the advanced tracker."""
        self.async_lidar.start()

    def stop(self):
        """Stop the advanced tracker."""
        self.async_lidar.stop()

    def angle_difference(self, angle1, angle2):
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def calculate_shortest_path(self, current_az, target_az):
        """Calculate shortest path with adaptive step size."""
        current_az = current_az % 360
        target_az = target_az % 360
        diff = self.angle_difference(current_az, target_az)

        # Use larger steps for distant movements
        if abs(diff) > 30:
            # Large movement - use bigger steps
            step_multiplier = 2.0
        else:
            step_multiplier = 1.0

        return (current_az + diff) % 360, step_multiplier

    def predictive_scan(self, center_az, center_el):
        """
        Advanced scan with prediction, async processing, and adaptive patterns.
        """
        scan_start = time.time()

        # Get prediction if available
        pred_az, pred_el, pred_confidence = self.velocity_predictor.predict_position(0.05)

        if pred_az is not None and pred_confidence > 0.3:
            # Use predicted position as scan center
            print(f"[AdvancedTracker] Using prediction: ({pred_az:.1f}°, {pred_el:.1f}°) "
                  f"confidence={pred_confidence:.2f}")
            center_az, center_el = pred_az, pred_el

        # Calculate velocity magnitude for pattern selection
        velocity_mag = 0.0
        if len(self.target_history) >= 2:
            last_two = self.target_history[-2:]
            dt = last_two[1]['timestamp'] - last_two[0]['timestamp']
            if dt > 0:
                az_diff = self.angle_difference(last_two[0]['azimuth'], last_two[1]['azimuth'])
                el_diff = last_two[1]['elevation'] - last_two[0]['elevation']
                velocity_mag = math.sqrt(az_diff ** 2 + el_diff ** 2) / dt

        # Select adaptive scan pattern
        scan_points, az_radius, el_radius, pattern_name = self.scan_pattern.select_pattern(
            self.current_distance or 300,  # Default distance
            self.tracking_confidence,
            velocity_mag,
            self.lost_target_count
        )

        print(f"[AdvancedTracker] {pattern_name} scan: {scan_points} points, "
              f"±{az_radius:.1f}°/±{el_radius:.1f}°")

        scan_results = []
        max_scan_time = 0.3  # Adaptive time limit

        # Generate scan positions with optimized ordering
        scan_positions = []
        for i in range(scan_points):
            angle = (2 * math.pi * i) / scan_points
            scan_az = center_az + az_radius * math.cos(angle)
            scan_el = center_el + el_radius * math.sin(angle)
            scan_el = max(0, min(90, scan_el))
            scan_positions.append((scan_az, scan_el, angle))

        # Sort by distance from current position for efficiency
        current_az = self.shared_data["stepper_degrees"].value
        scan_positions.sort(key=lambda pos: abs(self.angle_difference(current_az, pos[0])))

        for scan_az, scan_el, _ in scan_positions:
            if (time.time() - scan_start > max_scan_time or
                    self.shared_data["shutdown"].value or
                    not self.shared_data["debug_mode"].value):
                break

            # Move with adaptive precision
            self._move_fast_adaptive(scan_az, scan_el)

            # Get samples from async processor
            samples = self.async_lidar.get_samples_at_position(scan_az, scan_el, tolerance=3.0)

            # If no async samples, take a quick direct sample
            if not samples:
                samples = [self._get_current_sample()]

            # Process samples
            for sample in samples:
                if (sample and
                        self.clutter_filter(sample['azimuth'], sample['elevation'],
                                            sample['distance'], sample['strength'])):

                    # Adaptive threshold based on distance
                    distance = sample['distance']
                    adaptive_threshold = max(self.min_strength_threshold,
                                             self.min_strength_threshold * (distance / 300))

                    if sample['strength'] >= adaptive_threshold:
                        result = (sample['azimuth'], sample['elevation'],
                                  sample['distance'], sample['strength'])
                        scan_results.append(result)

                        # Early termination for very strong targets
                        if sample['strength'] >= self.good_target_threshold:
                            print(f"[AdvancedTracker] Strong target found ({sample['strength']}), "
                                  "ending scan early")
                            return scan_results

        return scan_results

    def _move_fast_adaptive(self, azimuth, elevation):
        """Fast movement with adaptive precision and step size."""
        if self.shared_data["shutdown"].value:
            return

        current_az = self.shared_data["stepper_degrees"].value
        target_az, step_multiplier = self.calculate_shortest_path(current_az, azimuth)

        # Adaptive positioning - use lower precision for fast scans
        tolerance = self.position_tolerance * step_multiplier

        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = elevation
        self.shared_data["go_to_target"].value = True

        # Reduced timeout with adaptive tolerance
        timeout = time.time() + (0.3 / step_multiplier)

        while self.shared_data["go_to_target"].value and time.time() < timeout:
            if self.shared_data["shutdown"].value:
                break

            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            az_diff = abs(self.angle_difference(current_az, target_az))
            el_diff = abs(current_el - elevation)

            if az_diff < tolerance and el_diff < tolerance:
                break

            time.sleep(0.0005)  # Very short sleep

    def _get_current_sample(self):
        """Get current sample directly."""
        try:
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            with self.shared_data["lidar_data"].get_lock():
                dist = self.shared_data["lidar_data"][0]
                strength = self.shared_data["lidar_data"][1]

            if dist > 0:
                return {
                    'timestamp': time.time(),
                    'azimuth': current_az,
                    'elevation': current_el,
                    'distance': dist,
                    'strength': strength
                }
        except:
            pass
        return None

    def find_best_target_advanced(self, scan_results):
        """Advanced target selection with confidence scoring."""
        if not scan_results:
            return None

        # Score targets based on multiple factors
        scored_targets = []

        for target in scan_results:
            az, el, dist, strength = target

            # Base score from strength
            score = strength

            # Distance bonus (closer targets preferred, but not too close)
            if 50 < dist < 500:
                distance_bonus = 1.0 + (500 - dist) / 500 * 0.5
            else:
                distance_bonus = 0.8
            score *= distance_bonus

            # Prediction bonus if target is near predicted position
            pred_az, pred_el, pred_conf = self.velocity_predictor.predict_position(0.01)
            if pred_az is not None and pred_conf > 0.2:
                pred_distance = math.sqrt(
                    self.angle_difference(az, pred_az) ** 2 + (el - pred_el) ** 2
                )
                if pred_distance < 5.0:
                    score *= (1.0 + pred_conf * 0.3)

            scored_targets.append((score, target))

        # Select best target
        best_score, best_target = max(scored_targets, key=lambda x: x[0])

        az, el, dist, strength = best_target

        # Update tracking confidence
        self.tracking_confidence = min(1.0, best_score / 200.0)  # Normalize to 0-1
        self.current_distance = dist

        print(f"[AdvancedTracker] Best target: ({az:.1f}°, {el:.1f}°) "
              f"dist={dist:.0f}cm, str={strength}, conf={self.tracking_confidence:.2f}")

        # Update satellite points
        self._update_satellite_points(az, el, dist, strength)

        return best_target

    def _update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update satellite points with error handling."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()
        except Exception as e:
            print(f"[AdvancedTracker] Error updating satellite_points: {e}")

    def run(self):
        """
        Main advanced tracking loop with all optimizations.
        """
        print("[AdvancedTracker] Starting advanced tracking with all optimizations")
        self.start()

        try:
            while not self.shared_data["shutdown"].value:
                if self.shared_data["debug_mode"].value:
                    cycle_start = time.time()

                    # Initialize tracking if needed
                    if self.current_target_az is None:
                        self.current_target_az = self.shared_data["stepper_degrees"].value or 180.0
                        self.current_target_el = self.shared_data["servo_degrees"].value or 45.0
                        print(f"[AdvancedTracker] Starting at ({self.current_target_az:.1f}°, "
                              f"{self.current_target_el:.1f}°)")

                    # Perform advanced scan
                    scan_results = self.predictive_scan(self.current_target_az, self.current_target_el)

                    # Find best target with advanced scoring
                    best_target = self.find_best_target_advanced(scan_results)

                    if best_target:
                        # Target found - update tracking state
                        az, el, dist, strength = best_target
                        self.lost_target_count = 0

                        # Add to velocity predictor
                        self.velocity_predictor.add_position(az, el)

                        # Update history
                        self.target_history.append({
                            'timestamp': time.time(),
                            'azimuth': az,
                            'elevation': el,
                            'distance': dist,
                            'strength': strength
                        })

                        if len(self.target_history) > 10:
                            self.target_history.pop(0)

                        # Move to target with prediction
                        pred_az, pred_el, pred_conf = self.velocity_predictor.predict_position(0.1)

                        if pred_conf > 0.4 and pred_az is not None:
                            # Use predicted position
                            move_az, move_el = pred_az, pred_el
                            print(f"[AdvancedTracker] Moving to predicted position")
                        else:
                            # Use current position
                            move_az, move_el = az, el

                        self.current_target_az = move_az
                        self.current_target_el = move_el

                        self._move_fast_adaptive(move_az, move_el)

                    else:
                        # Target lost
                        self.lost_target_count += 1
                        self.tracking_confidence *= 0.8  # Decay confidence

                        print(f"[AdvancedTracker] Target lost ({self.lost_target_count})")

                        # Clear satellite points
                        try:
                            with self.shared_data["satellite_points"].get_lock():
                                for i in range(5):
                                    self.shared_data["satellite_points"][i] = 0.0
                        except:
                            pass

                        # Quick expansion of search
                        if self.lost_target_count >= self.max_lost_count:
                            self.lost_target_count = 0

                    # Performance monitoring
                    cycle_time = time.time() - cycle_start
                    self.cycle_times.append(cycle_time)

                    avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times)
                    if len(self.cycle_times) >= 10 and avg_cycle_time > 0.4:
                        print(f"[AdvancedTracker] Performance warning: avg cycle {avg_cycle_time:.2f}s")

                else:
                    # Debug mode disabled
                    if self.current_target_az is not None:
                        print("[AdvancedTracker] Debug mode disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.target_history = []
                        self.velocity_predictor = VelocityPredictor()  # Reset predictor
                        self.lost_target_count = 0
                        self.tracking_confidence = 0.0

                    time.sleep(0.02)

                # Minimal delay
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("[AdvancedTracker] Interrupted by user")
        except Exception as e:
            print(f"[AdvancedTracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[AdvancedTracker] Shutting down")
            self.stop()


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """
    Run the advanced target tracker process.
    """
    print("[AdvancedTracker] Initializing advanced tracker process...")
    tracker = AdvancedTargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[AdvancedTracker] Process error: {e}")
    finally:
        print("[AdvancedTracker] Advanced process ended")