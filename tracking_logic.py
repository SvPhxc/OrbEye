#!/usr/bin/env python3
"""
Precision Tracking Logic with Position Validation
Validates that LiDAR measurements match expected motor positions
"""

import time
import math
import numpy as np
from collections import deque
from enum import Enum
import threading


class TrackingMode(Enum):
    """Tracking modes"""
    IDLE = 0
    ORBITAL = 1   # Predictive tracking for orbiting drone
    REACTIVE = 2  # Simple reactive tracking for hand
    ACQUIRING = 3 # Initial target acquisition


class PositionValidator:
    """
    Validates that LiDAR measurements correspond to expected positions
    Detects and compensates for synchronization errors
    """

    def __init__(self, tolerance_degrees=0.5):
        self.tolerance = tolerance_degrees
        self.validation_history = deque(maxlen=100)
        self.sync_offset_az = 0.0  # Learned synchronization offset
        self.sync_offset_el = 0.0
        self.calibrated = False

    def validate_measurement(self, lidar_az, lidar_el, motor_az, motor_el, timestamp):
        """
        Validate that LiDAR measurement position matches motor position
        Returns (is_valid, corrected_az, corrected_el)
        """
        # Calculate position error
        az_error = lidar_az - motor_az
        el_error = lidar_el - motor_el

        # Handle azimuth wrap-around
        if az_error > 180:
            az_error -= 360
        elif az_error < -180:
            az_error += 360

        # Store in history for calibration
        self.validation_history.append({
            'timestamp': timestamp,
            'az_error': az_error,
            'el_error': el_error,
            'lidar_az': lidar_az,
            'lidar_el': lidar_el,
            'motor_az': motor_az,
            'motor_el': motor_el
        })

        # Auto-calibrate after enough samples
        if not self.calibrated and len(self.validation_history) >= 50:
            self._calibrate_offsets()

        # Apply learned offsets
        corrected_az = lidar_az - self.sync_offset_az
        corrected_el = lidar_el - self.sync_offset_el

        # Check if within tolerance
        final_az_error = abs(corrected_az - motor_az)
        final_el_error = abs(corrected_el - motor_el)

        if final_az_error > 180:
            final_az_error = 360 - final_az_error

        is_valid = (final_az_error <= self.tolerance and final_el_error <= self.tolerance)

        if not is_valid and self.calibrated:
            print(f"[Validator] Position mismatch: Az={final_az_error:.2f}° El={final_el_error:.2f}°")

        return is_valid, corrected_az, corrected_el

    def _calibrate_offsets(self):
        """Auto-calibrate synchronization offsets from history"""
        if len(self.validation_history) < 20:
            return

        # Calculate median offsets (robust to outliers)
        az_errors = [h['az_error'] for h in self.validation_history]
        el_errors = [h['el_error'] for h in self.validation_history]

        self.sync_offset_az = np.median(az_errors)
        self.sync_offset_el = np.median(el_errors)

        # Calculate consistency
        az_std = np.std(az_errors)
        el_std = np.std(el_errors)

        if az_std < 1.0 and el_std < 1.0:  # Good consistency
            self.calibrated = True
            print(f"[Validator] Calibrated - Offsets: Az={self.sync_offset_az:.3f}° El={self.sync_offset_el:.3f}°")
            print(f"[Validator] Sync quality: Az_std={az_std:.3f}° El_std={el_std:.3f}°")
        else:
            print(f"[Validator] Poor sync consistency - Az_std={az_std:.3f}° El_std={el_std:.3f}°")


class PrecisionClutterFilter:
    """
    Enhanced clutter filter with position-aware filtering
    """

    def __init__(self, background_file="background_scan.npy"):
        self.background_data = None
        self.position_grid = {}
        self.loaded = False

        try:
            # Load background data with position information
            data = np.load(background_file)
            if len(data) > 0:
                # Build precise position grid
                for point in data:
                    if len(point) >= 4:
                        az, el, dist, strength = point[:4]
                        # Use 2-degree grid cells for higher precision
                        grid_key = (round(az/2)*2, round(el/2)*2)

                        if grid_key not in self.position_grid:
                            self.position_grid[grid_key] = []
                        self.position_grid[grid_key].append({
                            'dist': dist,
                            'strength': strength,
                            'exact_az': az,
                            'exact_el': el
                        })

                # Calculate statistics per grid cell
                for key in self.position_grid:
                    points = self.position_grid[key]
                    distances = [p['dist'] for p in points]
                    self.position_grid[key] = {
                        'mean_dist': np.mean(distances),
                        'min_dist': np.min(distances),
                        'std_dist': np.std(distances),
                        'count': len(points)
                    }

                self.loaded = True
                print(f"[ClutterFilter] Loaded {len(self.position_grid)} precision grid cells")

        except Exception as e:
            print(f"[ClutterFilter] No background data: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength, timestamp=None):
        """
        Precise validation considering exact position
        """
        # Basic signal quality check
        if strength < 100:
            return False

        if not self.loaded:
            # No background - use basic distance check
            return 100 < distance < 500

        # Get nearest grid cells (check 2x2 neighborhood)
        grid_az = round(azimuth/2)*2
        grid_el = round(elevation/2)*2

        # Check neighboring cells for better statistics
        min_background = float('inf')

        for daz in [-2, 0, 2]:
            for del_ in [-2, 0, 2]:
                key = ((grid_az + daz) % 360, max(0, min(90, grid_el + del_)))
                if key in self.position_grid:
                    cell = self.position_grid[key]
                    # Use minimum distance minus 2 standard deviations
                    threshold = cell['min_dist'] - 2 * max(10, cell['std_dist'])
                    min_background = min(min_background, threshold)

        # Target must be significantly closer than background
        if min_background < float('inf'):
            return distance < min_background - 30  # 30cm margin

        # No background data for this region
        return 100 < distance < 500


class PrecisionOrbitalTracker:
    """
    High-precision orbital tracker with Kalman filtering
    """

    def __init__(self, orbit_period=20.0, orbit_radius_m=2.0):
        self.orbit_period = orbit_period
        self.orbit_radius = orbit_radius_m * 100  # Convert to cm
        self.angular_velocity = 2 * math.pi / orbit_period  # rad/sec

        # Kalman filter state: [angle, angular_velocity, radius, radial_velocity]
        self.state = np.array([0.0, self.angular_velocity, self.orbit_radius, 0.0])
        self.covariance = np.eye(4) * 100

        # Process noise (how much the model can vary)
        self.Q = np.diag([0.01, 0.001, 10.0, 1.0])  # angle, vel, radius, radial_vel

        # Measurement noise (based on LiDAR precision)
        self.R = np.diag([0.1, 1.0])  # angle noise, distance noise (cm)

        self.initialized = False
        self.last_update_time = None
        self.tracking_quality = 0.0
        self.measurement_history = deque(maxlen=50)

        print(f"[OrbitalTracker] Precision tracker initialized")
        print(f"  Period: {orbit_period}s, Radius: {orbit_radius_m}m")

    def initialize_from_measurements(self, measurements):
        """
        Initialize tracker from multiple measurements
        measurements: list of (az, el, dist, strength, timestamp)
        """
        if len(measurements) < 3:
            return False

        # Estimate initial state from measurements
        angles = []
        distances = []
        times = []

        for m in measurements:
            az, el, dist, strength, ts = m
            # Convert to orbital angle (assumes horizontal orbit)
            angle = math.radians(az)
            angles.append(angle)
            distances.append(dist)
            times.append(ts)

        # Estimate angular velocity
        if len(angles) >= 2:
            dt = times[-1] - times[0]
            if dt > 0:
                dangle = angles[-1] - angles[0]
                # Handle wrap-around
                if dangle > math.pi:
                    dangle -= 2 * math.pi
                elif dangle < -math.pi:
                    dangle += 2 * math.pi
                angular_vel = dangle / dt
            else:
                angular_vel = self.angular_velocity
        else:
            angular_vel = self.angular_velocity

        # Set initial state
        self.state[0] = angles[-1]  # Current angle
        self.state[1] = angular_vel  # Angular velocity
        self.state[2] = np.mean(distances)  # Average radius
        self.state[3] = 0.0  # No radial velocity initially

        self.initialized = True
        self.last_update_time = times[-1]

        print(f"[OrbitalTracker] Initialized: angle={math.degrees(self.state[0]):.1f}°, "
              f"vel={math.degrees(self.state[1]):.1f}°/s, radius={self.state[2]:.0f}cm")

        return True

    def predict(self, dt):
        """Kalman filter prediction step"""
        if not self.initialized:
            return None

        # State transition matrix
        F = np.array([
            [1, dt, 0, 0],   # angle += angular_vel * dt
            [0, 1, 0, 0],    # angular_vel stays constant
            [0, 0, 1, dt],   # radius += radial_vel * dt
            [0, 0, 0, 0.95]  # radial_vel decays slightly
        ])

        # Predict state
        self.state = F @ self.state

        # Wrap angle to [-pi, pi]
        self.state[0] = (self.state[0] + math.pi) % (2 * math.pi) - math.pi

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

        return self.state.copy()

    def update(self, azimuth, elevation, distance, strength, timestamp):
        """
        Kalman filter update with new measurement
        Returns predicted position for next frame
        """
        if not self.initialized:
            # Try to initialize
            self.measurement_history.append((azimuth, elevation, distance, strength, timestamp))
            if len(self.measurement_history) >= 3:
                return self.initialize_from_measurements(list(self.measurement_history))
            return None

        # Calculate dt and predict
        dt = timestamp - self.last_update_time if self.last_update_time else 0.01
        self.predict(dt)

        # Measurement vector: [angle, distance]
        z = np.array([math.radians(azimuth), distance])

        # Expected measurement from state
        h = np.array([self.state[0], self.state[2]])

        # Measurement residual
        y = z - h

        # Wrap angle residual
        if y[0] > math.pi:
            y[0] -= 2 * math.pi
        elif y[0] < -math.pi:
            y[0] += 2 * math.pi

        # Measurement Jacobian
        H = np.array([
            [1, 0, 0, 0],  # angle measurement
            [0, 0, 1, 0]   # distance measurement
        ])

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

        # Update tracking quality based on innovation
        innovation_norm = np.sqrt(y.T @ np.linalg.inv(S) @ y)
        self.tracking_quality = max(0, min(1, 1.0 - innovation_norm / 10))

        self.last_update_time = timestamp

        # Store measurement
        self.measurement_history.append((azimuth, elevation, distance, strength, timestamp))

        # Return predicted position for next update
        return self.get_prediction(0.05)  # 50ms lookahead

    def get_prediction(self, lookahead_seconds):
        """Get predicted position after lookahead time"""
        if not self.initialized:
            return None

        # Make a prediction
        future_state = self.state.copy()
        future_state[0] += future_state[1] * lookahead_seconds
        future_state[2] += future_state[3] * lookahead_seconds

        # Convert to degrees
        future_az = math.degrees(future_state[0]) % 360

        # Simple elevation model (varies with orbit)
        orbit_phase = future_state[0] % (2 * math.pi)
        future_el = 45 + 20 * math.sin(orbit_phase)  # Varies between 25-65 degrees
        future_el = max(0, min(90, future_el))

        return future_az, future_el, self.tracking_quality

    def get_uncertainty(self):
        """Get current position uncertainty"""
        if not self.initialized:
            return float('inf')

        # Position uncertainty from covariance
        angle_var = self.covariance[0, 0]
        dist_var = self.covariance[2, 2]

        return math.sqrt(angle_var + dist_var/10000)  # Combined uncertainty metric


class AdaptiveReactiveTracker:
    """
    Adaptive reactive tracker with variable smoothing
    """

    def __init__(self):
        self.position_filter = None
        self.velocity_filter = None
        self.last_update_time = None
        self.tracking = False
        self.smoothing_factor = 0.3
        self.measurement_history = deque(maxlen=20)

        # Adaptive parameters
        self.min_smoothing = 0.1
        self.max_smoothing = 0.8

        print("[ReactiveTracker] Adaptive tracker initialized")

    def update(self, azimuth, elevation, distance, strength, timestamp):
        """
        Update with adaptive smoothing based on target behavior
        """
        if not self.tracking:
            # Initialize filters
            self.position_filter = np.array([azimuth, elevation])
            self.velocity_filter = np.array([0.0, 0.0])
            self.tracking = True
            self.last_update_time = timestamp
            print(f"[ReactiveTracker] Started tracking at Az={azimuth:.1f}° El={elevation:.1f}°")
            return azimuth, elevation

        # Calculate time delta
        dt = timestamp - self.last_update_time if self.last_update_time else 0.01

        # Calculate instantaneous velocity
        delta_az = azimuth - self.position_filter[0]
        delta_el = elevation - self.position_filter[1]

        # Handle azimuth wrap-around
        if delta_az > 180:
            delta_az -= 360
        elif delta_az < -180:
            delta_az += 360

        if dt > 0:
            instant_vel = np.array([delta_az / dt, delta_el / dt])
        else:
            instant_vel = np.array([0.0, 0.0])

        # Update velocity filter
        vel_smooth = 0.2
        self.velocity_filter = (1 - vel_smooth) * self.velocity_filter + vel_smooth * instant_vel

        # Calculate adaptive smoothing based on velocity
        speed = np.linalg.norm(self.velocity_filter)

        if speed < 10:  # Slow target - more smoothing
            self.smoothing_factor = self.max_smoothing
        elif speed > 100:  # Fast target - less smoothing
            self.smoothing_factor = self.min_smoothing
        else:
            # Linear interpolation
            factor = (speed - 10) / 90
            self.smoothing_factor = self.max_smoothing - factor * (self.max_smoothing - self.min_smoothing)

        # Adjust smoothing based on signal strength
        if strength > 800:  # Strong signal - trust it more
            self.smoothing_factor *= 0.7
        elif strength < 200:  # Weak signal - smooth more
            self.smoothing_factor = min(self.max_smoothing, self.smoothing_factor * 1.3)

        # Apply adaptive smoothing
        new_position = np.array([azimuth, elevation])
        self.position_filter = (1 - self.smoothing_factor) * self.position_filter + self.smoothing_factor * new_position

        # Store in history
        self.measurement_history.append({
            'timestamp': timestamp,
            'position': self.position_filter.copy(),
            'velocity': self.velocity_filter.copy(),
            'raw_position': new_position,
            'strength': strength
        })

        self.last_update_time = timestamp

        return self.position_filter[0], self.position_filter[1]

    def predict_ahead(self, seconds):
        """Predict position ahead in time"""
        if not self.tracking:
            return None

        predicted = self.position_filter + self.velocity_filter * seconds

        # Clamp elevation
        predicted[1] = max(0, min(90, predicted[1]))

        # Wrap azimuth
        predicted[0] = predicted[0] % 360

        return predicted[0], predicted[1]

    def reset(self):
        """Reset tracker"""
        self.tracking = False
        self.position_filter = None
        self.velocity_filter = None
        self.measurement_history.clear()
        print("[ReactiveTracker] Reset")


class PrecisionTrackingLogic:
    """
    Main precision tracking logic with position validation
    """

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.mode = TrackingMode.IDLE
        self.running = False

        # Initialize components
        self.validator = PositionValidator(tolerance_degrees=0.5)
        self.clutter_filter = PrecisionClutterFilter()
        self.orbital_tracker = PrecisionOrbitalTracker()
        self.reactive_tracker = AdaptiveReactiveTracker()

        # Tracking state
        self.last_valid_measurement = None
        self.lost_target_time = None
        self.consecutive_detections = 0
        self.search_state = {'index': 0, 'pattern': 'spiral'}

        # Performance monitoring
        self.stats = {
            'updates': 0,
            'valid_measurements': 0,
            'sync_errors': 0,
            'last_print_time': time.time()
        }

        print("[TrackingLogic] Precision tracking initialized")

    def _get_synchronized_measurement(self):
        """Get LiDAR measurement with validated position"""
        # Get raw data
        with self.shared_data["lidar_data"].get_lock():
            dist, strength, timestamp = self.shared_data["lidar_data"][:]

        # Get LiDAR-reported position
        with self.shared_data["lidar_position"].get_lock():
            lidar_az, lidar_el = self.shared_data["lidar_position"][:]

        # Get current motor position
        motor_az = self.shared_data["stepper_degrees"].value
        motor_el = self.shared_data["servo_degrees"].value

        # Validate synchronization
        is_valid, corrected_az, corrected_el = self.validator.validate_measurement(
            lidar_az, lidar_el, motor_az, motor_el, timestamp
        )

        if not is_valid:
            self.stats['sync_errors'] += 1

        return corrected_az, corrected_el, dist, strength, timestamp, is_valid

    def _command_motors_precise(self, azimuth, elevation, priority=1):
        """Send precise movement command with priority"""
        # Check if motors are ready
        if self.shared_data["target_reached"].value or not self.shared_data["go_to_target"].value:
            self.shared_data["target_azimuth"].value = azimuth
            self.shared_data["target_elevation"].value = elevation
            self.shared_data["go_to_target"].value = True
            return True
        return False

    def _adaptive_search_pattern(self):
        """
        Generate adaptive search pattern based on lost target
        """
        if not self.last_valid_measurement:
            return None

        base_az, base_el, last_dist, _, last_time = self.last_valid_measurement
        time_lost = time.time() - last_time

        # Expand search radius based on time lost
        search_radius = min(30, 5 + time_lost * 5)  # Max 30 degrees

        # Spiral pattern
        patterns = []
        steps = 8
        for ring in range(1, 4):
            radius = search_radius * ring / 3
            for i in range(steps * ring):
                angle = (2 * math.pi * i) / (steps * ring)
                daz = radius * math.cos(angle)
                del_ = radius * math.sin(angle) * 0.5  # Elliptical for elevation
                patterns.append((daz, del_))

        if self.search_state['index'] >= len(patterns):
            self.search_state['index'] = 0

        pattern = patterns[self.search_state['index']]
        self.search_state['index'] += 1

        target_az = (base_az + pattern[0]) % 360
        target_el = max(0, min(90, base_el + pattern[1]))

        return target_az, target_el

    def run(self):
        """Main precision tracking loop"""
        self.running = True
        self.shared_data["tracking_logic_ready"].value = True

        print("[TrackingLogic] Precision tracking loop started")
        print("  Mode selection:")
        print("    debug_mode=True  -> Orbital tracking (drone)")
        print("    reactive_mode=True -> Reactive tracking (hand)")

        update_interval = 0.001  # 1ms for 1000Hz operation
        last_update = time.time()

        while self.running and not self.shared_data["shutdown"].value:
            try:
                current_time = time.time()

                # Maintain precise update rate
                if current_time - last_update < update_interval:
                    time.sleep(0.0001)
                    continue

                last_update = current_time

                # Get synchronized measurement
                az, el, dist, strength, timestamp, sync_valid = self._get_synchronized_measurement()

                # Update statistics
                self.stats['updates'] += 1

                # Determine mode
                if self.shared_data["debug_mode"].value:
                    new_mode = TrackingMode.ORBITAL
                elif self.shared_data["reactive_mode"].value:
                    new_mode = TrackingMode.REACTIVE
                else:
                    new_mode = TrackingMode.IDLE

                # Mode change handling
                if new_mode != self.mode:
                    if new_mode == TrackingMode.ORBITAL:
                        self.orbital_tracker = PrecisionOrbitalTracker()
                        print("[TrackingLogic] Switched to ORBITAL mode")
                    elif new_mode == TrackingMode.REACTIVE:
                        self.reactive_tracker.reset()
                        print("[TrackingLogic] Switched to REACTIVE mode")
                    self.mode = new_mode

                # Process based on mode
                if self.mode != TrackingMode.IDLE:
                    # Check measurement validity
                    measurement_valid = (
                            sync_valid and
                            dist > 0 and
                            self.clutter_filter.is_valid_target(az, el, dist, strength, timestamp)
                    )

                    if measurement_valid:
                        # Valid target detected
                        self.stats['valid_measurements'] += 1
                        self.consecutive_detections += 1
                        self.last_valid_measurement = (az, el, dist, strength, timestamp)
                        self.lost_target_time = None
                        self.search_state['index'] = 0

                        # Process based on tracking mode
                        if self.mode == TrackingMode.ORBITAL:
                            # Update orbital tracker
                            result = self.orbital_tracker.update(az, el, dist, strength, timestamp)
                            if result:
                                if isinstance(result, tuple) and len(result) == 3:
                                    pred_az, pred_el, quality = result

                                    if quality > 0.3:  # Good tracking quality
                                        # Command motors to predicted position
                                        if self._command_motors_precise(pred_az, pred_el):
                                            # Update shared data for visualization
                                            self.shared_data["predicted_azimuth"].value = pred_az
                                            self.shared_data["predicted_elevation"].value = pred_el
                                            self.shared_data["ekf_confidence"].value = quality

                        elif self.mode == TrackingMode.REACTIVE:
                            # Update reactive tracker
                            target_az, target_el = self.reactive_tracker.update(
                                az, el, dist, strength, timestamp
                            )

                            # Get prediction for smoother tracking
                            predicted = self.reactive_tracker.predict_ahead(0.02)  # 20ms ahead
                            if predicted:
                                pred_az, pred_el = predicted
                            else:
                                pred_az, pred_el = target_az, target_el

                            # Command motors
                            if self._command_motors_precise(pred_az, pred_el):
                                # Update shared data
                                self.shared_data["predicted_azimuth"].value = pred_az
                                self.shared_data["predicted_elevation"].value = pred_el

                    else:
                        # No valid target
                        self.consecutive_detections = 0

                        if self.lost_target_time is None:
                            self.lost_target_time = current_time
                            print(f"[TrackingLogic] Target lost at {current_time:.3f}")

                        time_lost = current_time - self.lost_target_time

                        if time_lost < 3.0:  # Search for 3 seconds
                            # Execute search pattern
                            search_pos = self._adaptive_search_pattern()
                            if search_pos and time_lost > 0.1:  # Wait 100ms before searching
                                self._command_motors_precise(search_pos[0], search_pos[1], priority=0)
                        else:
                            # Give up
                            print(f"[TrackingLogic] Search timeout after {time_lost:.1f}s")
                            if self.mode == TrackingMode.ORBITAL:
                                self.orbital_tracker.initialized = False
                            else:
                                self.reactive_tracker.reset()
                            self.lost_target_time = None

                # Print statistics periodically
                if current_time - self.stats['last_print_time'] > 5.0:
                    update_rate = self.stats['updates'] / 5.0
                    valid_rate = self.stats['valid_measurements'] / 5.0
                    sync_errors = self.stats['sync_errors']

                    print(f"[TrackingLogic] Stats: {update_rate:.0f}Hz updates, "
                          f"{valid_rate:.0f}Hz valid, {sync_errors} sync errors")

                    # Reset counters
                    self.stats['updates'] = 0
                    self.stats['valid_measurements'] = 0
                    self.stats['sync_errors'] = 0
                    self.stats['last_print_time'] = current_time

                    # Print tracking quality
                    if self.mode == TrackingMode.ORBITAL and self.orbital_tracker.initialized:
                        uncertainty = self.orbital_tracker.get_uncertainty()
                        print(f"[TrackingLogic] Orbital uncertainty: {uncertainty:.3f}")

            except Exception as e:
                print(f"[TrackingLogic] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)

        print("[TrackingLogic] Main loop ended")

    def stop(self):
        """Stop tracking logic"""
        self.running = False

        # Print final statistics
        if self.validator.calibrated:
            print(f"[TrackingLogic] Final sync offsets: "
                  f"Az={self.validator.sync_offset_az:.3f}° "
                  f"El={self.validator.sync_offset_el:.3f}°")

        print("[TrackingLogic] Stopped")


def run_tracker_process(shared_data):
    """Entry point for precision tracking process"""
    tracker = None
    try:
        tracker = PrecisionTrackingLogic(shared_data)
        tracker.run()
    except Exception as e:
        print(f"[TrackingLogic] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tracker:
            tracker.stop()
        print("[TrackingLogic] Process terminated")