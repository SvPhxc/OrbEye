# tracking_logic.py

import time
import threading
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import math
from enum import Enum


class TrackingState(Enum):
    IDLE = 0
    ACQUIRING = 1
    TRACKING = 2
    DEBUG_MODE = 3
    REACTIVE_MODE = 4


class ClutterFilter:
    """Environmental awareness filter to reject static background objects."""

    def __init__(self, background_file="background_data.npy", distance_tolerance=50.0, strength_tolerance=100):
        """
        Initialize clutter filter with background map.

        Args:
            background_file: Path to background scan data
            distance_tolerance: Distance threshold in cm for clutter detection
            strength_tolerance: Strength threshold for clutter detection
        """
        self.distance_tolerance = distance_tolerance
        self.strength_tolerance = strength_tolerance
        self.background_tree = None
        self.background_data = None

        try:
            # Load background data [azimuth, elevation, distance_cm, strength]
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points")

            # Build k-d tree for efficient spatial queries
            # Use azimuth, elevation, and distance for spatial indexing
            coords = self.background_data[:, [0, 1, 2]]  # [az, el, dist]
            self.background_tree = cKDTree(coords)

        except FileNotFoundError:
            print(
                f"[ClutterFilter] Warning: Background file '{background_file}' not found. Running without clutter filtering.")
        except Exception as e:
            print(f"[ClutterFilter] Error loading background: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """
        Check if measurement represents a valid target (not background clutter).

        Returns:
            bool: True if target is valid, False if it's likely clutter
        """
        if self.background_tree is None:
            return True  # No background data, accept all measurements

        # Query k-d tree for nearby background points
        query_point = np.array([azimuth, elevation, distance])

        # Find closest background point
        try:
            dist, idx = self.background_tree.query(query_point, k=1)

            if dist < self.distance_tolerance:
                # Check if strength is significantly different from background
                bg_strength = self.background_data[idx, 3]
                strength_diff = abs(strength - bg_strength)

                if strength_diff < self.strength_tolerance:
                    return False  # Too similar to background, likely clutter

            return True  # Sufficiently different from background

        except Exception:
            return True  # If query fails, accept measurement


class OrbitalEKF:
    """Extended Kalman Filter for orbital tracking with strength-aware measurement noise."""

    def __init__(self):
        """Initialize EKF for 6-state orbital tracking [x, y, z, vx, vy, vz]."""
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1000  # Large initial uncertainty
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # Process noise
        self.initialized = False
        self.last_update_time = time.time()

    def predict(self, dt):
        """Predict step with orbital dynamics."""
        if not self.initialized:
            return

        # State transition matrix (simplified orbital dynamics)
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        # Simple orbital acceleration model (could be enhanced)
        r = np.linalg.norm(self.state[0:3])
        if r > 1.0:  # Avoid division by zero
            # Gravitational acceleration (simplified)
            mu = 1000  # Gravitational parameter (tunable)
            acc_factor = -mu / (r ** 3)
            F[3, 0] = acc_factor * dt
            F[4, 1] = acc_factor * dt
            F[5, 2] = acc_factor * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, strength):
        """
        Update step with strength-aware measurement noise.

        Args:
            measurement: [azimuth, elevation, distance] in degrees and cm
            strength: LiDAR return strength
        """
        if not self.initialized:
            return

        az, el, dist = measurement

        # Convert measurement to Cartesian
        az_rad = np.radians(az)
        el_rad = np.radians(el)
        dist_m = dist / 100.0

        z_meas = np.array([
            dist_m * np.cos(el_rad) * np.cos(az_rad),  # x
            dist_m * np.cos(el_rad) * np.sin(az_rad),  # y
            dist_m * np.sin(el_rad)  # z
        ])

        # Predicted measurement
        h_pred = self.state[0:3]

        # Measurement Jacobian
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        # **KEY FEATURE**: Strength-aware measurement noise
        base_var_pos = 1.0
        base_var_angle = 0.1

        # High strength = direct hit = low noise
        # Low strength = edge hit = high angular noise, but range is still good
        strength_factor = max(0.1, min(1.0, strength / 1000.0))  # Normalize strength

        if strength > 500:  # High strength - direct hit
            pos_variance = base_var_pos * 0.1
            angular_noise_factor = 1.0
        else:  # Low strength - edge hit
            pos_variance = base_var_pos * 1.0
            angular_noise_factor = 5.0  # Much higher angular uncertainty

        R = np.diag([pos_variance, pos_variance * angular_noise_factor, pos_variance * angular_noise_factor])

        # Innovation
        y = z_meas - h_pred

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P

        self.last_update_time = time.time()

    def get_predicted_position(self, future_time_sec=0.5):
        """Get predicted position at future time."""
        if not self.initialized:
            return None

        # Predict forward in time
        temp_state = self.state.copy()
        dt = future_time_sec

        # Simple ballistic prediction
        temp_state[0:3] += temp_state[3:6] * dt

        # Convert back to spherical coordinates
        x, y, z = temp_state[0:3]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if r < 1.0:
            return None

        el = np.degrees(np.arcsin(z / r))
        az = np.degrees(np.arctan2(y, x))

        # Ensure azimuth is in [0, 360)
        if az < 0:
            az += 360

        return az, el, r * 100  # Return in degrees and cm


class Acquirer:
    """Initial Orbit Determination using multiple measurements."""

    def __init__(self):
        self.measurements = []
        self.required_points = 3

    def add_measurement(self, azimuth, elevation, distance, timestamp):
        """Add a measurement for IOD calculation."""
        # Convert to Cartesian
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        dist_m = distance / 100.0

        pos = np.array([
            dist_m * np.cos(el_rad) * np.cos(az_rad),
            dist_m * np.cos(el_rad) * np.sin(az_rad),
            dist_m * np.sin(el_rad)
        ])

        self.measurements.append((pos, timestamp))

        if len(self.measurements) > self.required_points:
            self.measurements.pop(0)  # Keep only latest measurements

        return len(self.measurements) >= self.required_points

    def compute_initial_state(self):
        """Compute initial state vector using simplified Herrick-Gibbs method."""
        if len(self.measurements) < self.required_points:
            return None

        # Get positions and times
        positions = [m[0] for m in self.measurements]
        times = [m[1] for m in self.measurements]

        # Simple velocity estimation using finite differences
        r1, r2, r3 = positions
        t1, t2, t3 = times

        dt1 = t2 - t1
        dt2 = t3 - t2

        if dt1 <= 0 or dt2 <= 0:
            return None

        # Estimate velocity at middle point
        v2 = (r3 - r1) / (dt1 + dt2)

        # Return state [position, velocity]
        state = np.concatenate([r2, v2])
        return state


class ReactiveTracker:
    """Non-predictive tracker for immediate, reactive tracking of any target."""

    def __init__(self, smoothing_factor=0.4):
        """
        Initialize reactive tracker with smoothing.

        Args:
            smoothing_factor: Weight for new measurements (0-1)
                             Higher = more responsive, Lower = more stable
        """
        self.smoothing_factor = smoothing_factor
        self.target_az = None
        self.target_el = None
        self.measurement_history = deque(maxlen=5)  # Keep last 5 measurements
        self.last_update_time = time.time()

    def update(self, azimuth, elevation, distance, strength):
        """
        Update target position with adaptive smoothing based on measurement quality.

        Args:
            azimuth: Current azimuth in degrees
            elevation: Current elevation in degrees
            distance: Distance in cm
            strength: LiDAR return strength

        Returns:
            tuple: (smoothed_azimuth, smoothed_elevation)
        """
        current_time = time.time()

        # Store measurement with timestamp
        measurement = {
            'az': azimuth,
            'el': elevation,
            'dist': distance,
            'strength': strength,
            'time': current_time
        }
        self.measurement_history.append(measurement)

        # Adaptive smoothing based on strength and consistency
        adaptive_smoothing = self._calculate_adaptive_smoothing(strength)

        if self.target_az is None:
            # First measurement - no smoothing
            self.target_az = azimuth
            self.target_el = elevation
        else:
            # Handle azimuth wrap-around (0°/360°)
            az_diff = azimuth - self.target_az
            if az_diff > 180:
                az_diff -= 360
            elif az_diff < -180:
                az_diff += 360

            # Apply adaptive smoothing
            self.target_az = self.target_az + (az_diff * adaptive_smoothing)
            self.target_el = (self.target_el * (1 - adaptive_smoothing) +
                              elevation * adaptive_smoothing)

            # Normalize azimuth to [0, 360)
            self.target_az = self.target_az % 360

        self.last_update_time = current_time
        return self.target_az, self.target_el

    def _calculate_adaptive_smoothing(self, strength):
        """
        Calculate adaptive smoothing factor based on measurement quality.

        Args:
            strength: LiDAR return strength

        Returns:
            float: Adaptive smoothing factor
        """
        # Base smoothing factor
        base_smoothing = self.smoothing_factor

        # Adjust based on strength (higher strength = more responsive)
        if strength > 800:  # Very strong signal
            strength_factor = 1.5
        elif strength > 400:  # Good signal
            strength_factor = 1.2
        elif strength > 200:  # Weak signal
            strength_factor = 0.8
        else:  # Very weak signal
            strength_factor = 0.5

        # Check measurement consistency
        consistency_factor = self._check_measurement_consistency()

        # Combine factors
        adaptive_factor = base_smoothing * strength_factor * consistency_factor

        # Clamp to reasonable range
        return max(0.1, min(1.0, adaptive_factor))

    def _check_measurement_consistency(self):
        """
        Check consistency of recent measurements to adjust responsiveness.

        Returns:
            float: Consistency factor (1.0 = very consistent, 0.5 = inconsistent)
        """
        if len(self.measurement_history) < 3:
            return 1.0

        # Calculate variance in recent measurements
        recent_az = [m['az'] for m in list(self.measurement_history)[-3:]]
        recent_el = [m['el'] for m in list(self.measurement_history)[-3:]]

        # Handle azimuth wrap-around for variance calculation
        az_diffs = []
        for i in range(1, len(recent_az)):
            diff = recent_az[i] - recent_az[i - 1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            az_diffs.append(abs(diff))

        el_diffs = [abs(recent_el[i] - recent_el[i - 1]) for i in range(1, len(recent_el))]

        # Calculate average change rate
        if az_diffs and el_diffs:
            avg_az_change = np.mean(az_diffs)
            avg_el_change = np.mean(el_diffs)

            # If changes are small and consistent, be more responsive
            # If changes are large or erratic, be more conservative
            total_change = avg_az_change + avg_el_change

            if total_change < 2.0:  # Very stable
                return 1.2
            elif total_change < 5.0:  # Moderately stable
                return 1.0
            elif total_change < 10.0:  # Somewhat erratic
                return 0.8
            else:  # Very erratic
                return 0.6

        return 1.0

    def get_current_target(self):
        """Get current target without updating."""
        if self.target_az is None:
            return None
        return self.target_az, self.target_el


def command_motors_to_target(azimuth, elevation, shared_data):
    """Command the motor controller to move to a specific target position."""
    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True

    print(f"[TrackingLogic] Commanding motors to Az={azimuth:.1f}°, El={elevation:.1f}°")


def run_tracking_logic(shared_data):
    """Main tracking logic process."""
    print("[TrackingLogic] Starting tracking logic process...")

    # Initialize components
    clutter_filter = ClutterFilter(shared_data.get("background_path", "background_data.npy").value)
    orbital_ekf = OrbitalEKF()
    acquirer = Acquirer()

    reactive_tracker = ReactiveTracker()

    state = TrackingState.IDLE
    last_prediction_time = time.time()
    prediction_interval = 0.1  # Update predictions at 10Hz

    # Signal that tracking logic is ready
    shared_data["tracking_logic_ready"].value = True

    print("[TrackingLogic] Ready and running...")
    print("[TrackingLogic] Available modes:")
    print("  - debug_mode=True: Hand tracking (simple smoothing)")
    print("  - reactive_mode=True: Reactive tracking (no prediction)")
    print("  - Normal mode: Advanced orbital tracking with prediction")

    while not shared_data["shutdown"].value:
        try:
            current_time = time.time()

            # Get latest LiDAR measurement
            with shared_data["lidar_data"].get_lock():
                dist, strength, timestamp = shared_data["lidar_data"][:]

            # Get current gimbal position
            current_az = shared_data["stepper_degrees"].value
            current_el = shared_data["servo_degrees"].value

            # Check for valid measurement (basic range check)
            measurement_valid = (10.0 <= dist <= 16000.0 and
                                 shared_data.get("lidar_acceptance_range", [3.0, 50.0])[0] <= dist / 100.0 <=
                                 shared_data.get("lidar_acceptance_range", [3.0, 50.0])[1])

            # State machine logic - Priority order: Debug > Reactive > Advanced
            if shared_data["debug_mode"].value:
                if state != TrackingState.DEBUG_MODE:
                    print("[TrackingLogic] Switching to DEBUG_MODE (Hand Tracking)")
                    state = TrackingState.DEBUG_MODE

                # Process measurement in debug mode
                if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                    target_az, target_el = hand_tracker.update(current_az, current_el, dist, strength)

                    # Command motors directly
                    command_motors_to_target(target_az, target_el, shared_data)

                    # Also update predicted values for consistency
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = target_az
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = target_el

                    print(f"[HandTracker] Commanding target: Az={target_az:.1f}°, El={target_el:.1f}°")

            elif shared_data["reactive_mode"].value:
                if state != TrackingState.REACTIVE_MODE:
                    print("[TrackingLogic] Switching to REACTIVE_MODE (Non-predictive tracking)")
                    state = TrackingState.REACTIVE_MODE
                    reactive_tracker = ReactiveTracker()  # Reset tracker

                # Process measurement in reactive mode
                if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                    target_az, target_el = reactive_tracker.update(current_az, current_el, dist, strength)

                    # Command motors directly
                    command_motors_to_target(target_az, target_el, shared_data)

                    # Also update predicted values for consistency
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = target_az
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = target_el

                    print(
                        f"[ReactiveTracker] Commanding target: Az={target_az:.1f}°, El={target_el:.1f}°, Strength={strength}")

            else:
                # Advanced orbital tracking mode
                if shared_data["acquire_points"].value:
                    if state != TrackingState.ACQUIRING:
                        print("[TrackingLogic] Switching to ACQUIRING mode")
                        state = TrackingState.ACQUIRING
                        acquirer = Acquirer()  # Reset acquirer
                        shared_data["acquirer_status"].value = 1

                    # Process measurement for acquisition
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        is_complete = acquirer.add_measurement(current_az, current_el, dist, current_time)

                        if is_complete:
                            # Compute initial state
                            initial_state = acquirer.compute_initial_state()
                            if initial_state is not None:
                                orbital_ekf.state = initial_state
                                orbital_ekf.initialized = True

                                # Signal EKF initialization
                                shared_data["ekf_initialized"].value = True
                                shared_data["acquire_points"].value = False
                                shared_data["acquirer_status"].value = 0

                                print("[TrackingLogic] IOD complete, EKF initialized")
                                print(f"[TrackingLogic] Initial state: {initial_state}")

                elif shared_data["lidar_track_mode_active"].value and orbital_ekf.initialized:
                    if state != TrackingState.TRACKING:
                        print("[TrackingLogic] Switching to TRACKING mode (Predictive)")
                        state = TrackingState.TRACKING

                    # Process measurement for tracking
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        orbital_ekf.update([current_az, current_el, dist], strength)
                        print(
                            f"[OrbitalEKF] Updated with measurement: Az={current_az:.1f}°, El={current_el:.1f}°, Dist={dist:.0f}cm, Str={strength}")

                else:
                    if state != TrackingState.IDLE:
                        print("[TrackingLogic] Switching to IDLE mode")
                        state = TrackingState.IDLE

            # Continuous prediction for orbital tracking only
            if orbital_ekf.initialized and state == TrackingState.TRACKING:
                if current_time - last_prediction_time >= prediction_interval:
                    # Run prediction step
                    dt = current_time - orbital_ekf.last_update_time
                    orbital_ekf.predict(min(dt, 1.0))  # Cap dt to prevent instability

                    # Get predicted position
                    prediction = orbital_ekf.get_predicted_position(0.5)  # 0.5 sec ahead

                    if prediction is not None:
                        pred_az, pred_el, pred_dist = prediction

                        # Update shared data with predictions
                        with shared_data["predicted_azimuth"].get_lock():
                            shared_data["predicted_azimuth"].value = pred_az
                        with shared_data["predicted_elevation"].get_lock():
                            shared_data["predicted_elevation"].value = pred_el

                        # Command motors to predicted position
                        command_motors_to_target(pred_az, pred_el, shared_data)

                        print(f"[OrbitalEKF] Commanding prediction: Az={pred_az:.1f}°, El={pred_el:.1f}°")

                    last_prediction_time = current_time

            # Sleep to prevent CPU overload
            time.sleep(0.05)  # 20Hz main loop

        except Exception as e:
            print(f"[TrackingLogic] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    print("[TrackingLogic] Shutting down...")


