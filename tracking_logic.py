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
    """
    A robust filter that rejects static background objects by first finding clutter
    in the same direction (az/el) and then checking if the new point is
    significantly closer than the known background object.
    """

    def __init__(self, background_file="background_data.npy", angular_tolerance=1.0, distance_margin_cm=30.0):
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

            # --- KEY CHANGE ---
            # Build k-d tree on directional coordinates ONLY [az, el] for fast angular search.
            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords)
            print("[ClutterFilter] 2D (directional) k-d tree built successfully.")

        except FileNotFoundError:
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found. Running without clutter filtering.")
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

        except Exception:
            return True # Fail-safe: if the query fails, accept the measurement.


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





import time
import math
from enum import Enum


# --- Helper objects for the class to be self-contained ---

class HandTrackerState(Enum):
    """Defines the possible states for the HandTracker."""
    IDLE = 0
    SCANNING = 1
    COASTING = 2





# --- Modified HandTracker Class ---

class HandTracker:
    """
    High-performance predictive tracker with velocity smoothing, dynamic rate adjustment,
    and a predictive "coasting" search mode for target reacquisition.
    """

    def __init__(self, scan_radius=10, scan_points=8, time_per_waypoint=0.030, timeout=1.0, coast_timeout=1.5,
                 prediction_factor=1, velocity_smoothing_factor=0.5):
        self.scan_radius = scan_radius
        self.scan_points = scan_points
        self.time_per_waypoint = time_per_waypoint
        self.timeout = timeout
        self.coast_timeout = coast_timeout  # NEW: How long to search before giving up
        self.prediction_factor = prediction_factor
        self.velocity_smoothing_factor = velocity_smoothing_factor

        self.state = HandTrackerState.IDLE
        self.best_point = {'az': 0, 'el': 0, 'strength': 0, 'dist': 0, 'time': 0}
        self.previous_best_point = None
        self.scan_path = []
        self.scan_index = 0
        self.last_waypoint_time = 0

        # State for smoothed velocity
        self.smoothed_velocity = {'az': 0.0, 'el': 0.0}

        # --- NEW: State variables for coasting ---
        self.coast_start_time = 0
        self.last_coast_update_time = 0
        self.coasting_target_pos = {'az': 0.0, 'el': 0.0}

    def reset(self):
        """Resets the tracker to its initial state."""
        self.state = HandTrackerState.IDLE
        self.best_point['strength'] = 0
        self.best_point['time'] = 0
        self.previous_best_point = None
        self.smoothed_velocity = {'az': 0.0, 'el': 0.0}
        self.coast_start_time = 0
        print("[HandTracker] Reset.")

    def _generate_scan_path(self, center_az, center_el):
        """Generates a circular scan path around a center point."""
        self.scan_path = []
        for i in range(self.scan_points):
            angle = (i / self.scan_points) * 2 * math.pi
            az_offset = self.scan_radius * math.cos(angle)
            el_offset = self.scan_radius * math.sin(angle)
            self.scan_path.append((center_az + az_offset, center_el + el_offset))
        norm_az = center_az % 360
        print(f"[HandTracker] Generated new scan path centered at Az={norm_az:.1f}, El={center_el:.1f}")

    def update(self, current_az, current_el, measurement, shared_data):
        current_time = time.time()

        if self.state == HandTrackerState.IDLE:
            if measurement:
                dist, strength = measurement
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                   'time': current_time}
                self.previous_best_point = self.best_point.copy()
                self._generate_scan_path(current_az, current_el)
                self.scan_index = 0
                self.last_waypoint_time = current_time
                print(f"[HandTracker] Acquired target. Starting scan at Az={current_az:.1f}, El={current_el:.1f}")

        elif self.state == HandTrackerState.SCANNING:
            # --- TRANSITION TO COASTING on target loss ---
            if current_time - self.best_point['time'] > self.timeout:
                print("[HandTracker] Target lost. Entering COASTING mode.")
                self.state = HandTrackerState.COASTING
                self.coast_start_time = current_time
                self.last_coast_update_time = current_time
                self.coasting_target_pos = {'az': self.best_point['az'], 'el': self.best_point['el']}
                return

            if measurement:
                dist, strength = measurement
                #check distance against background scan

                if strength > self.best_point['strength']:
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                        'time': current_time}

            # Dynamic Rate Adjustment
            time_since_last_waypoint = current_time - self.last_waypoint_time
            if time_since_last_waypoint >= self.time_per_waypoint:
                waypoints_to_advance = max(1, int(time_since_last_waypoint / self.time_per_waypoint))
                self.last_waypoint_time = current_time
                for _ in range(waypoints_to_advance):
                    command_az, command_el = self.scan_path[self.scan_index]
                    command_motors_to_target(command_az, command_el, shared_data)
                    self.scan_index = (self.scan_index + 1)

                    if self.scan_index >= len(self.scan_path):
                        self.scan_index = 0
                        next_center_az, next_center_el = self.best_point['az'], self.best_point['el']

                        if self.previous_best_point and self.previous_best_point['strength'] > 0:
                            raw_delta_az = self.best_point['az'] - self.previous_best_point['az']
                            if raw_delta_az > 180:
                                raw_delta_az -= 360
                            elif raw_delta_az < -180:
                                raw_delta_az += 360
                            raw_delta_el = self.best_point['el'] - self.previous_best_point['el']
                            s = self.velocity_smoothing_factor
                            self.smoothed_velocity['az'] = (s * self.smoothed_velocity['az']) + ((1 - s) * raw_delta_az)
                            self.smoothed_velocity['el'] = (s * self.smoothed_velocity['el']) + ((1 - s) * raw_delta_el)
                            next_center_az = self.best_point['az'] + (
                                        self.smoothed_velocity['az'] * self.prediction_factor)
                            next_center_el = self.best_point['el'] + (
                                        self.smoothed_velocity['el'] * self.prediction_factor)

                        # --- NEW: Save best point to shared data ---
                        if "tracking_history" in shared_data:
                            point_data = [self.best_point['az'], self.best_point['el'], self.best_point['dist'],
                                          self.best_point['strength'], self.best_point['time']]
                            shared_data["tracking_history"].append(point_data)
                            # When you decide it's time to generate the TLE
                            if len(shared_data["tracking_history"]) >= 30:  # Example: trigger after 5 points
                                print("[Tracking Logic] Triggering TLE generation.")
                                shared_data["generate_tle"].value = True

                        self.previous_best_point = self.best_point.copy()
                        self._generate_scan_path(next_center_az, next_center_el)
                        self.best_point['strength'] *= 0.9

        # --- NEW: COASTING STATE LOGIC ---
        elif self.state == HandTrackerState.COASTING:
            # Failure: Coasted for too long without finding the target
            if current_time - self.coast_start_time > self.coast_timeout:
                print("[HandTracker] Coasting failed to reacquire target. Resetting.")
                self.reset()
                return

            # Success: Found a target while coasting
            if measurement:
                dist, strength = measurement
                print("[HandTracker] Reacquired target during coast! Resuming scan.")
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                   'time': current_time}
                # Center new scan on the reacquired point
                self._generate_scan_path(current_az, current_el)
                self.scan_index = 0
                self.last_waypoint_time = current_time
                return

            # Still coasting: Predict and move along the last known velocity vector
            dt = current_time - self.last_coast_update_time
            self.last_coast_update_time = current_time

            # Calculate the predicted change in position
            predicted_delta_az = 4.5*self.smoothed_velocity['az'] * (dt / (self.scan_points * self.time_per_waypoint))
            predicted_delta_el = 1.5*self.smoothed_velocity['el'] * (dt / (self.scan_points * self.time_per_waypoint))

            # Update the coasting target position
            self.coasting_target_pos['az'] += predicted_delta_az
            self.coasting_target_pos['el'] += predicted_delta_el

            command_motors_to_target(self.coasting_target_pos['az'], self.coasting_target_pos['el'], shared_data)

class ReactiveTracker:
    """Non-predictive tracker for immediate, reactive tracking of any target."""

    def __init__(self, smoothing_factor=0.4):
        self.smoothing_factor = smoothing_factor
        self.target_az = None
        self.target_el = None
        self.measurement_history = deque(maxlen=5)
        self.last_update_time = time.time()

    def update(self, azimuth, elevation, distance, strength):
        current_time = time.time()
        measurement = {'az': azimuth, 'el': elevation, 'dist': distance, 'strength': strength, 'time': current_time}
        self.measurement_history.append(measurement)
        adaptive_smoothing = self._calculate_adaptive_smoothing(strength)

        if self.target_az is None:
            self.target_az = azimuth
            self.target_el = elevation
        else:
            az_diff = azimuth - self.target_az
            if az_diff > 180:
                az_diff -= 360
            elif az_diff < -180:
                az_diff += 360
            self.target_az = self.target_az + (az_diff * adaptive_smoothing)
            self.target_el = (self.target_el * (1 - adaptive_smoothing) + elevation * adaptive_smoothing)
            self.target_az = self.target_az % 360

        self.last_update_time = current_time
        return self.target_az, self.target_el

    def _calculate_adaptive_smoothing(self, strength):
        base_smoothing = self.smoothing_factor
        if strength > 800:
            strength_factor = 1.5
        elif strength > 400:
            strength_factor = 1.2
        elif strength > 200:
            strength_factor = 0.8
        else:
            strength_factor = 0.5
        consistency_factor = self._check_measurement_consistency()
        adaptive_factor = base_smoothing * strength_factor * consistency_factor
        return max(0.1, min(1.0, adaptive_factor))

    def _check_measurement_consistency(self):
        if len(self.measurement_history) < 3: return 1.0
        recent_az = [m['az'] for m in list(self.measurement_history)[-3:]]
        recent_el = [m['el'] for m in list(self.measurement_history)[-3:]]
        az_diffs = []
        for i in range(1, len(recent_az)):
            diff = recent_az[i] - recent_az[i - 1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            az_diffs.append(abs(diff))
        el_diffs = [abs(recent_el[i] - recent_el[i - 1]) for i in range(1, len(recent_el))]
        if az_diffs and el_diffs:
            total_change = np.mean(az_diffs) + np.mean(el_diffs)
            if total_change < 2.0:
                return 1.2
            elif total_change < 5.0:
                return 1.0
            elif total_change < 10.0:
                return 0.8
            else:
                return 0.6
        return 1.0

    def get_current_target(self):
        if self.target_az is None: return None
        return self.target_az, self.target_el


def command_motors_to_target(azimuth, elevation, shared_data):
    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True


def run_tracking_logic(shared_data):
    """Main tracking logic process."""
    print("[TrackingLogic] Starting tracking logic process...")

    # Initialize components
    clutter_filter = ClutterFilter(shared_data.get("background_path", "background_data.npy").value)
    orbital_ekf = OrbitalEKF()
    acquirer = Acquirer()
    reactive_tracker = ReactiveTracker()
    hand_tracker = HandTracker()

    state = TrackingState.IDLE
    last_prediction_time = time.time()
    prediction_interval = 0.1

    shared_data["tracking_logic_ready"].value = True

    print("[TrackingLogic] Ready and running...")
    print("  - debug_mode=True: High-performance predictive hand tracking")
    print("  - reactive_mode=True: Simple reactive tracking")
    print("  - Normal mode: Advanced orbital tracking")

    while not shared_data["shutdown"].value:
        try:
            current_time = time.time()
            with shared_data["lidar_data"].get_lock():
                dist, strength, timestamp = shared_data["lidar_data"][:]
            current_az = shared_data["stepper_degrees"].value
            current_el = shared_data["servo_degrees"].value
            measurement_valid = (10.0 <= dist <= 600.0)

            if shared_data["debug_mode"].value:
                if state != TrackingState.DEBUG_MODE:
                    print("[TrackingLogic] Switching to DEBUG_MODE (Predictive Hand Tracking)")
                    state = TrackingState.DEBUG_MODE
                    hand_tracker.reset()
                is_valid_target = measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist,
                                                                                       strength)
                measurement_data = (dist, strength) if is_valid_target else None
                hand_tracker.update(current_az, current_el, measurement_data, shared_data)
                if hand_tracker.state == HandTrackerState.SCANNING:
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = hand_tracker.best_point['az']
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = hand_tracker.best_point['el']

            elif shared_data["reactive_mode"].value:
                # ... (rest of the logic remains the same)
                if state != TrackingState.REACTIVE_MODE:
                    print("[TrackingLogic] Switching to REACTIVE_MODE (Non-predictive tracking)")
                    state = TrackingState.REACTIVE_MODE
                    reactive_tracker = ReactiveTracker()
                if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                    target_az, target_el = reactive_tracker.update(current_az, current_el, dist, strength)
                    command_motors_to_target(target_az, target_el, shared_data)
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = target_az
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = target_el
            else:
                # ... (rest of the logic remains the same)
                if shared_data["acquire_points"].value:
                    if state != TrackingState.ACQUIRING:
                        print("[TrackingLogic] Switching to ACQUIRING mode")
                        state = TrackingState.ACQUIRING
                        acquirer = Acquirer()
                        shared_data["acquirer_status"].value = 1
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        if acquirer.add_measurement(current_az, current_el, dist, current_time):
                            initial_state = acquirer.compute_initial_state()
                            if initial_state is not None:
                                orbital_ekf.state = initial_state
                                orbital_ekf.initialized = True
                                shared_data["ekf_initialized"].value = True
                                shared_data["acquire_points"].value = False
                                shared_data["acquirer_status"].value = 0
                elif shared_data["lidar_track_mode_active"].value and orbital_ekf.initialized:
                    if state != TrackingState.TRACKING:
                        print("[TrackingLogic] Switching to TRACKING mode (Predictive)")
                        state = TrackingState.TRACKING
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        orbital_ekf.update([current_az, current_el, dist], strength)
                else:
                    if state != TrackingState.IDLE: state = TrackingState.IDLE

            if orbital_ekf.initialized and state == TrackingState.TRACKING:
                if current_time - last_prediction_time >= prediction_interval:
                    dt = current_time - orbital_ekf.last_update_time
                    orbital_ekf.predict(min(dt, 1.0))
                    prediction = orbital_ekf.get_predicted_position(0.5)
                    if prediction is not None:
                        pred_az, pred_el, pred_dist = prediction
                        with shared_data["predicted_azimuth"].get_lock():
                            shared_data["predicted_azimuth"].value = pred_az
                        with shared_data["predicted_elevation"].get_lock():
                            shared_data["predicted_elevation"].value = pred_el
                        command_motors_to_target(pred_az, pred_el, shared_data)
                    last_prediction_time = current_time

            time.sleep(0.0005)

        except Exception as e:
            print(f"[TrackingLogic] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    print("[TrackingLogic] Shutting down...")