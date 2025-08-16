# tracking_logic.py

import time
import threading
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import math
from enum import Enum

# --- SAFETY & TUNING CONSTANTS ---
MAX_TRACKING_ELEVATION = 85.0  # Degrees. Prevents the tracker from locking onto the ceiling.


class TrackingState(Enum):
    IDLE = 0
    ACQUIRING = 1
    TRACKING = 2
    DEBUG_MODE = 3
    REACTIVE_MODE = 4


class ClutterFilter:
    # ... (This class remains unchanged)
    def __init__(self, background_file="background_data.npy", distance_tolerance=50.0, strength_tolerance=100):
        self.distance_tolerance = distance_tolerance;
        self.strength_tolerance = strength_tolerance
        self.background_tree = None;
        self.background_data = None
        try:
            self.background_data = np.load(background_file);
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points")
            coords = self.background_data[:, [0, 1, 2]];
            self.background_tree = cKDTree(coords)
        except FileNotFoundError:
            print(
                f"[ClutterFilter] Warning: Background file '{background_file}' not found. Running without clutter filtering.")
        except Exception as e:
            print(f"[ClutterFilter] Error loading background: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        if self.background_tree is None: return True
        query_point = np.array([azimuth, elevation, distance])
        try:
            dist, idx = self.background_tree.query(query_point, k=1)
            if dist < self.distance_tolerance:
                bg_strength = self.background_data[idx, 3];
                strength_diff = abs(strength - bg_strength)
                if strength_diff < self.strength_tolerance: return False
            return True
        except Exception:
            return True


class OrbitalEKF:
    # ... (This class remains unchanged)
    def __init__(self):
        self.state = np.zeros(6);
        self.P = np.eye(6) * 1000;
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        self.initialized = False;
        self.last_update_time = time.time()

    def predict(self, dt):
        if not self.initialized: return
        F = np.eye(6);
        F[0:3, 3:6] = np.eye(3) * dt
        r = np.linalg.norm(self.state[0:3])
        if r > 1.0:
            mu = 1000;
            acc_factor = -mu / (r ** 3)
            F[3, 0] = acc_factor * dt;
            F[4, 1] = acc_factor * dt;
            F[5, 2] = acc_factor * dt
        self.state = F @ self.state;
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, strength):
        if not self.initialized: return
        az, el, dist = measurement;
        az_rad, el_rad, dist_m = np.radians(az), np.radians(el), dist / 100.0
        z_meas = np.array([dist_m * np.cos(el_rad) * np.cos(az_rad), dist_m * np.cos(el_rad) * np.sin(az_rad),
                           dist_m * np.sin(el_rad)])
        h_pred = self.state[0:3];
        H = np.zeros((3, 6));
        H[0:3, 0:3] = np.eye(3)
        base_var_pos = 1.0
        if strength > 500:
            pos_variance, angular_noise_factor = base_var_pos * 0.1, 1.0
        else:
            pos_variance, angular_noise_factor = base_var_pos * 1.0, 5.0
        R = np.diag([pos_variance, pos_variance * angular_noise_factor, pos_variance * angular_noise_factor])
        y = z_meas - h_pred;
        S = H @ self.P @ H.T + R;
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y;
        self.P = (np.eye(6) - K @ H) @ self.P
        self.last_update_time = time.time()

    def get_predicted_position(self, future_time_sec=0.5):
        if not self.initialized: return None
        temp_state = self.state.copy();
        temp_state[0:3] += temp_state[3:6] * future_time_sec
        x, y, z = temp_state[0:3];
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r < 1.0: return None
        el = np.degrees(np.arcsin(z / r));
        az = np.degrees(np.arctan2(y, x))
        if az < 0: az += 360
        return az, el, r * 100


class Acquirer:
    # ... (This class remains unchanged)
    def __init__(self):
        self.measurements = []; self.required_points = 3

    def add_measurement(self, azimuth, elevation, distance, timestamp):
        az_rad, el_rad, dist_m = np.radians(azimuth), np.radians(elevation), distance / 100.0
        pos = np.array([dist_m * np.cos(el_rad) * np.cos(az_rad), dist_m * np.cos(el_rad) * np.sin(az_rad),
                        dist_m * np.sin(el_rad)])
        self.measurements.append((pos, timestamp))
        if len(self.measurements) > self.required_points: self.measurements.pop(0)
        return len(self.measurements) >= self.required_points

    def compute_initial_state(self):
        if len(self.measurements) < self.required_points: return None
        positions = [m[0] for m in self.measurements];
        times = [m[1] for m in self.measurements]
        r1, r2, r3 = positions;
        t1, t2, t3 = times;
        dt1, dt2 = t2 - t1, t3 - t2
        if dt1 <= 0 or dt2 <= 0: return None
        v2 = (r3 - r1) / (dt1 + dt2)
        return np.concatenate([r2, v2])


class HandTrackerState(Enum):
    IDLE = 0
    SCANNING = 1
    COASTING = 2


class HandTrackerKalmanFilter:
    # ... (This class remains unchanged but is still used)
    def __init__(self, process_noise=10.0, measurement_noise=25.0):
        self.dim_x = 4;
        self.dim_z = 2
        self.x = np.zeros(self.dim_x);
        self.P = np.eye(self.dim_x) * 500
        self.Q = np.diag([0.25, 0.25, process_noise, process_noise])
        self.R = np.eye(self.dim_z) * measurement_noise
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]);
        self.F = np.eye(self.dim_x)
        self.initialized = False

    def reset(self):
        self.initialized = False;
        self.x = np.zeros(self.dim_x);
        self.P = np.eye(self.dim_x) * 500

    def initialize(self, measurement):
        self.x[0:2] = measurement[0:2];
        self.initialized = True

    def predict(self, dt):
        if not self.initialized: return None, None
        self.F[0, 2] = dt;
        self.F[1, 3] = dt
        self.x = self.F @ self.x;
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, measurement):
        if not self.initialized: self.initialize(measurement); return
        y = measurement - self.H @ self.x
        if y[0] > 180:
            y[0] -= 360
        elif y[0] < -180:
            y[0] += 360
        S = self.H @ self.P @ self.H.T + self.R;
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y;
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P


class HandTracker:
    """
    Adaptive tracker using a Kalman Filter, dynamic scan radius, and coasting.
    """

    def __init__(self, min_scan_radius=5.0, max_scan_radius=12.0, max_speed_for_adapt=50.0,
                 scan_points=8, time_per_waypoint=0.045, timeout=1.0, coast_timeout=1.0):
        # --- ADAPTIVE PARAMETERS ---
        self.min_scan_radius = min_scan_radius
        self.max_scan_radius = max_scan_radius
        self.max_speed_for_adapt = max_speed_for_adapt  # Speed (deg/s) at which radius is maxed out

        # --- STATIC PARAMETERS ---
        self.scan_points = scan_points
        self.time_per_waypoint = time_per_waypoint
        self.timeout = timeout
        self.coast_timeout = coast_timeout

        self.state = HandTrackerState.IDLE
        self.best_point = {'az': 0, 'el': 0, 'strength': 0, 'time': 0, 'dist': 0}
        self.scan_path = [];
        self.scan_index = 0
        self.last_waypoint_time = self.last_scan_completion_time = 0
        self.coast_start_time = self.last_coast_update_time = 0

        self.kf = HandTrackerKalmanFilter(process_noise=80.0, measurement_noise=15.0)

    def reset(self):
        self.state = HandTrackerState.IDLE
        self.best_point['strength'] = 0
        self.kf.reset()
        print("[HandTracker] Reset.")

    def _generate_scan_path(self, center_az, center_el, radius):
        self.scan_path = []
        for i in range(self.scan_points):
            angle = (i / self.scan_points) * 2 * math.pi
            az_offset = radius * math.cos(angle)
            el_offset = radius * math.sin(angle)
            self.scan_path.append((center_az + az_offset, center_el + el_offset))
        norm_az = center_az % 360
        print(f"[HandTracker] New scan path. Center: Az={norm_az:.1f}, El={center_el:.1f}, Radius: {radius:.1f} deg")

    def update(self, current_az, current_el, measurement, shared_data):
        current_time = time.time()

        if self.state == HandTrackerState.IDLE:
            if measurement:
                dist, strength = measurement
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'strength': strength, 'dist': dist,
                                   'time': current_time}
                self.kf.initialize(np.array([current_az, current_el]))
                self._generate_scan_path(current_az, current_el, self.max_scan_radius)  # Start wide
                self.scan_index = 0;
                self.last_waypoint_time = self.last_scan_completion_time = current_time
                print(f"[HandTracker] Acquired target. Starting adaptive tracking.")

        elif self.state == HandTrackerState.SCANNING:
            if current_time - self.best_point['time'] > self.timeout:
                print("[HandTracker] Target lost. Entering COASTING mode.")
                self.state = HandTrackerState.COASTING
                self.coast_start_time = self.last_coast_update_time = current_time
                return

            if measurement and measurement[1] > self.best_point['strength']:
                dist, strength = measurement
                self.best_point = {'az': current_az, 'el': current_el, 'strength': strength, 'dist': dist,
                                   'time': current_time}

            if current_time - self.last_waypoint_time >= self.time_per_waypoint:
                waypoints_to_advance = max(1, int((current_time - self.last_waypoint_time) / self.time_per_waypoint))
                self.last_waypoint_time = current_time
                for _ in range(waypoints_to_advance):
                    command_az, command_el = self.scan_path[self.scan_index]
                    command_motors_to_target(command_az, command_el, shared_data)
                    self.scan_index = (self.scan_index + 1)

                    if self.scan_index >= len(self.scan_path):
                        self.scan_index = 0
                        dt = current_time - self.last_scan_completion_time
                        self.last_scan_completion_time = current_time

                        # Update filter with the best point found
                        measurement_vec = np.array([self.best_point['az'], self.best_point['el']])
                        self.kf.update(measurement_vec)

                        # --- ADAPTIVE SCAN RADIUS ---
                        # 1. Get current speed from Kalman Filter state
                        current_speed = np.linalg.norm(self.kf.x[2:])  # Magnitude of velocity vector

                        # 2. Map speed to radius (linear interpolation)
                        adaptive_radius = np.interp(
                            current_speed,
                            [0, self.max_speed_for_adapt],
                            [self.min_scan_radius, self.max_scan_radius]
                        )
                        adaptive_radius = np.clip(adaptive_radius, self.min_scan_radius, self.max_scan_radius)

                        # Predict next position
                        scan_duration = self.scan_points * self.time_per_waypoint
                        predicted_az, predicted_el = self.kf.predict(scan_duration)

                        if predicted_az is not None:
                            self._generate_scan_path(predicted_az, predicted_el, adaptive_radius)

                        self.best_point['strength'] *= 0.9

        elif self.state == HandTrackerState.COASTING:
            if current_time - self.coast_start_time > self.coast_timeout:
                print("[HandTracker] Coasting failed. Resetting.")
                command_motors_to_target(self.kf.x[0], self.kf.x[1], shared_data)
                self.reset()
                return

            if measurement:
                print("[HandTracker] Reacquired target during coast!")
                self.state = HandTrackerState.SCANNING
                dist, strength = measurement
                self.best_point = {'az': current_az, 'el': current_el, 'strength': strength, 'dist': dist,
                                   'time': current_time}
                self.kf.update(np.array([current_az, current_el]))
                self._generate_scan_path(current_az, current_el, self.max_scan_radius)  # Start wide again
                self.scan_index = 0;
                self.last_waypoint_time = current_time
                return

            dt = current_time - self.last_coast_update_time
            self.last_coast_update_time = current_time
            predicted_az, predicted_el = self.kf.predict(dt)
            if predicted_az is not None:
                command_motors_to_target(predicted_az, predicted_el, shared_data)


class ReactiveTracker:
    # ... (This class remains unchanged)
    def __init__(self, smoothing_factor=0.4):
        self.smoothing_factor = smoothing_factor; self.target_az = None; self.target_el = None; self.measurement_history = deque(
            maxlen=5); self.last_update_time = time.time()

    def update(self, azimuth, elevation, distance, strength):
        current_time = time.time();
        measurement = {'az': azimuth, 'el': elevation, 'dist': distance, 'strength': strength, 'time': current_time};
        self.measurement_history.append(measurement)
        adaptive_smoothing = self._calculate_adaptive_smoothing(strength)
        if self.target_az is None:
            self.target_az, self.target_el = azimuth, elevation
        else:
            az_diff = azimuth - self.target_az
            if az_diff > 180:
                az_diff -= 360
            elif az_diff < -180:
                az_diff += 360
            self.target_az += (az_diff * adaptive_smoothing);
            self.target_el = (self.target_el * (1 - adaptive_smoothing) + elevation * adaptive_smoothing);
            self.target_az %= 360
        self.last_update_time = current_time;
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
        return max(0.1, min(1.0, base_smoothing * strength_factor * consistency_factor))

    def _check_measurement_consistency(self):
        if len(self.measurement_history) < 3: return 1.0
        recent_az = [m['az'] for m in list(self.measurement_history)[-3:]];
        recent_el = [m['el'] for m in list(self.measurement_history)[-3:]]
        az_diffs = [abs(recent_az[i] - recent_az[i - 1]) for i in range(1, len(recent_az))];
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
    """Command the motor controller, with a safety clamp for max elevation."""

    # --- CEILING FIX ---
    # Clamp the elevation to the maximum allowed tracking angle.
    original_elevation = elevation
    elevation = min(elevation, MAX_TRACKING_ELEVATION)
    if elevation != original_elevation:
        print(f"[SafetyClamp] Capped elevation command from {original_elevation:.1f} to {elevation:.1f} deg.")

    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True


def run_tracking_logic(shared_data):
    # ... (The rest of this function remains unchanged)
    print("[TrackingLogic] Starting tracking logic process...")
    clutter_filter = ClutterFilter(shared_data.get("background_path", "background_data.npy").value)
    orbital_ekf = OrbitalEKF();
    acquirer = Acquirer();
    reactive_tracker = ReactiveTracker()
    hand_tracker = HandTracker()
    state = TrackingState.IDLE;
    last_prediction_time = time.time();
    prediction_interval = 0.1
    shared_data["tracking_logic_ready"].value = True
    print("[TrackingLogic] Ready and running...")
    print("  - debug_mode=True: Adaptive hand tracking with coasting")

    while not shared_data["shutdown"].value:
        try:
            current_time = time.time()
            with shared_data["lidar_data"].get_lock():
                dist, strength, timestamp = shared_data["lidar_data"][:]
            current_az = shared_data["stepper_degrees"].value;
            current_el = shared_data["servo_degrees"].value
            measurement_valid = (10.0 <= dist <= 16000.0 and shared_data.get("lidar_acceptance_range", [3.0, 50.0])[
                0] <= dist / 100.0 <= shared_data.get("lidar_acceptance_range", [3.0, 50.0])[1])

            if shared_data["debug_mode"].value:
                if state != TrackingState.DEBUG_MODE:
                    print("[TrackingLogic] Switching to DEBUG_MODE (Adaptive Hand Tracking)")
                    state = TrackingState.DEBUG_MODE;
                    hand_tracker.reset()
                is_valid_target = measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist,
                                                                                       strength)
                measurement_data = (dist, strength) if is_valid_target else None
                hand_tracker.update(current_az, current_el, measurement_data, shared_data)
                if hand_tracker.state in [HandTrackerState.SCANNING, HandTrackerState.COASTING]:
                    with shared_data["predicted_azimuth"].get_lock(): shared_data["predicted_azimuth"].value = \
                    hand_tracker.kf.x[0]
                    with shared_data["predicted_elevation"].get_lock(): shared_data["predicted_elevation"].value = \
                    hand_tracker.kf.x[1]

            # ... (other states remain unchanged) ...
            elif shared_data["reactive_mode"].value:
                if state != TrackingState.REACTIVE_MODE:
                    state = TrackingState.REACTIVE_MODE;
                    reactive_tracker = ReactiveTracker()
                if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                    target_az, target_el = reactive_tracker.update(current_az, current_el, dist, strength)
                    command_motors_to_target(target_az, target_el, shared_data)
                    with shared_data["predicted_azimuth"].get_lock(): shared_data["predicted_azimuth"].value = target_az
                    with shared_data["predicted_elevation"].get_lock(): shared_data[
                        "predicted_elevation"].value = target_el
            else:
                if shared_data["acquire_points"].value:
                    if state != TrackingState.ACQUIRING: state = TrackingState.ACQUIRING; acquirer = Acquirer();
                    shared_data["acquirer_status"].value = 1
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        if acquirer.add_measurement(current_az, current_el, dist, current_time):
                            initial_state = acquirer.compute_initial_state()
                            if initial_state is not None:
                                orbital_ekf.state = initial_state;
                                orbital_ekf.initialized = True
                                shared_data["ekf_initialized"].value = True;
                                shared_data["acquire_points"].value = False;
                                shared_data["acquirer_status"].value = 0
                elif shared_data["lidar_track_mode_active"].value and orbital_ekf.initialized:
                    if state != TrackingState.TRACKING: state = TrackingState.TRACKING
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist,
                                                                            strength): orbital_ekf.update(
                        [current_az, current_el, dist], strength)
                else:
                    if state != TrackingState.IDLE: state = TrackingState.IDLE

            if orbital_ekf.initialized and state == TrackingState.TRACKING:
                if current_time - last_prediction_time >= prediction_interval:
                    dt = current_time - orbital_ekf.last_update_time;
                    orbital_ekf.predict(min(dt, 1.0))
                    prediction = orbital_ekf.get_predicted_position(0.5)
                    if prediction is not None:
                        pred_az, pred_el, pred_dist = prediction
                        command_motors_to_target(pred_az, pred_el, shared_data)
                        with shared_data["predicted_azimuth"].get_lock(): shared_data[
                            "predicted_azimuth"].value = pred_az
                        with shared_data["predicted_elevation"].get_lock(): shared_data[
                            "predicted_elevation"].value = pred_el
                    last_prediction_time = current_time

            time.sleep(0.01)
        except Exception as e:
            print(f"[TrackingLogic] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    print("[TrackingLogic] Shutting down...")
