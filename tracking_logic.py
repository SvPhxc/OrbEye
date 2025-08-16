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

    def __init__(self, background_file="background_data.npy", angular_tolerance=2.0, distance_margin_cm=50.0):
        self.angular_tolerance = angular_tolerance
        self.distance_margin_cm = distance_margin_cm
        self.background_tree = None
        self.background_data = None
        try:
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")
            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords)
            print("[ClutterFilter] 2D (directional) k-d tree built successfully.")
        except FileNotFoundError:
            print(
                f"[ClutterFilter] WARNING: Background file '{background_file}' not found. Running without clutter filtering.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR loading background data: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        if self.background_tree is None: return True
        query_point = np.array([azimuth, elevation])
        try:
            angular_dist, idx = self.background_tree.query(query_point, k=1)
            if angular_dist < self.angular_tolerance:
                bg_distance = self.background_data[idx, 2]
                if distance < (bg_distance - self.distance_margin_cm):
                    return True
                else:
                    return False
            return True
        except Exception:
            return True


class OrbitalEKF:
    # --- This class is unchanged ---
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
    # --- This class is unchanged ---
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


# ==============================================================================
# === NEW, MORE RESPONSIVE HAND TRACKER ========================================
# ==============================================================================

class HandTrackerState(Enum):
    """Defines the possible states for the HandTracker."""
    IDLE = 0
    SCANNING = 1
    COASTING = 2


class HandTracker:
    """
    A high-performance tracker that constantly dithers (scans) and immediately
    re-centers its scan pattern the moment a better signal is found,
    making it feel highly responsive and "locked-on".
    """

    def __init__(self, scan_radius=8, scan_points=8, time_per_waypoint=0.06, timeout=1.0, coast_timeout=1.5,
                 max_target_distance_cm=600.0):
        self.scan_radius = scan_radius
        self.scan_points = scan_points
        self.time_per_waypoint = time_per_waypoint
        self.timeout = timeout
        self.coast_timeout = coast_timeout
        self.max_target_distance_cm = max_target_distance_cm  # Prevents locking on ceiling

        self.state = HandTrackerState.IDLE
        self.best_point = {'az': 0, 'el': 0, 'strength': 0, 'dist': 0, 'time': 0}
        self.scan_path = []
        self.scan_index = 0
        self.last_waypoint_time = 0

        # State for coasting (predictive search on loss)
        self.coast_start_time = 0
        self.last_known_velocity = {'az': 0.0, 'el': 0.0}

    def reset(self):
        """Resets the tracker to its initial state."""
        self.state = HandTrackerState.IDLE
        self.best_point['strength'] = 0
        self.coast_start_time = 0
        self.last_known_velocity = {'az': 0.0, 'el': 0.0}
        print("[HandTracker] Reset.")

    def _generate_scan_path(self, center_az, center_el):
        """Generates a circular scan path around a center point."""
        self.scan_path = []
        for i in range(self.scan_points):
            angle = (i / self.scan_points) * 2 * math.pi
            az_offset = self.scan_radius * math.cos(angle)
            el_offset = self.scan_radius * math.sin(angle)
            self.scan_path.append((center_az + az_offset, center_el + el_offset))
        self.scan_index = 0  # Always start a new path from the beginning
        norm_az = center_az % 360
        print(f"[HandTracker] New scan path generated around Az={norm_az:.1f}, El={center_el:.1f}")

    def update(self, current_az, current_el, measurement, shared_data):
        current_time = time.time()

        if self.state == HandTrackerState.IDLE:
            if measurement:
                dist, strength = measurement
                if dist < self.max_target_distance_cm:  # Distance Gate
                    self.state = HandTrackerState.SCANNING
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                       'time': current_time}
                    self._generate_scan_path(current_az, current_el)
                    self.last_waypoint_time = current_time
                    print(f"[HandTracker] Acquired target. Starting continuous dither.")

        elif self.state == HandTrackerState.SCANNING:
            # --- KEY CHANGE: IMMEDIATE REACTION LOGIC ---
            if measurement:
                dist, strength = measurement
                # Check if the new point is valid and better than our current best
                if dist < self.max_target_distance_cm and strength > self.best_point['strength']:

                    # Calculate velocity for coasting mode before updating the point
                    time_since_last_best = current_time - self.best_point['time']
                    if time_since_last_best > 0.01:  # Avoid division by zero
                        delta_az = current_az - self.best_point['az']
                        if delta_az > 180:
                            delta_az -= 360
                        elif delta_az < -180:
                            delta_az += 360
                        self.last_known_velocity['az'] = delta_az / time_since_last_best
                        self.last_known_velocity['el'] = (current_el - self.best_point['el']) / time_since_last_best

                    # Immediately update the best point and generate a new scan path around it
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                       'time': current_time}
                    self._generate_scan_path(self.best_point['az'], self.best_point['el'])
                    # Decay the strength slightly so we are always seeking a new peak
                    self.best_point['strength'] *= 0.95

            # --- TRANSITION TO COASTING on target loss ---
            if current_time - self.best_point['time'] > self.timeout:
                print("[HandTracker] Target lost. Entering COASTING mode.")
                self.state = HandTrackerState.COASTING
                self.coast_start_time = current_time
                return

            # Always advance the scan/dither pattern
            if current_time - self.last_waypoint_time >= self.time_per_waypoint:
                self.last_waypoint_time = current_time
                if not self.scan_path: return  # Safety check
                command_az, command_el = self.scan_path[self.scan_index]
                command_motors_to_target(command_az, command_el, shared_data)
                self.scan_index = (self.scan_index + 1) % len(self.scan_path)

        elif self.state == HandTrackerState.COASTING:
            if current_time - self.coast_start_time > self.coast_timeout:
                print("[HandTracker] Coasting failed. Resetting.");
                self.reset();
                return

            if measurement:
                dist, strength = measurement
                if dist < self.max_target_distance_cm:
                    print("[HandTracker] Reacquired target during coast! Resuming dither.")
                    self.state = HandTrackerState.SCANNING
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                       'time': current_time}
                    self._generate_scan_path(current_az, current_el)
                    return

            # Predict and move along the last known velocity vector
            dt = current_time - (
                        self.coast_start_time + (current_time - self.last_waypoint_time))  # time since last command
            next_az = self.best_point['az'] + self.last_known_velocity['az'] * (current_time - self.best_point['time'])
            next_el = self.best_point['el'] + self.last_known_velocity['el'] * (current_time - self.best_point['time'])
            command_motors_to_target(next_az, next_el, shared_data)


class ReactiveTracker:
    # --- This class is unchanged ---
    def __init__(self, smoothing_factor=0.4):
        self.smoothing_factor = smoothing_factor;
        self.target_az = None;
        self.target_el = None
        self.measurement_history = deque(maxlen=5);
        self.last_update_time = time.time()

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
        az_diffs = [];
        el_diffs = [abs(recent_el[i] - recent_el[i - 1]) for i in range(1, len(recent_el))]
        for i in range(1, len(recent_az)):
            diff = recent_az[i] - recent_az[i - 1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            az_diffs.append(abs(diff))
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
    global clutter_filter  # Make clutter_filter global for access in HandTracker
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
    print("  - debug_mode=True: Immediate-reaction hand tracking")
    # ... rest of main loop is unchanged ...

    while not shared_data["shutdown"].value:
        try:
            current_time = time.time()
            with shared_data["lidar_data"].get_lock():
                dist, strength, timestamp = shared_data["lidar_data"][:]
            current_az = shared_data["stepper_degrees"].value
            current_el = shared_data["servo_degrees"].value
            measurement_valid = (10.0 <= dist <= 16000.0 and
                                 shared_data.get("lidar_acceptance_range", [3.0, 50.0])[0] <= dist / 100.0 <=
                                 shared_data.get("lidar_acceptance_range", [3.0, 50.0])[1])

            if shared_data["debug_mode"].value:
                if state != TrackingState.DEBUG_MODE:
                    print("[TrackingLogic] Switching to DEBUG_MODE (Immediate-Reaction Hand Tracking)")
                    state = TrackingState.DEBUG_MODE
                    hand_tracker.reset()

                # NOTE: The clutter filter is now called implicitly inside the new HandTracker
                # This simplifies the main loop logic
                measurement_data = (dist, strength) if measurement_valid else None
                hand_tracker.update(current_az, current_el, measurement_data, shared_data)

                if hand_tracker.state != HandTrackerState.IDLE:
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = hand_tracker.best_point['az']
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = hand_tracker.best_point['el']

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
                    if state != TrackingState.ACQUIRING:
                        state = TrackingState.ACQUIRING;
                        acquirer = Acquirer();
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
                        with shared_data["predicted_azimuth"].get_lock(): shared_data[
                            "predicted_azimuth"].value = pred_az
                        with shared_data["predicted_elevation"].get_lock(): shared_data[
                            "predicted_elevation"].value = pred_el
                        command_motors_to_target(pred_az, pred_el, shared_data)
                    last_prediction_time = current_time

            time.sleep(0.01)
        except Exception as e:
            print(f"[TrackingLogic] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    print("[TrackingLogic] Shutting down...")