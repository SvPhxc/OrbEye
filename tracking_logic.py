# tracking_logic.py

import time
import threading
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import math
from enum import Enum

# --- All classes before HandTracker remain the same ---

class TrackingState(Enum):
    IDLE = 0
    ACQUIRING = 1
    TRACKING = 2
    DEBUG_MODE = 3
    REACTIVE_MODE = 4

class ClutterFilter:
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
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found. Running without clutter filtering.")
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
    # ... (code is unchanged)
    def __init__(self):
        self.state = np.zeros(6); self.P = np.eye(6) * 1000; self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        self.initialized = False; self.last_update_time = time.time()
    def predict(self, dt):
        if not self.initialized: return
        F = np.eye(6); F[0:3, 3:6] = np.eye(3) * dt
        r = np.linalg.norm(self.state[0:3])
        if r > 1.0:
            mu = 1000; acc_factor = -mu / (r ** 3)
            F[3, 0] = acc_factor * dt; F[4, 1] = acc_factor * dt; F[5, 2] = acc_factor * dt
        self.state = F @ self.state; self.P = F @ self.P @ F.T + self.Q
    def update(self, measurement, strength):
        if not self.initialized: return
        az, el, dist = measurement; az_rad, el_rad, dist_m = np.radians(az), np.radians(el), dist / 100.0
        z_meas = np.array([dist_m*np.cos(el_rad)*np.cos(az_rad), dist_m*np.cos(el_rad)*np.sin(az_rad), dist_m*np.sin(el_rad)])
        h_pred = self.state[0:3]; H = np.zeros((3, 6)); H[0:3, 0:3] = np.eye(3)
        base_var_pos = 1.0
        if strength > 500: pos_variance, angular_noise_factor = base_var_pos * 0.1, 1.0
        else: pos_variance, angular_noise_factor = base_var_pos * 1.0, 5.0
        R = np.diag([pos_variance, pos_variance*angular_noise_factor, pos_variance*angular_noise_factor])
        y = z_meas - h_pred; S = H @ self.P @ H.T + R; K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y; self.P = (np.eye(6) - K @ H) @ self.P
        self.last_update_time = time.time()
    def get_predicted_position(self, future_time_sec=0.5):
        if not self.initialized: return None
        temp_state = self.state.copy(); temp_state[0:3] += temp_state[3:6] * future_time_sec
        x, y, z = temp_state[0:3]; r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1.0: return None
        el = np.degrees(np.arcsin(z / r)); az = np.degrees(np.arctan2(y, x))
        if az < 0: az += 360
        return az, el, r * 100

class Acquirer:
    # ... (code is unchanged)
    def __init__(self): self.measurements = []; self.required_points = 3
    def add_measurement(self, azimuth, elevation, distance, timestamp):
        az_rad, el_rad, dist_m = np.radians(azimuth), np.radians(elevation), distance/100.0
        pos = np.array([dist_m*np.cos(el_rad)*np.cos(az_rad), dist_m*np.cos(el_rad)*np.sin(az_rad), dist_m*np.sin(el_rad)])
        self.measurements.append((pos, timestamp))
        if len(self.measurements) > self.required_points: self.measurements.pop(0)
        return len(self.measurements) >= self.required_points
    def compute_initial_state(self):
        if len(self.measurements) < self.required_points: return None
        positions = [m[0] for m in self.measurements]; times = [m[1] for m in self.measurements]
        r1, r2, r3 = positions; t1, t2, t3 = times; dt1, dt2 = t2-t1, t3-t2
        if dt1 <= 0 or dt2 <= 0: return None
        v2 = (r3 - r1) / (dt1 + dt2)
        return np.concatenate([r2, v2])

class HandTrackerState(Enum):
    IDLE = 0
    SCANNING = 1
    COASTING = 2

# --- CORRECTED HandTracker Class ---
class HandTracker:
    """
    High-performance predictive tracker with velocity smoothing, dynamic rate adjustment,
    and a predictive "coasting" search mode for target reacquisition.
    """

    def __init__(self, scan_radius=8, scan_points=8, time_per_waypoint=0.06, timeout=1.0, coast_timeout=1.5,
                 prediction_factor=0.75, velocity_smoothing_factor=0.5):
        self.scan_radius = scan_radius
        self.scan_points = scan_points
        self.time_per_waypoint = time_per_waypoint
        self.timeout = timeout
        self.coast_timeout = coast_timeout
        self.prediction_factor = prediction_factor
        self.velocity_smoothing_factor = velocity_smoothing_factor

        self.state = HandTrackerState.IDLE
        self.best_point = {'az': 0, 'el': 0, 'strength': 0, 'dist': 0, 'time': 0}
        self.previous_best_point = None
        self.scan_path = []
        self.scan_index = 0
        self.last_waypoint_time = 0
        self.smoothed_velocity = {'az': 0.0, 'el': 0.0}
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
            if current_time - self.best_point['time'] > self.timeout:
                print("[HandTracker] Target lost. Entering COASTING mode.")
                self.state = HandTrackerState.COASTING
                self.coast_start_time = current_time
                self.last_coast_update_time = current_time
                self.coasting_target_pos = {'az': self.best_point['az'], 'el': self.best_point['el']}
                return

            if measurement:
                dist, strength = measurement
                # --- FIX IS HERE ---
                # The invalid clutter_filter check has been removed.
                # The 'measurement' variable is already pre-validated.
                if strength > self.best_point['strength']:
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                       'time': current_time}

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
                            if raw_delta_az > 180: raw_delta_az -= 360
                            elif raw_delta_az < -180: raw_delta_az += 360
                            raw_delta_el = self.best_point['el'] - self.previous_best_point['el']
                            s = self.velocity_smoothing_factor
                            self.smoothed_velocity['az'] = (s * self.smoothed_velocity['az']) + ((1 - s) * raw_delta_az)
                            self.smoothed_velocity['el'] = (s * self.smoothed_velocity['el']) + ((1 - s) * raw_delta_el)
                            next_center_az = self.best_point['az'] + (self.smoothed_velocity['az'] * self.prediction_factor)
                            next_center_el = self.best_point['el'] + (self.smoothed_velocity['el'] * self.prediction_factor)

                        if "tracking_history" in shared_data:
                            point_data = [self.best_point['az'], self.best_point['el'], self.best_point['dist'],
                                          self.best_point['strength'], self.best_point['time']]
                            shared_data["tracking_history"].append(point_data)
                            if len(shared_data["tracking_history"]) >= 30:
                                print("[Tracking Logic] Triggering TLE generation.")
                                shared_data["generate_tle"].value = True

                        self.previous_best_point = self.best_point.copy()
                        self._generate_scan_path(next_center_az, next_center_el)
                        self.best_point['strength'] *= 0.9

        elif self.state == HandTrackerState.COASTING:
            if current_time - self.coast_start_time > self.coast_timeout:
                print("[HandTracker] Coasting failed to reacquire target. Resetting.")
                self.reset()
                return

            if measurement:
                dist, strength = measurement
                print("[HandTracker] Reacquired target during coast! Resuming scan.")
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                   'time': current_time}
                self._generate_scan_path(current_az, current_el)
                self.scan_index = 0
                self.last_waypoint_time = current_time
                return

            dt = current_time - self.last_coast_update_time
            self.last_coast_update_time = current_time
            predicted_delta_az = 3*self.smoothed_velocity['az'] * (dt / (self.scan_points * self.time_per_waypoint))
            predicted_delta_el = 3*self.smoothed_velocity['el'] * (dt / (self.scan_points * self.time_per_waypoint))
            self.coasting_target_pos['az'] += predicted_delta_az
            self.coasting_target_pos['el'] += predicted_delta_el
            command_motors_to_target(self.coasting_target_pos['az'], self.coasting_target_pos['el'], shared_data)

# --- All classes and functions after HandTracker remain the same ---

class ReactiveTracker:
    # ... (code is unchanged)
    def __init__(self, smoothing_factor=0.4): self.smoothing_factor=smoothing_factor; self.target_az=None; self.target_el=None; self.measurement_history=deque(maxlen=5); self.last_update_time=time.time()
    def update(self, azimuth, elevation, distance, strength):
        current_time=time.time(); measurement={'az':azimuth,'el':elevation,'dist':distance,'strength':strength,'time':curren