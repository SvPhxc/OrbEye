# LiDAR/Kalman_Filter.py

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
import traceback

try:
    from analysis.plotter import plot_ekf_vs_measured
except ImportError:
    print("[EKF] Warning: Plotter module not found. Plotting will be disabled.")


    def plot_ekf_vs_measured(h_m, h_e):
        pass


# region Helper Functions
def spherical_to_cartesian(az_rad, el_rad, dist):
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if dist == 0: return 0, 0, 0
    az = np.arctan2(y, x)
    el = np.arcsin(z / dist)
    return az, el, dist


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def state_to_angles(state):
    x, y, z = state[0], state[1], state[2]
    az_rad, el_rad, _ = cartesian_to_spherical(x, y, z)
    return np.rad2deg(normalize_angle(az_rad)), np.rad2deg(el_rad)


def angle_difference(angle1, angle2):
    return normalize_angle(angle1 - angle2)


# endregion

class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    def __init__(self, std_acc, distance_constraint):
        super().__init__(dim_x=6, dim_z=3)
        self.std_acc = std_acc
        self.distance_constraint = distance_constraint
        self.P = np.eye(6) * 10.0
        self.P[3:, 3:] *= 2.0
        self.initialized = False
        self.last_time = None

    def update_matrices(self, dt):
        self.F = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        q_pos = Q_discrete_white_noise(dim=3, dt=dt, var=self.std_acc ** 2, block_size=1)
        q_vel = np.eye(3) * (self.std_acc * dt) ** 2
        self.Q = block_diag(q_pos, q_vel)

    def h(self, x):
        az, el, dist = cartesian_to_spherical(x[0], x[1], x[2])
        return np.array([az, el, dist])

    def HJacobian(self, x):
        H = np.zeros((3, 6))
        x0, x1, x2 = x[0], x[1], x[2]
        eps = 1e-6
        x_sq_y_sq = x0 ** 2 + x1 ** 2
        dist_sq = x_sq_y_sq + x2 ** 2
        if x_sq_y_sq < eps or dist_sq < eps: return H
        dist = np.sqrt(dist_sq)
        sqrt_x_sq_y_sq = np.sqrt(x_sq_y_sq)
        H[0, 0] = -x1 / x_sq_y_sq
        H[0, 1] = x0 / x_sq_y_sq
        H[1, 0] = -x0 * x2 / (sqrt_x_sq_y_sq * dist_sq)
        H[1, 1] = -x1 * x2 / (sqrt_x_sq_y_sq * dist_sq)
        H[1, 2] = sqrt_x_sq_y_sq / dist_sq
        H[2, 0] = x0 / dist
        H[2, 1] = x1 / dist
        H[2, 2] = x2 / dist
        return H

    def predict(self):
        super().predict()
        if self.distance_constraint:
            current_dist = np.linalg.norm(self.x[0:3])
            if current_dist > 12.0:
                self.x[0:3] *= 12.0 / current_dist
            elif current_dist < 3.0:
                self.x[0:3] *= 3.0 / current_dist

    def update_with_angle_wrapping(self, z, HJacobian, Hx, R):
        hx = Hx(self.x)
        y = z - hx
        y[0] = angle_difference(z[0], hx[0])
        H = HJacobian(self.x)
        PHT = self.P @ H.T
        S = H @ PHT + R
        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHT @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_x) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


# region EKF Helper Functions
def init_ekf(ekf, initial_points):
    meas1, meas2 = initial_points[0], initial_points[1]
    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])
    dt = meas2['time'] - meas1['time']
    if dt <= 0.01: dt = 0.1
    vx, vy, vz = (x2 - x1) / dt, (y2 - y1) / dt, (z2 - z1) / dt
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])


def create_measurement_noise_matrix(strength, measurement_count):
    base_angular_var = (np.deg2rad(0.5)) ** 2
    base_dist_var = (0.05) ** 2
    strength_factor = max(1.0, strength / 1000.0)
    angular_var = base_angular_var / strength_factor
    dist_var = base_dist_var / strength_factor
    return np.diag([angular_var, angular_var, dist_var])


def get_next_prediction(ekf, dt):
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt)
    temp_ekf.predict()
    return state_to_angles(temp_ekf.x)


def calculate_confidence(P):
    position_uncertainty = np.trace(P[0:3, 0:3])
    return min(max(1.0 / (1.0 + position_uncertainty), 0.0), 1.0)


# endregion

def run_ekf_tracker(shared_data):
    print("[EKF] Starting tracker process...")
    ekf = None
    history_measurements, history_estimates = [], []

    while not shared_data['shutdown'].value:
        try:
            if ekf is None or not ekf.initialized:
                if shared_data['ekf_start'].value and shared_data['points_count'].value >= 2:
                    print("[EKF] Initialization signal received.")
                    is_debug = shared_data["debug_mode"].value
                    ekf = CloseRangeDroneTrackerEKF(std_acc=0.5 if is_debug else 0.04,
                                                    distance_constraint=not is_debug)
                    history_measurements.clear();
                    history_estimates.clear()

                    # --- This now correctly reads the 5-element point data ---
                    pb = shared_data['points_buffer']
                    p1 = {'z': np.array([np.deg2rad(pb[0]), np.deg2rad(pb[1]), pb[2]]), 'time': pb[4]}
                    p2 = {'z': np.array([np.deg2rad(pb[5]), np.deg2rad(pb[6]), pb[7]]), 'time': pb[9]}
                    init_ekf(ekf, [p1, p2])

                    ekf.last_time = p2['time']
                    ekf.initialized = True
                    shared_data['ekf_initialized'].value = True
                    shared_data['ekf_running'].value = True
                    shared_data['ekf_start'].value = False
                    shared_data['points_count'].value = 0
                    print("[EKF] Initialized successfully. Tracking started.")
                else:
                    time.sleep(0.1);
                    continue

            now = time.time()
            if shared_data['ekf_running'].value:
                dt = now - (ekf.last_time if ekf.last_time is not None else now)
                if dt <= 0: dt = 1e-3
                ekf.update_matrices(dt);
                ekf.predict()

                with shared_data["lidar_data"].get_lock():
                    dist_cm, strength, ts = shared_data["lidar_data"]
                az, el = shared_data["stepper_degrees"].value, shared_data["servo_degrees"].value
                min_m, max_m = shared_data["lidar_acceptance_range"]
                has_new = (not history_measurements) or (ts > history_measurements[-1]['time'])
                valid_range = (min_m <= dist_cm / 100.0 <= max_m)
                min_strength = 1000 if shared_data["debug_mode"].value else 5000
                valid_strength = (strength >= min_strength)

                if has_new and valid_range and valid_strength:
                    z = np.array([np.deg2rad(az), np.deg2rad(el), dist_cm / 100.0])
                    R = create_measurement_noise_matrix(strength, len(history_measurements))
                    ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                    history_measurements.append({'z': z, 'time': ts})

                history_estimates.append(ekf.x.copy())
                ekf.last_time = now
                pred_az, pred_el = get_next_prediction(ekf, max(dt, 0.02))
                shared_data["predicted_azimuth"].value = pred_az
                shared_data["predicted_elevation"].value = pred_el
                shared_data["ekf_confidence"].value = calculate_confidence(ekf.P)
            else:
                if shared_data['generate_plot_on_stop'].value:
                    print("[EKF] EKF stopped. Generating final plot...")
                    plot_ekf_vs_measured(history_measurements, history_estimates)
                    shared_data['generate_plot_on_stop'].value = False
                ekf.initialized = False
                shared_data['ekf_initialized'].value = False
                time.sleep(0.1)

            time.sleep(0.01)
        except Exception as e:
            print(f"[EKF] CRITICAL ERROR in tracking loop: {e}");
            traceback.print_exc()
            if ekf: ekf.initialized = False
            shared_data['ekf_running'].value = False
            time.sleep(1)
    print("[EKF] Shutting down.")