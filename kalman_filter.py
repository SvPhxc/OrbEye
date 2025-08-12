# kalman_filter.py

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
from multiprocessing import Process, Array, Value


# (Helper Functions: spherical_to_cartesian, cartesian_to_spherical, etc. remain the same)
def spherical_to_cartesian(az_rad, el_rad, dist):
    """Converts spherical coordinates (azimuth, elevation, distance) to Cartesian (x, y, z)."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to spherical (azimuth, elevation, distance)."""
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return az, el, dist


def normalize_angle(angle):
    """Normalize angle to [-π, π] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def state_to_angles(state):
    """Convert state vector to azimuth and elevation in degrees."""
    x, y, z = state[0], state[1], state[2]
    az, el, _ = cartesian_to_spherical(x, y, z)
    return np.rad2deg(normalize_angle(az)), np.rad2deg(el)


def angle_difference(angle1, angle2):
    """Calculate the shortest angular distance between two angles."""
    diff = angle1 - angle2
    return normalize_angle(diff)


class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    # (Class implementation remains the same)
    def __init__(self, std_acc=0.04, distance_constraint=True):
        dim_x = 6  # State: [x, y, z, vx, vy, vz]
        dim_z = 3  # Measurement: [azimuth, elevation, distance]
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.std_acc = std_acc
        self.distance_constraint = distance_constraint
        self.P = np.eye(6) * 5
        self.P[0:2, 0:2] *= 1.0
        self.P[2, 2] *= 0.5
        self.P[3:5, 3:5] *= 0.3
        self.P[5, 5] *= 0.1
        self.initialized = False
        self.last_time = None

    def update_matrices(self, dt):
        self.F = np.array(
            [[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        q_block = Q_discrete_white_noise(dim=2, dt=dt, var=self.std_acc ** 2)
        self.Q = block_diag(q_block, q_block, q_block)

    def h(self, x):
        az, el, dist = cartesian_to_spherical(x[0], x[1], x[2])
        return np.array([az, el, dist])

    def HJacobian(self, x):
        H = np.zeros((3, 6));
        x0, x1, x2 = x[0], x[1], x[2];
        eps = 1e-6
        x_sq_y_sq = x0 ** 2 + x1 ** 2 + eps;
        dist_sq = x_sq_y_sq + x2 ** 2 + eps
        dist = np.sqrt(dist_sq);
        sqrt_x_sq_y_sq = np.sqrt(x_sq_y_sq)
        H[0, 0] = -x1 / x_sq_y_sq;
        H[0, 1] = x0 / x_sq_y_sq
        H[1, 0] = -x0 * x2 / (sqrt_x_sq_y_sq * dist_sq);
        H[1, 1] = -x1 * x2 / (sqrt_x_sq_y_sq * dist_sq)
        H[1, 2] = sqrt_x_sq_y_sq / dist_sq
        H[2, 0] = x0 / dist;
        H[2, 1] = x1 / dist;
        H[2, 2] = x2 / dist
        return H

    def predict(self):
        super().predict()
        if self.distance_constraint:
            current_dist = np.sqrt(self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2)
            if current_dist < 6.0:
                self.x[0:3] *= 6.0 / current_dist
            elif current_dist > 12.0:
                self.x[0:3] *= 12.0 / current_dist

    def update_with_angle_wrapping(self, z, HJacobian, Hx, R):
        hx = Hx(self.x);
        y = z - hx;
        y[0] = angle_difference(z[0], hx[0])
        if abs(y[1]) > np.deg2rad(2.0): y[1] = np.sign(y[1]) * np.deg2rad(2.0)
        H = HJacobian(self.x);
        PHT = self.P @ H.T;
        S = H @ PHT + R
        S += np.eye(S.shape[0]) * 1e-8
        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHT @ np.linalg.pinv(S)
        K_modified = K.copy();
        K_modified[:, 1] *= 0.8
        self.x = self.x + K_modified @ y
        I_KH = np.eye(self.x.shape[0]) - K_modified @ H
        self.P = I_KH @ self.P @ I_KH.T + K_modified @ R @ K_modified.T


def run_ekf_tracker(shared_data):
    print("[EKF] Starting tracker...")
    ekf = CloseRangeDroneTrackerEKF(std_acc=0.04, distance_constraint=True)
    waiting_for_init = True
    last_measurement_time = None
    measurement_count = 0

    while not shared_data["shutdown"].value:
        try:
            now = time.time()

            # --- EKF Initialization Logic (remains the same) ---
            if waiting_for_init and shared_data['ekf_start'].value:
                if shared_data['points_count'].value >= 2:
                    pb = shared_data['points_buffer']
                    z1 = np.array([np.deg2rad(pb[0]), np.deg2rad(pb[1]), pb[2]])
                    z2 = np.array([np.deg2rad(pb[4]), np.deg2rad(pb[5]), pb[6]])
                    init_buffer = [{'z': z1, 'time': now}, {'z': z2, 'time': now + 0.1}]
                    init_ekf(ekf, init_buffer)
                    ekf.initialized = True
                    ekf.last_time = now
                    shared_data['ekf_initialized'].value = True
                    shared_data['ekf_running'].value = True
                    waiting_for_init = False
                    print("[EKF] Initialized from acquired points.")
                    # Optional 3rd point update
                    if shared_data['points_count'].value >= 3:
                        z3 = np.array([np.deg2rad(pb[8]), np.deg2rad(pb[9]), pb[10]])
                        R3 = create_measurement_noise_matrix(pb[11], 0)
                        ekf.update_with_angle_wrapping(z3, ekf.HJacobian, ekf.h, R3)
                        last_measurement_time = now + 0.2

            if not ekf.initialized or not shared_data["ekf_running"].value:
                time.sleep(0.02)
                continue

            # --- Prediction Step ---
            dt = now - (ekf.last_time if ekf.last_time is not None else now)
            if dt <= 0.0: dt = 1e-3
            ekf.update_matrices(dt)
            ekf.predict()

            # --- Update Step ---
            # *** NEW: Prioritize high-confidence points from ActiveTracker ***
            measurement_updated = False
            if shared_data["satellite_detected"].value:
                with shared_data["satellite_points"].get_lock():
                    sp = shared_data["satellite_points"]
                    z = np.array([np.deg2rad(sp[0]), np.deg2rad(sp[1]), sp[2] / 100.0])
                    strength = sp[3]
                    ts = sp[4]
                    # Reset the flag after reading
                    shared_data["satellite_detected"].value = False

                # Use a very low noise for these high-confidence points
                R = create_measurement_noise_matrix(strength, 0, is_satellite=True)
                ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                last_measurement_time = ts
                measurement_updated = True
                print(f"[EKF] Updated with satellite point. Strength: {strength:.0f}")

            # If no satellite point, use standard LiDAR reading
            if not measurement_updated:
                with shared_data["lidar_data"].get_lock():
                    dist_cm, strength, ts = shared_data["lidar_data"]

                has_new = (last_measurement_time is None) or (ts > last_measurement_time)
                min_m, max_m = shared_data["lidar_acceptance_range"]
                is_valid = (min_m <= dist_cm / 100.0 <= max_m) and (strength > 4000)

                if has_new and is_valid:
                    with shared_data["stepper_degrees"].get_lock():
                        az_deg = shared_data["stepper_degrees"].value
                    with shared_data["servo_degrees"].get_lock():
                        el_deg = shared_data["servo_degrees"].value

                    z = np.array([np.deg2rad(az_deg), np.deg2rad(el_deg), dist_cm / 100.0])
                    R = create_measurement_noise_matrix(strength, measurement_count)
                    ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                    last_measurement_time = ts
                    measurement_count += 1

            # --- Publish Outputs ---
            est_az, est_el = state_to_angles(ekf.x)
            pred_az, pred_el = get_next_prediction(ekf, max(dt, 0.02))
            conf = calculate_confidence(ekf.P)

            shared_data['estimated_azimuth'].value = est_az
            shared_data['estimated_elevation'].value = est_el
            shared_data['predicted_azimuth'].value = pred_az
            shared_data['predicted_elevation'].value = pred_el
            shared_data['ekf_confidence'].value = conf

            ekf.last_time = now
            time.sleep(0.01)

        except Exception as e:
            print(f"[EKF] Error: {e}")
            time.sleep(0.05)

    print("[EKF] Shutting down...")


def init_ekf(ekf, initialization_buffer):
    # (Function remains the same)
    meas1, meas2 = initialization_buffer[0], initialization_buffer[1]
    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])
    dt = meas2['time'] - meas1['time']
    if dt <= 0: dt = 0.5
    vx = (x2 - x1) / dt;
    vy = (y2 - y1) / dt;
    vz = (z2 - z1) / dt
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])


def create_measurement_noise_matrix(strength, measurement_count, is_satellite=False):
    """Create R matrix. Satellite points get much lower noise."""
    if is_satellite:
        # Very high confidence in satellite points
        angular_var = (np.deg2rad(0.1)) ** 2
        dist_var = 0.01 ** 2
    else:
        # Standard noise model
        base_angular_var = (np.deg2rad(0.3)) ** 2;
        base_dist_var = 0.03 ** 2
        strength_factor = max(strength, 1.0)
        angular_var = base_angular_var / strength_factor ** 0.5
        dist_var = base_dist_var / strength_factor ** 0.5

    elevation_var = angular_var * 0.4  # Assume better elevation precision
    return np.diag([angular_var, elevation_var, dist_var])


def get_next_prediction(ekf, dt):
    # (Function remains the same)
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt)
    temp_ekf.predict()
    return state_to_angles(temp_ekf.x)


def calculate_confidence(P):
    # (Function remains the same)
    position_uncertainty = np.trace(P[0:3, 0:3])
    confidence = 1.0 / (1.0 + position_uncertainty)
    return min(max(confidence, 0.0), 1.0)