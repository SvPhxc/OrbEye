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
    def plot_ekf_vs_measured(h_m, h_e):
        pass


# (Helper functions like spherical_to_cartesian, etc. are unchanged)
def spherical_to_cartesian(az, el, d): return d * np.cos(el) * np.cos(az), d * np.cos(el) * np.sin(az), d * np.sin(el)


def cartesian_to_spherical(x, y, z): d = np.sqrt(x ** 2 + y ** 2 + z ** 2); return (0, 0, 0) if d == 0 else (
    np.arctan2(y, x), np.arcsin(z / d), d)


def normalize_angle(a): return np.arctan2(np.sin(a), np.cos(a))


def state_to_angles(s): az, el, _ = cartesian_to_spherical(s[0], s[1], s[2]); return np.rad2deg(
    normalize_angle(az)), np.rad2deg(el)


def angle_difference(a1, a2): return normalize_angle(a1 - a2)


class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    # (The class definition is unchanged)
    def __init__(self, std_acc, distance_constraint):
        super().__init__(dim_x=6, dim_z=3)
        self.std_acc, self.distance_constraint = std_acc, distance_constraint
        self.P = np.eye(6) * 10.;
        self.P[3:, 3:] *= 2.
        self.initialized, self.last_time = False, None

    def update_matrices(self, dt):
        self.F = np.array(
            [[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        q_p = Q_discrete_white_noise(3, dt, self.std_acc ** 2);
        q_v = np.eye(3) * (self.std_acc * dt) ** 2;
        self.Q = block_diag(q_p, q_v)

    def h(self, x):
        az, el, d = cartesian_to_spherical(x[0], x[1], x[2]); return np.array([az, el, d])

    def HJacobian(self, x):
        H = np.zeros((3, 6));
        x0, x1, x2 = x[0], x[1], x[2];
        xsq_ysq = x0 ** 2 + x1 ** 2;
        d_sq = xsq_ysq + x2 ** 2
        if xsq_ysq < 1e-6 or d_sq < 1e-6: return H
        d = np.sqrt(d_sq);
        sd = np.sqrt(xsq_ysq);
        H[0, 0] = -x1 / xsq_ysq;
        H[0, 1] = x0 / xsq_ysq;
        H[1, 0] = -x0 * x2 / (sd * d_sq);
        H[1, 1] = -x1 * x2 / (sd * d_sq)
        H[1, 2] = sd / d_sq;
        H[2, 0] = x0 / d;
        H[2, 1] = x1 / d;
        H[2, 2] = x2 / d;
        return H

    def predict(self):
        super().predict()
        if self.distance_constraint:
            d = np.linalg.norm(self.x[0:3]);
            s = 1.
            if d > 12.:
                s = 12. / d
            elif d < 3.:
                s = 3. / d
            self.x[0:3] *= s

    def update_with_angle_wrapping(self, z, R):
        y = z - self.h(self.x);
        y[0] = angle_difference(z[0], self.h(self.x)[0]);
        H = self.HJacobian(self.x)
        PHT = self.P @ H.T;
        S = H @ PHT + R;
        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHT @ np.linalg.pinv(S)
        self.x = self.x + K @ y;
        I_KH = np.eye(6) - K @ H;
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


def init_ekf(ekf, initial_points):
    p1, p2 = initial_points[0], initial_points[1];
    x1, y1, z1 = spherical_to_cartesian(*p1['z'])
    x2, y2, z2 = spherical_to_cartesian(*p2['z']);
    dt = p2['time'] - p1['time']
    if dt <= 0.01: dt = 0.1
    ekf.x = np.array([x2, y2, z2, (x2 - x1) / dt, (y2 - y1) / dt, (z2 - z1) / dt])


def create_measurement_noise_matrix(strength):
    s_f = max(1., strength / 1000.);
    ang_v = (np.deg2rad(0.5)) ** 2 / s_f;
    d_v = (0.05) ** 2 / s_f
    return np.diag([ang_v, ang_v, d_v])


def get_next_prediction(ekf, dt):
    t = copy.deepcopy(ekf);
    t.update_matrices(dt);
    t.predict();
    return state_to_angles(t.x)


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
                    pb = shared_data['points_buffer']
                    p1 = {'z': np.array([np.deg2rad(pb[0]), np.deg2rad(pb[1]), pb[2]]), 'time': pb[4]}
                    p2 = {'z': np.array([np.deg2rad(pb[5]), np.deg2rad(pb[6]), pb[7]]), 'time': pb[9]}
                    init_ekf(ekf, [p1, p2]);
                    ekf.last_time = p2['time']
                    ekf.initialized = True;
                    shared_data['ekf_initialized'].value = True
                    shared_data['ekf_running'].value = True;
                    shared_data['ekf_start'].value = False
                    print("[EKF] Initialized. Tracking active.")
                else:
                    time.sleep(0.1);
                    continue

            # --- Main Tracking Loop ---
            if shared_data['ekf_running'].value:
                now = time.time()
                dt = now - (ekf.last_time or now)
                if dt > 0: ekf.update_matrices(dt); ekf.predict()

                # --- FIX: UPDATE ONLY ON REFINED MEASUREMENT ---
                if shared_data["satellite_detected"].value:
                    with shared_data["satellite_points"].get_lock():
                        az, el, dist_cm, strength, ts = shared_data["satellite_points"]
                        # Consume the point
                        shared_data["satellite_detected"].value = False

                    z = np.array([np.deg2rad(az), np.deg2rad(el), dist_cm / 100.0])
                    R = create_measurement_noise_matrix(strength)
                    ekf.update_with_angle_wrapping(z, R)

                    # Log for plotting
                    history_measurements.append({'z': z, 'time': ts})
                    history_estimates.append(ekf.x.copy())
                    ekf.last_time = ts  # Use measurement time for better accuracy

                # Always provide a prediction for the hunter
                pred_az, pred_el = get_next_prediction(ekf, 0.05)  # Predict 50ms ahead
                shared_data["predicted_azimuth"].value = pred_az
                shared_data["predicted_elevation"].value = pred_el
            else:
                if ekf and ekf.initialized:
                    if shared_data['generate_plot_on_stop'].value:
                        print("[EKF] EKF stopped. Generating plot...")
                        plot_ekf_vs_measured(history_measurements, history_estimates)
                        shared_data['generate_plot_on_stop'].value = False
                    ekf.initialized = False;
                    shared_data['ekf_initialized'].value = False
                    print("[EKF] Tracker reset.")
                time.sleep(0.1)

            time.sleep(0.01)
        except Exception as e:
            print(f"[EKF] CRITICAL ERROR: {e}");
            traceback.print_exc()
            if ekf: ekf.initialized = False
            shared_data['ekf_running'].value = False
            time.sleep(1)
    print("[EKF] Shutting down.")