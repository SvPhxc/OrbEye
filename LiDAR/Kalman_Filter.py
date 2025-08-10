# ==============================================================================
# LiDAR/Kalman_Filter.py
# ------------------------------------------------------------------------------
# This module runs the Extended Kalman Filter (EKF) for tracking.
# It operates as a separate process and communicates with other parts of the
# system via shared memory.
#
# Key Fixes:
# - EKF tuning parameters (especially Measurement Noise R) are now different
#   for Debug Mode to provide smoother tracking of close, erratic targets.
# - The main loop is a robust state machine handling all operational states.
# ==============================================================================

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
import traceback

from analysis.plotter import plot_ekf_vs_measured


# region Helper Functions
def spherical_to_cartesian(az_rad, el_rad, dist):
    """Converts spherical coordinates (azimuth, elevation, distance) to Cartesian (x, y, z)."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to spherical (azimuth, elevation, distance)."""
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if dist < 1e-6:  # Avoid division by zero for points at the origin
        return 0, 0, 0
    az = np.arctan2(y, x)
    el = np.arcsin(z / dist)
    return az, el, dist


def normalize_angle(angle):
    """Normalize angle to [-π, π] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def state_to_angles(state):
    """Convert state vector to azimuth and elevation in degrees."""
    x, y, z = state[0], state[1], state[2]
    az_rad, el_rad, _ = cartesian_to_spherical(x, y, z)
    return np.rad2deg(normalize_angle(az_rad)), np.rad2deg(el_rad)


def angle_difference(angle1, angle2):
    """Calculate the shortest angular distance between two angles in radians."""
    return normalize_angle(angle1 - angle2)


# endregion

class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    def __init__(self, std_acc, distance_constraint):
        dim_x = 6  # State: [x, y, z, vx, vy, vz]
        dim_z = 3  # Measurement: [azimuth, elevation, distance]
        super().__init__(dim_x=dim_x, dim_z=dim_z)

        self.std_acc = std_acc
        self.distance_constraint = distance_constraint

        self.P = np.eye(6) * 10.0
        self.P[3:, 3:] *= 2.0

        self.initialized = False
        self.last_time = None

    def update_matrices(self, dt):
        """Updates F (State Transition) and Q (Process Noise) matrices for a given time step dt."""
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # Process noise Q - models uncertainty in the drone's/hand's motion
        q = Q_discrete_white_noise(dim=3, dt=dt, var=self.std_acc ** 2)
        self.Q = block_diag(q, q)

    def h(self, x):
        """Measurement function: maps state space [x,y,z,...] to measurement space [az, el, dist]."""
        az, el, dist = cartesian_to_spherical(x[0], x[1], x[2])
        return np.array([az, el, dist])

    def HJacobian(self, x):
        """Jacobian of the measurement function H with improved numerical stability."""
        H = np.zeros((3, 6))
        x0, x1, x2 = x[0], x[1], x[2]

        eps = 1e-6
        x_sq_y_sq = x0 ** 2 + x1 ** 2
        dist_sq = x_sq_y_sq + x2 ** 2

        if x_sq_y_sq < eps or dist_sq < eps:
            return H

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
        """Predict step with optional distance constraint for drone mode."""
        super().predict()

        if self.distance_constraint:
            current_dist = np.linalg.norm(self.x[0:3])
            if current_dist > 12.0:
                self.x[0:3] *= (12.0 / current_dist)
            elif current_dist < 3.0:
                self.x[0:3] *= (3.0 / current_dist)

    def update_with_angle_wrapping(self, z, HJacobian, Hx, R):
        """Custom update step with proper angle wrapping for azimuth."""
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
    """Initializes the EKF state vector from the first two measurements."""
    meas1, meas2 = initial_points[0], initial_points[1]
    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])

    dt = meas2['time'] - meas1['time']
    if dt <= 0.01:
        dt = 0.1
        print(f"[EKF WARN] Invalid dt in init ({dt:.4f}s), defaulting to {dt}s.")

    vx, vy, vz = (x2 - x1) / dt, (y2 - y1) / dt, (z2 - z1) / dt
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])


def create_measurement_noise_matrix(strength, measurement_count, debug_mode):
    """
    Creates the R (measurement noise) matrix, adapting to the operational mode.
    """
    if debug_mode:
        # --- DEBUG MODE (Hand Tracking): High angular noise, low distance noise ---
        # Trust distance measurement highly, but be skeptical of the jittery angles.
        angular_var = (np.deg2rad(4.0)) ** 2  # Assume 4 deg standard deviation for angles
        dist_var = (0.05) ** 2  # Assume 5cm standard deviation for distance
    else:
        # --- DRONE MODE: Noise based on signal strength ---
        base_angular_var = (np.deg2rad(0.5)) ** 2
        base_dist_var = (0.05) ** 2

        strength_factor = max(1.0, strength / 1000.0)
        angular_var = base_angular_var / strength_factor
        dist_var = base_dist_var / strength_factor

    return np.diag([angular_var, angular_var, dist_var])


def get_next_prediction(ekf, dt):
    """Peeks at the prediction for the next time step without modifying the main filter state."""
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt)
    temp_ekf.predict()
    return state_to_angles(temp_ekf.x)


def calculate_confidence(P):
    """Calculates a tracking confidence score (0-1) based on the covariance matrix."""
    position_uncertainty = np.trace(P[0:3, 0:3])
    confidence = 1.0 / (1.0 + position_uncertainty)
    return min(max(confidence, 0.0), 1.0)


# endregion

def run_ekf_tracker(shared_data):
    """Main EKF tracking loop with a robust state machine."""
    print("[EKF] Starting tracker process...")
    ekf = None
    history_measurements, history_estimates = [], []

    ekf_start, ekf_running = shared_data['ekf_start'], shared_data['ekf_running']
    points_count, points_buffer = shared_data['points_count'], shared_data['points_buffer']
    shutdown, generate_plot = shared_data['shutdown'], shared_data['generate_plot_on_stop']

    while not shutdown.value:
        try:
            # STATE 1: UNINITIALIZED
            if ekf is None or not ekf.initialized:
                if ekf_start.value and points_count.value >= 2:
                    print("[EKF] Initialization signal received.")

                    if shared_data["debug_mode"].value:
                        print("[EKF] Configuring for DEBUG MODE (Hand Tracking).")
                        ekf = CloseRangeDroneTrackerEKF(std_acc=0.7, distance_constraint=False)
                    else:
                        print("[EKF] Configuring for DRONE MODE.")
                        ekf = CloseRangeDroneTrackerEKF(std_acc=0.04, distance_constraint=True)

                    history_measurements.clear()
                    history_estimates.clear()

                    p1 = {'z': np.array([np.deg2rad(points_buffer[0]), np.deg2rad(points_buffer[1]), points_buffer[2]]),
                          'time': points_buffer[4]}
                    p2 = {'z': np.array([np.deg2rad(points_buffer[5]), np.deg2rad(points_buffer[6]), points_buffer[7]]),
                          'time': points_buffer[9]}
                    init_ekf(ekf, [p1, p2])

                    ekf.last_time = p2['time']
                    ekf.initialized = True
                    shared_data['ekf_initialized'].value = True
                    ekf_running.value = True
                    ekf_start.value = False
                    points_count.value = 0
                    print(f"[EKF] Initialized successfully. Tracking started.")
                else:
                    time.sleep(0.1)
                    continue

            # STATE 2: INITIALIZED
            now = time.time()
            if ekf_running.value:
                dt = now - (ekf.last_time if ekf.last_time is not None else now)
                if dt <= 0: dt = 1e-3
                ekf.update_matrices(dt)
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
                    R = create_measurement_noise_matrix(strength, len(history_measurements),
                                                        shared_data["debug_mode"].value)
                    ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                    history_measurements.append({'z': z, 'time': ts})

                history_estimates.append(ekf.x.copy())
                ekf.last_time = now

                pred_az, pred_el = get_next_prediction(ekf, max(dt, 0.02))
                shared_data["predicted_azimuth"].value = pred_az
                shared_data["predicted_elevation"].value = pred_el
                shared_data["ekf_confidence"].value = calculate_confidence(ekf.P)
            else:
                if generate_plot.value:
                    print("[EKF] EKF stopped. Generating final plot...")
                    plot_ekf_vs_measured(history_measurements, history_estimates)
                    generate_plot.value = False

                print("[EKF] Resetting tracker state. Ready for new acquisition.")
                ekf.initialized = False
                shared_data['ekf_initialized'].value = False

            time.sleep(0.01)
        except Exception as e:
            print(f"[EKF] CRITICAL ERROR in tracking loop: {e}")
            traceback.print_exc()
            if ekf: ekf.initialized = False
            ekf_running.value = False
            time.sleep(1)

    print("[EKF] Shutting down...")