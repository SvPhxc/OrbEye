# acquirer.py

import time
import math
import numpy as np

# Tunable parameters
COARSE_HALF_SPAN_DEG = 15.0
COARSE_STEP_DEG = 3.0
REFINE_RADIUS_DEG = 2.0
REFINE_STEP_DEG = 0.5
DWELL_TIME_S = 0.05
POSE_REACHED_TIMEOUT_S = 2.0
MIN_POINT_SEPARATION_S = 0.2


def _normalize_az(az):
    return az % 360.0


def _clamp_tilt(el):
    return max(0.0, min(90.0, el))


def _read_lidar(shared):
    arr = shared["lidar_data"]
    with arr.get_lock():
        d, s, ts = float(arr[0]), float(arr[1]), float(arr[2])
    return d, s, ts


def _populate_points_buffer(shared, points):
    """Populates the shared buffer with 3 points, now including the timestamp."""
    buf = shared["points_buffer"]
    count = shared["points_count"]
    with count.get_lock(), buf.get_lock():
        for i, p in enumerate(points):
            # --- FIX: Save 5 values per point ---
            base = i * 5
            buf[base + 0] = float(p['az'])
            buf[base + 1] = float(p['el'])
            buf[base + 2] = float(p['distance_m'])
            buf[base + 3] = float(p['strength'])
            buf[base + 4] = float(p['timestamp'])  # <-- The missing value
        count.value = len(points)


def _wait_for_fresh_lidar(shared, old_ts, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        d, s, ts = _read_lidar(shared)
        if ts > old_ts:
            az = float(shared["stepper_degrees"].value)
            el = float(shared["servo_degrees"].value)
            return {'az': az, 'el': el, 'distance_m': d / 100.0, 'strength': s, 'timestamp': ts}
        time.sleep(0.005)
    return None


def _move_and_wait(shared, target_az, target_el):
    if shared["shutdown"].value: return False

    shared["target_azimuth"].value = target_az
    shared["target_elevation"].value = target_el
    shared["target_reached"].value = False
    shared["go_to_target"].value = True

    deadline = time.monotonic() + POSE_REACHED_TIMEOUT_S
    while time.monotonic() < deadline and not shared["shutdown"].value:
        if shared["target_reached"].value:
            time.sleep(DWELL_TIME_S)
            return True
        time.sleep(0.01)

    shared["go_to_target"].value = False
    print(f"[Acquirer] WARN: Timeout waiting for pose ({target_az:.1f}, {target_el:.1f})")
    return False


def _generate_spiral_grid(center_az, center_el, half_span_deg, step_deg):
    yield center_az, _clamp_tilt(center_el)
    max_offset_steps = int(math.ceil(half_span_deg / step_deg))
    for layer in range(1, max_offset_steps + 1):
        for dx in range(-layer, layer + 1): yield _normalize_az(center_az + dx * step_deg), _clamp_tilt(
            center_el + layer * step_deg)
        for dy in range(layer - 1, -layer - 1, -1): yield _normalize_az(center_az + layer * step_deg), _clamp_tilt(
            center_el + dy * step_deg)
        for dx in range(layer - 1, -layer - 1, -1): yield _normalize_az(center_az + dx * step_deg), _clamp_tilt(
            center_el - layer * step_deg)
        for dy in range(-layer + 1, layer): yield _normalize_az(center_az - layer * step_deg), _clamp_tilt(
            center_el + dy * step_deg)


def _refine_target_local(shared, rough_az, rough_el):
    best_sample = None
    grid = _generate_spiral_grid(rough_az, rough_el, REFINE_RADIUS_DEG, REFINE_STEP_DEG)

    for az, el in grid:
        if shared["shutdown"].value: return None
        if not _move_and_wait(shared, az, el): continue

        _, _, old_ts = _read_lidar(shared)
        sample = _wait_for_fresh_lidar(shared, old_ts, 0.2)
        if sample is None: continue

        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 5000

        if (min_m <= sample['distance_m'] <= max_m) and (sample['strength'] >= min_strength):
            if best_sample is None or sample['strength'] > best_sample['strength']:
                best_sample = sample

    if best_sample: _move_and_wait(shared, best_sample['az'], best_sample['el'])
    return best_sample


def acquire_three_points(shared):
    print("[Acquirer] Starting 3-point acquisition...")
    shared["acquirer_status"].value = 1

    seed_az = shared["stepper_degrees"].value
    seed_el = shared["servo_degrees"].value
    points = []

    print("[Acquirer] Searching for first point...")
    coarse_grid = _generate_spiral_grid(seed_az, seed_el, COARSE_HALF_SPAN_DEG, COARSE_STEP_DEG)
    initial_detection = None
    for az, el in coarse_grid:
        if shared["shutdown"].value: break
        if not _move_and_wait(shared, az, el): continue

        dist, strength, _ = _read_lidar(shared)
        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 5000

        if (min_m <= dist / 100.0 <= max_m) and (strength >= min_strength):
            initial_detection = {'az': az, 'el': el};
            break

    if not initial_detection:
        print("[Acquirer] Coarse search failed. Aborting.");
        shared["acquirer_status"].value = 3;
        return

    print("[Acquirer] Refining point 1...");
    p1 = _refine_target_local(shared, initial_detection['az'], initial_detection['el'])
    if not p1: print("[Acquirer] Failed to refine point 1. Aborting."); shared["acquirer_status"].value = 3; return
    points.append(p1);
    print(f"[Acquirer] Point 1 locked: Str {p1['strength']:.0f}")

    for i in range(2, 4):
        time.sleep(MIN_POINT_SEPARATION_S)
        print(f"[Acquirer] Refining point {i}...");
        p_next = _refine_target_local(shared, points[-1]['az'], points[-1]['el'])
        if not p_next: print(f"[Acquirer] Failed to refine point {i}. Aborting."); shared[
            "acquirer_status"].value = 3; return
        points.append(p_next);
        print(f"[Acquirer] Point {i} locked: Str {p_next['strength']:.0f}")

    _populate_points_buffer(shared, points)
    print("[Acquirer] Successfully acquired 3 points. Handing over to EKF.")
    shared["acquirer_status"].value = 2;
    shared["ekf_start"].value = True


def run_acquirer(shared_data):
    print("[Acquirer] Process started.")
    while not shared_data["shutdown"].value:
        if shared_data["acquire_points"].value:
            shared_data["acquire_points"].value = False
            acquire_three_points(shared_data)
        time.sleep(0.1)
    print("[Acquirer] Process shutting down.")


``` ** *
### 3. `LiDAR/Kalman_Filter.py` (Unchanged from your version)
This
file is now
correct
because
the
other
files
have
been
fixed
to
match
its
expectations.No
changes
are
needed
here.

```python
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
                has_new = (not history_measurements