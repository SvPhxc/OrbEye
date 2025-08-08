#change location of the file later 

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
from multiprocessing import Process, Array, Value

# Helper Functions
def spherical_to_cartesian(az_rad, el_rad, dist):
    """Converts spherical coordinates (azimuth, elevation, distance) to Cartesian (x, y, z)."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to spherical (azimuth, elevation, distance)."""
    dist = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
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

# Close-Range Drone Tracker EKF
class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    def __init__(self, std_acc=0.04, distance_constraint=True):
        dim_x = 6  # State: [x, y, z, vx, vy, vz]
        dim_z = 3  # Measurement: [azimuth, elevation, distance]
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.std_acc = std_acc
        self.distance_constraint = distance_constraint
        
        # Initialize with smaller uncertainty for close-range tracking
        self.P = np.eye(6) * 5
        # Position uncertainty - be more confident in z (elevation-related)
        self.P[0:2, 0:2] *= 1.0  # x, y position uncertainty
        self.P[2, 2] *= 0.5      # z position uncertainty (more confident)
        # Velocity uncertainty  
        self.P[3:5, 3:5] *= 0.3  # x, y velocity uncertainty
        self.P[5, 5] *= 0.1      # z velocity uncertainty (very confident)
        
        # Track initialization status
        self.initialized = False
        self.last_time = None
        
    def update_matrices(self, dt):
        """Updates F and Q matrices for a given time step dt."""
        # State Transition Matrix F (constant velocity model)
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        # Much smaller process noise for close-range, slower drone movements
        q_block = Q_discrete_white_noise(dim=2, dt=dt, var=self.std_acc**2)
        self.Q = block_diag(q_block, q_block, q_block)
    
    def h(self, x):
        """Measurement function: maps state space to [azimuth, elevation, distance]."""
        az, el, dist = cartesian_to_spherical(x[0], x[1], x[2])
        return np.array([az, el, dist])
    
    def HJacobian(self, x):
        """Jacobian of the measurement function H with improved numerical stability."""
        H = np.zeros((3, 6))
        x0, x1, x2 = x[0], x[1], x[2]
        
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        x_sq_y_sq = x0**2 + x1**2 + eps
        dist_sq = x_sq_y_sq + x2**2 + eps
        dist = np.sqrt(dist_sq)
        sqrt_x_sq_y_sq = np.sqrt(x_sq_y_sq)
        
        # Partials for azimuth
        H[0, 0] = -x1 / x_sq_y_sq
        H[0, 1] = x0 / x_sq_y_sq
        
        # Partials for elevation (most sensitive for close range)
        H[1, 0] = -x0 * x2 / (sqrt_x_sq_y_sq * dist_sq)
        H[1, 1] = -x1 * x2 / (sqrt_x_sq_y_sq * dist_sq)
        H[1, 2] = sqrt_x_sq_y_sq / dist_sq
        
        # Partials for distance
        H[2, 0] = x0 / dist
        H[2, 1] = x1 / dist
        H[2, 2] = x2 / dist
        
        return H
    
    def predict(self):
        """Predict step with distance constraint for close-range tracking."""
        super().predict()
        
        # Apply distance constraint (keep drone in 6-12m range)
        if self.distance_constraint:
            current_dist = np.sqrt(self.x[0]**2 + self.x[1]**2 + self.x[2]**2)
            if current_dist < 6.0:
                # Scale position to minimum distance
                scale = 6.0 / current_dist
                self.x[0:3] *= scale
            elif current_dist > 12.0:
                # Scale position to maximum distance  
                scale = 12.0 / current_dist
                self.x[0:3] *= scale
    
    def update_with_angle_wrapping(self, z, HJacobian, Hx, R):
        """Update step with proper angle wrapping and enhanced elevation tracking."""
        # Predict measurement
        hx = Hx(self.x)
        
        # Calculate innovation with angle wrapping for azimuth
        y = z - hx
        y[0] = angle_difference(z[0], hx[0])  # Wrap azimuth difference
        
        # For elevation, apply small smoothing to reduce noise sensitivity
        # Only apply small corrections to prevent over-correction
        if abs(y[1]) > np.deg2rad(2.0):  # If elevation error > 2 degrees
            y[1] = np.sign(y[1]) * np.deg2rad(2.0)  # Limit maximum correction
        
        # Rest of the update step
        H = HJacobian(self.x)
        PHT = self.P @ H.T
        S = H @ PHT + R
        
        # Add small regularization for numerical stability
        S += np.eye(S.shape[0]) * 1e-8
        
        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            K = PHT @ np.linalg.pinv(S)
        
        # Apply selective update - be more conservative with elevation updates
        K_modified = K.copy()
        K_modified[:, 1] *= 0.8  # Reduce elevation correction gain by 20%
        
        # Update state and covariance
        self.x = self.x + K_modified @ y
        I_KH = np.eye(self.x.shape[0]) - K_modified @ H
        self.P = I_KH @ self.P @ I_KH.T + K_modified @ R @ K_modified.T

def run_ekf_tracker(shared_data):
    """
    EKF tracking loop:
      - Waits for ekf_start + at least 2 acquired points
      - Initializes EKF from the first two points (position + velocity)
      - Then runs predict() every cycle and update() whenever a fresh, valid LiDAR sample arrives
      - Publishes estimated_* and predicted_* angles + ekf_confidence to shared_data
    """
    print("[EKF] Starting tracker...")
    ekf = CloseRangeDroneTrackerEKF(std_acc=0.04, distance_constraint=True)

    # State for init & measurements
    waiting_for_init = True
    last_measurement_time = None
    measurement_count = 0

    # Handy refs to shared Values (created in main via setup_ekf_shared_data)
    ekf_initialized     = shared_data['ekf_initialized']
    estimated_azimuth   = shared_data['estimated_azimuth']
    estimated_elevation = shared_data['estimated_elevation']
    predicted_azimuth   = shared_data['predicted_azimuth']
    predicted_elevation = shared_data['predicted_elevation']
    ekf_confidence      = shared_data['ekf_confidence']

    # Optional control flags you added
    ekf_running   = shared_data['ekf_running']      # bool: track predictions?
    ekf_start     = shared_data['ekf_start']        # bool: set True after spiral acquisition
    points_count  = shared_data['points_count']     # int: how many valid points captured (0..3)
    points_buffer = shared_data['points_buffer']    # 12 doubles: [az,el,dist_m,str]*3

    while not shared_data["shutdown"].value:
        try:
            now = time.time()

            # ---------------------------
            # 1) Handle EKF initialization
            # ---------------------------
            if waiting_for_init and ekf_start.value:
                k = points_count.value
                if k >= 2:
                    pb = points_buffer  # layout: [az0, el0, dist0_m, str0, az1, el1, dist1_m, str1, az2, el2, dist2_m, str2]

                    # First two points → init buffer
                    z1 = np.array([np.deg2rad(pb[0]), np.deg2rad(pb[1]), pb[2]])
                    z2 = np.array([np.deg2rad(pb[4]), np.deg2rad(pb[5]), pb[6]])
                    initialization_buffer = [
                        {'z': z1, 'time': now,         'strength': pb[3]},
                        {'z': z2, 'time': now + 0.10,  'strength': pb[7]},
                    ]

                    init_ekf(ekf, initialization_buffer)
                    ekf.initialized = True
                    ekf.last_time = now
                    with ekf_initialized.get_lock():
                        ekf_initialized.value = True
                    with ekf_running.get_lock():
                        ekf_running.value = True  # start following predictions (your motor loop decides how to act)

                    waiting_for_init = False
                    print("[EKF] Initialized from acquired points.")
                    # Optional: immediately use 3rd point as first update if present
                    if k >= 3:
                        z3 = np.array([np.deg2rad(pb[8]), np.deg2rad(pb[9]), pb[10]])
                        R3 = create_measurement_noise_matrix(pb[11], measurement_count)
                        ekf.update_with_angle_wrapping(z3, ekf.HJacobian, ekf.h, R3)
                        measurement_count += 1
                        last_measurement_time = now + 0.20
                    continue  # next loop

            # If not initialized yet, just idle
            if not ekf.initialized:
                time.sleep(0.02)
                continue

            # ---------------------------
            # 2) Normal EKF tracking loop
            # ---------------------------
            # Time update
            dt = now - (ekf.last_time if ekf.last_time is not None else now)
            if dt <= 0.0:
                dt = 1e-3

            ekf.update_matrices(dt)
            ekf.predict()

            # Try to read a fresh LiDAR sample
            with shared_data["lidar_data"].get_lock():
                lidar_distance_cm = shared_data["lidar_data"][0]
                lidar_strength    = shared_data["lidar_data"][1]
                lidar_ts          = shared_data["lidar_data"][2]

            # Current sensor pointing (for az/el)
            with shared_data["stepper_degrees"].get_lock():
                current_az_deg = shared_data["stepper_degrees"].value
            with shared_data["servo_degrees"].get_lock():
                current_el_deg = shared_data["servo_degrees"].value

            # Only update if we have a *new* and *plausible* measurement
            has_new = (last_measurement_time is None) or (lidar_ts > last_measurement_time)
            valid_range = (shared_data["lidar_acceptance_range"][0] <= lidar_distance_cm*100 <= shared_data["lidar_acceptance_range"][1] )  
            valid_strength = (lidar_strength >= 5000)             # conservative TFmini gate

            if has_new and valid_range and valid_strength:
                z = np.array([
                    np.deg2rad(current_az_deg % 360.0),
                    np.deg2rad(max(0.0, min(90.0, current_el_deg))),
                    lidar_distance_cm / 100.0  # m
                ])
                R = create_measurement_noise_matrix(lidar_strength, measurement_count)
                ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)

                last_measurement_time = lidar_ts
                measurement_count += 1

            # Publish outputs (estimates + one-step prediction)
            est_az_deg, est_el_deg = state_to_angles(ekf.x)
            pred_az_deg, pred_el_deg = get_next_prediction(ekf, max(dt, 0.02))
            conf = calculate_confidence(ekf.P)

            with estimated_azimuth.get_lock():
                estimated_azimuth.value = float(est_az_deg)
            with estimated_elevation.get_lock():
                estimated_elevation.value = float(est_el_deg)
            with predicted_azimuth.get_lock():
                predicted_azimuth.value = float(pred_az_deg)
            with predicted_elevation.get_lock():
                predicted_elevation.value = float(pred_el_deg)
            with ekf_confidence.get_lock():
                ekf_confidence.value = float(conf)

            ekf.last_time = now
            time.sleep(0.01)  # ~50 Hz loop

        except Exception as e:
            print(f"[EKF] Error: {e}")
            time.sleep(0.05)

    print("[EKF] Shutting down...")


def init_ekf(ekf, initialization_buffer):
    """Initialize EKF state using first two measurements."""
    meas1 = initialization_buffer[0]
    meas2 = initialization_buffer[1]
    
    # Convert to Cartesian coordinates
    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])
    
    # Calculate time difference and initial velocity
    dt = meas2['time'] - meas1['time']
    if dt <= 0:
        dt = 0.5  # Default time step
    
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt  
    vz = (z2 - z1) / dt
    
    # Set initial state (use second measurement as starting position)
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])

def create_measurement_noise_matrix(strength, measurement_count):
    """Create measurement noise matrix based on signal strength and search phase."""
    # Base noise levels for close-range tracking with spiral search
    base_angular_var = (np.deg2rad(0.3))**2
    base_dist_var = 0.03**2
    
    # Adjust based on signal strength
    angular_var = base_angular_var / (strength + 0.3)**2
    elevation_var = angular_var * 0.4  # Better elevation precision
    dist_var = base_dist_var / (strength + 0.3)**2
    
    # Add search phase uncertainty (simulate aggressive search periods)
    if measurement_count % 20 < 5:  # 25% of time in "aggressive search mode"
        angular_var *= 1.8
        elevation_var *= 1.4
    
    return np.diag([angular_var, elevation_var, dist_var])

def get_next_prediction(ekf, dt):
    """Get prediction for next time step without modifying main filter."""
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt)
    temp_ekf.predict()
    return state_to_angles(temp_ekf.x)

def calculate_confidence(P):
    """Calculate tracking confidence based on covariance matrix."""
    # Use trace of position covariance as confidence metric
    position_uncertainty = np.trace(P[0:3, 0:3])
    # Convert to 0-1 scale (lower uncertainty = higher confidence)
    confidence = 1.0 / (1.0 + position_uncertainty)
    return min(max(confidence, 0.0), 1.0)


