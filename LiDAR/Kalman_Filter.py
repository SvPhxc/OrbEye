# kalman_filter.py

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
from multiprocessing import Process, Array, Value

# Helper Functions (unchanged)
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

# Close-Range Drone Tracker EKF (unchanged)
class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
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
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        q_block = Q_discrete_white_noise(dim=2, dt=dt, var=self.std_acc**2)
        self.Q = block_diag(q_block, q_block, q_block)
    
    def h(self, x):
        az, el, dist = cartesian_to_spherical(x[0], x[1], x[2])
        return np.array([az, el, dist])
    
    def HJacobian(self, x):
        H = np.zeros((3, 6))
        x0, x1, x2 = x[0], x[1], x[2]
        
        eps = 1e-6
        x_sq_y_sq = x0**2 + x1**2 + eps
        dist_sq = x_sq_y_sq + x2**2 + eps
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
            current_dist = np.sqrt(self.x[0]**2 + self.x[1]**2 + self.x[2]**2)
            if current_dist < 6.0:
                scale = 6.0 / current_dist
                self.x[0:3] *= scale
            elif current_dist > 12.0:
                scale = 12.0 / current_dist
                self.x[0:3] *= scale
    
    def update_with_angle_wrapping(self, z, HJacobian, Hx, R):
        hx = Hx(self.x)
        
        y = z - hx
        y[0] = angle_difference(z[0], hx[0])
        
        if abs(y[1]) > np.deg2rad(2.0):
            y[1] = np.sign(y[1]) * np.deg2rad(2.0)
        
        H = HJacobian(self.x)
        PHT = self.P @ H.T
        S = H @ PHT + R
        
        S += np.eye(S.shape[0]) * 1e-8
        
        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHT @ np.linalg.pinv(S)
        
        K_modified = K.copy()
        K_modified[:, 1] *= 0.8
        
        self.x = self.x + K_modified @ y
        I_KH = np.eye(self.x.shape[0]) - K_modified @ H
        self.P = I_KH @ self.P @ I_KH.T + K_modified @ R @ K_modified.T

# --- MODIFIED EKF Process ---
def run_ekf_tracker(shared_data):
    """
    Main EKF tracking process that waits for satellite detections and then
    provides position estimates and predictions.
    """
    print("EKF Tracker: Starting drone tracking process...")
    
    ekf = CloseRangeDroneTrackerEKF(std_acc=0.04, distance_constraint=True)
    
    initialization_buffer = []
    INIT_BUFFER_SIZE = 2
    measurement_count = 0
    
    ekf_initialized = shared_data['ekf_initialized']
    estimated_azimuth = shared_data['estimated_azimuth']
    estimated_elevation = shared_data['estimated_elevation']
    predicted_azimuth = shared_data['predicted_azimuth']
    predicted_elevation = shared_data['predicted_elevation']
    ekf_confidence = shared_data['ekf_confidence']
    
    print("EKF Tracker: Waiting for satellite detection...")
    
    while not shared_data["shutdown"].value:
        try:
            # Wait for a satellite detection event
            if shared_data["satellite_detected"].value:
                # --- A satellite has been detected by the LiDAR handler ---
                measurement_count += 1
                current_time = time.time()
                
                with shared_data["satellite_points"].get_lock():
                    azimuth_deg = shared_data["satellite_points"][0]
                    elevation_deg = shared_data["satellite_points"][1]
                    strength = shared_data["satellite_points"][2]
                    distance_cm = shared_data["satellite_points"][3]

                # Reset the flag to avoid processing the same detection again
                with shared_data["satellite_detected"].get_lock():
                    shared_data["satellite_detected"].value = False

                print(f"EKF Tracker: Received detection #{measurement_count}: Az={azimuth_deg}, El={elevation_deg}, Dist={distance_cm}cm")

                # Convert measurements to standard units (radians, meters)
                az_rad = np.deg2rad(azimuth_deg)
                el_rad = np.deg2rad(elevation_deg)
                distance_m = distance_cm / 100.0
                
                z = np.array([az_rad, el_rad, distance_m])
                
                if not ekf.initialized:
                    measurement_data = {'z': z, 'time': current_time, 'strength': strength}
                    initialization_buffer.append(measurement_data)
                    
                    if len(initialization_buffer) >= INIT_BUFFER_SIZE:
                        init_ekf(ekf, initialization_buffer)
                        ekf.initialized = True
                        ekf.last_time = current_time
                        
                        with ekf_initialized.get_lock():
                            ekf_initialized.value = True
                        print(f"EKF Tracker: Initialized after {measurement_count} measurements.")
                else:
                    # Normal EKF operation
                    dt = current_time - ekf.last_time
                    if dt > 0:
                        ekf.update_matrices(dt)
                        R = create_measurement_noise_matrix(strength, measurement_count)
                        
                        ekf.predict()
                        ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                        
                        est_az, est_el = state_to_angles(ekf.x)
                        pred_az, pred_el = get_next_prediction(ekf, dt) # Predict for the next logical step
                        confidence = calculate_confidence(ekf.P)
                        
                        # Update shared variables for other processes to use
                        with estimated_azimuth.get_lock(): estimated_azimuth.value = est_az
                        with estimated_elevation.get_lock(): estimated_elevation.value = est_el
                        with predicted_azimuth.get_lock(): predicted_azimuth.value = pred_az
                        with predicted_elevation.get_lock(): predicted_elevation.value = pred_el
                        with ekf_confidence.get_lock(): ekf_confidence.value = confidence
                        
                        ekf.last_time = current_time
                        
                        if measurement_count % 5 == 0: # Status update every 5 detections
                            print(f"EKF Update: Est Az={est_az:.2f}°, El={est_el:.2f}° | Pred Az={pred_az:.2f}°, El={pred_el:.2f}° | Conf={confidence:.3f}")
            else:
                # If no detection, sleep briefly to prevent high CPU usage
                time.sleep(0.02)
        
        except Exception as e:
            print(f"EKF Tracker Error: {e}")
            time.sleep(0.1)
    
    print("EKF Tracker: Shutting down...")

# Helper functions (unchanged)
def init_ekf(ekf, initialization_buffer):
    meas1 = initialization_buffer[0]
    meas2 = initialization_buffer[1]
    
    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])
    
    dt = meas2['time'] - meas1['time']
    if dt <= 0: dt = 0.5
    
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt  
    vz = (z2 - z1) / dt
    
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])

def create_measurement_noise_matrix(strength, measurement_count):
    base_angular_var = (np.deg2rad(0.3))**2
    base_dist_var = 0.03**2
    
    angular_var = base_angular_var / (strength + 0.3)**2
    elevation_var = angular_var * 0.4
    dist_var = base_dist_var / (strength + 0.3)**2
    
    if measurement_count % 20 < 5:
        angular_var *= 1.8
        elevation_var *= 1.4
    
    return np.diag([angular_var, elevation_var, dist_var])

def get_next_prediction(ekf, dt):
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt)
    temp_ekf.predict()
    return state_to_angles(temp_ekf.x)

def calculate_confidence(P):
    position_uncertainty = np.trace(P[0:3, 0:3])
    confidence = 1.0 / (1.0 + position_uncertainty)
    return min(max(confidence, 0.0), 1.0)

def setup_ekf_shared_data(shared_data):
    """Add EKF-specific shared variables to the shared_data dictionary."""
    shared_data['ekf_initialized'] = Value('b', False)
    shared_data['estimated_azimuth'] = Value('d', 0.0)
    shared_data['estimated_elevation'] = Value('d', 0.0) 
    shared_data['predicted_azimuth'] = Value('d', 0.0)
    shared_data['predicted_elevation'] = Value('d', 0.0)
    shared_data['ekf_confidence'] = Value('d', 0.0)
    return shared_data
