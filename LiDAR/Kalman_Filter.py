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
    Main EKF tracking process that reads shared sensor data and provides 
    position estimates and predictions.
    """
    print("EKF Tracker: Starting drone tracking process...")
    
    # Initialize EKF
    ekf = CloseRangeDroneTrackerEKF(std_acc=0.04, distance_constraint=True)
    
    # Tracking state
    initialization_buffer = []
    INIT_BUFFER_SIZE = 2
    last_measurement_time = None
    measurement_count = 0
    
    # Add new shared variables for EKF outputs
    ekf_initialized = shared_data.get('ekf_initialized', Value('b', False))
    estimated_azimuth = shared_data.get('estimated_azimuth', Value('d', 0.0))
    estimated_elevation = shared_data.get('estimated_elevation', Value('d', 0.0))
    predicted_azimuth = shared_data.get('predicted_azimuth', Value('d', 0.0))
    predicted_elevation = shared_data.get('predicted_elevation', Value('d', 0.0))
    ekf_confidence = shared_data.get('ekf_confidence', Value('d', 0.0))
    
    print("EKF Tracker: Waiting for sensor data...")
    
    while not shared_data["shutdown"].value:
        try:
            # Read current sensor data
            current_time = time.time()
            
            # Get measurements from shared data
            with shared_data["lidar_data"].get_lock():
                lidar_distance = shared_data["lidar_data"][0]
                lidar_strength = shared_data["lidar_data"][1] 
                lidar_timestamp = shared_data["lidar_data"][2]
            
            with shared_data["stepper_degrees"].get_lock():
                current_azimuth = shared_data["stepper_degrees"].value
                
            with shared_data["servo_degrees"].get_lock():
                current_elevation = shared_data["servo_degrees"].value
            
            # Check if we have valid new measurements
            if (lidar_distance > 0 and 6.0 <= lidar_distance <= 12.0 and 
                lidar_strength > 0.1 and
                (last_measurement_time is None or lidar_timestamp > last_measurement_time)):
                
                last_measurement_time = lidar_timestamp
                measurement_count += 1
                
                # Convert measurements to proper format
                az_rad = np.deg2rad(current_azimuth)
                el_rad = np.deg2rad(current_elevation)
                
                # Create measurement vector
                z = np.array([az_rad, el_rad, lidar_distance])
                
                if not ekf.initialized:
                    # Collect initialization data
                    measurement_data = {
                        'z': z,
                        'time': current_time,
                        'strength': lidar_strength
                    }
                    initialization_buffer.append(measurement_data)
                    
                    if len(initialization_buffer) >= INIT_BUFFER_SIZE:
                        # Initialize EKF with first two measurements
                        init_ekf(ekf, initialization_buffer)
                        ekf.initialized = True
                        ekf.last_time = current_time
                        
                        with ekf_initialized.get_lock():
                            ekf_initialized.value = True
                            
                        print(f"EKF Tracker: Initialized after {measurement_count} measurements")
                else:
                    # Normal EKF operation
                    dt = current_time - ekf.last_time
                    if dt > 0:
                        # Update EKF matrices
                        ekf.update_matrices(dt)
                        
                        # Create dynamic measurement noise based on signal strength
                        R = create_measurement_noise_matrix(lidar_strength, measurement_count)
                        
                        # Predict and update
                        ekf.predict()
                        ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)
                        
                        # Get current estimates
                        est_az, est_el = state_to_angles(ekf.x)
                        
                        # Get prediction for next time step
                        pred_az, pred_el = get_next_prediction(ekf, dt)
                        
                        # Calculate confidence based on covariance trace
                        confidence = calculate_confidence(ekf.P)
                        
                        # Update shared variables
                        with estimated_azimuth.get_lock():
                            estimated_azimuth.value = est_az
                        with estimated_elevation.get_lock():
                            estimated_elevation.value = est_el
                        with predicted_azimuth.get_lock():
                            predicted_azimuth.value = pred_az
                        with predicted_elevation.get_lock():
                            predicted_elevation.value = pred_el
                        with ekf_confidence.get_lock():
                            ekf_confidence.value = confidence
                        
                        ekf.last_time = current_time
                        
                        # Optional: Print periodic status
                        if measurement_count % 20 == 0:
                            print(f"EKF: Az={est_az:.2f}°, El={est_el:.2f}°, "
                                f"Pred: Az={pred_az:.2f}°, El={pred_el:.2f}°, "
                                f"Conf={confidence:.3f}")
            
        except Exception as e:
            print(f"EKF Tracker Error: {e}")
            time.sleep(0.01)
        
        time.sleep(0.02)  # 50Hz update rate
    
    print("EKF Tracker: Shutting down...")

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

# Updated main function for integration
def setup_ekf_shared_data(shared_data):
    """Add EKF-specific shared variables to the shared_data dictionary."""
    shared_data['ekf_initialized'] = Value('b', False)
    shared_data['estimated_azimuth'] = Value('d', 0.0)
    shared_data['estimated_elevation'] = Value('d', 0.0) 
    shared_data['predicted_azimuth'] = Value('d', 0.0)
    shared_data['predicted_elevation'] = Value('d', 0.0)
    shared_data['ekf_confidence'] = Value('d', 0.0)
    return shared_data

