# ekf_tracker.py

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time
import copy
import traceback

def spherical_to_cartesian(az_rad, el_rad, dist): x = dist*np.cos(el_rad)*np.cos(az_rad); y = dist*np.cos(el_rad)*np.sin(az_rad); z = dist*np.sin(el_rad); return x,y,z
def cartesian_to_spherical(x,y,z): dist=np.sqrt(x**2+y**2+z**2); az=np.arctan2(y,x); el=np.arctan2(z,np.sqrt(x**2+y**2)); return az,el,dist
def normalize_angle(angle): return np.arctan2(np.sin(angle), np.cos(angle))
def state_to_angles(state): x,y,z=state[0],state[1],state[2]; az,el,_=cartesian_to_spherical(x,y,z); return np.rad2deg(normalize_angle(az)),np.rad2deg(el)
def angle_difference(a1,a2): return normalize_angle(a1-a2)

class CloseRangeDroneTrackerEKF(ExtendedKalmanFilter):
    def __init__(self,std_acc,distance_constraint):
        super().__init__(dim_x=6,dim_z=3)
        self.std_acc,self.distance_constraint=std_acc,distance_constraint
        self.P=np.eye(6)*5; self.P[2,2]*=0.5; self.P[5,5]*=0.1
        self.initialized=False; self.last_time=None
    def update_matrices(self,dt):
        self.F=np.eye(6); self.F[0,3],self.F[1,4],self.F[2,5]=dt,dt,dt
        q=Q_discrete_white_noise(dim=3,dt=dt,var=self.std_acc**2); self.Q=block_diag(q,q)
    def h(self,x):
        az,el,dist=cartesian_to_spherical(x[0],x[1],x[2]); return np.array([az,el,dist])
    def HJacobian(self,x):
        H=np.zeros((3,6)); x0,x1,x2=x[0],x[1],x[2]; eps=1e-6
        x_sq_y_sq=x0**2+x1**2+eps; dist_sq=x_sq_y_sq+x2**2+eps; dist=np.sqrt(dist_sq)
        sqrt_x_sq_y_sq=np.sqrt(x_sq_y_sq)
        H[0,0]=-x1/x_sq_y_sq; H[0,1]=x0/x_sq_y_sq
        H[1,0]=-x0*x2/(sqrt_x_sq_y_sq*dist_sq); H[1,1]=-x1*x2/(sqrt_x_sq_y_sq*dist_sq)
        H[1,2]=sqrt_x_sq_y_sq/dist_sq
        H[2,0]=x0/dist; H[2,1]=x1/dist; H[2,2]=x2/dist
        return H
    def update_with_angle_wrapping(self,z,HJacobian,Hx,R):
        hx=Hx(self.x); y=z-hx; y[0]=angle_difference(z[0],hx[0])
        H=HJacobian(self.x); PHT=self.P@H.T
        S=H@PHT+R; K=PHT@np.linalg.inv(S)
        self.x=self.x+K@y; I_KH=np.eye(self.x.shape[0])-K@H
        self.P=I_KH@self.P@I_KH.T+K@R@K.T

def init_ekf(ekf, points_buffer):
    p1_az, p1_el, p1_dist, _, p1_ts = points_buffer[0:5]
    p2_az, p2_el, p2_dist, _, p2_ts = points_buffer[5:10]
    x1, y1, z1 = spherical_to_cartesian(np.deg2rad(p1_az), np.deg2rad(p1_el), p1_dist)
    x2, y2, z2 = spherical_to_cartesian(np.deg2rad(p2_az), np.deg2rad(p2_el), p2_dist)
    dt = p2_ts - p1_ts
    if dt <= 0.01: dt = 0.5
    vx, vy, vz = (x2-x1)/dt, (y2-y1)/dt, (z2-z1)/dt
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])
    print(f"[EKF] Initialized with state: pos=({x2:.1f},{y2:.1f},{z2:.1f}), vel=({vx:.1f},{vy:.1f},{vz:.1f})")

def create_measurement_noise_matrix(strength):
    strength_factor = max(1.0, strength / 1000.0)
    angular_var = (np.deg2rad(0.5) / strength_factor)**2
    dist_var = (0.05 / strength_factor)**2
    return np.diag([angular_var, angular_var, dist_var])

def get_next_prediction(ekf, dt):
    temp_ekf = copy.deepcopy(ekf)
    temp_ekf.update_matrices(dt); temp_ekf.predict()
    return state_to_angles(temp_ekf.x)

def calculate_confidence(P):
    pos_unc = np.trace(P[0:3, 0:3]); return 1.0 / (1.0 + pos_unc)

def run_ekf_tracker(shared_data):
    print("[EKF] EKF Tracker process started.")
    ekf = CloseRangeDroneTrackerEKF(std_acc=0.05, distance_constraint=False)

    while not shared_data["shutdown"].value:
        try:
            if not ekf.initialized:
                if shared_data["lidar_track_mode_active"].value: shared_data["lidar_track_mode_active"].value = False
                acquirer_status = shared_data["acquirer_status"].value
                if acquirer_status == 2 and shared_data["points_count"].value >= 2:
                    print("[EKF] Acquisition complete. Initializing filter...")
                    init_ekf(ekf, shared_data["points_buffer"][:])
                    ekf.initialized = True; ekf.last_time = time.time()
                    shared_data["ekf_initialized"].value = True
                    shared_data["lidar_track_mode_active"].value = True # Auto handoff
                    shared_data["acquirer_status"].value = 0
                    print("[EKF] Initialization successful. Engaging active tracking.")
                else:
                    if acquirer_status == 3: shared_data["acquirer_status"].value = 0
                    time.sleep(0.2)
                continue

            if not shared_data["lidar_track_mode_active"].value:
                print("[EKF] Tracking disabled. Resetting EKF state for re-acquisition."); ekf.initialized = False; shared_data["ekf_initialized"].value = False; time.sleep(0.2); continue

            now = time.time(); dt = now - ekf.last_time; ekf.last_time = now
            if dt <= 0: dt = 0.01

            ekf.update_matrices(dt); ekf.predict()
            pred_az, pred_el = get_next_prediction(ekf, 0.05)
            shared_data["predicted_azimuth"].value = pred_az; shared_data["predicted_elevation"].value = pred_el; shared_data["new_prediction_available"].value = True

            wait_start_time = time.time()
            while not shared_data["refined_measurement_updated"].value:
                time.sleep(0.002)
                if not shared_data["lidar_track_mode_active"].value or shared_data["shutdown"].value: break
                if time.time() - wait_start_time > 1.5: print("[EKF] Timeout waiting for hardware. Disabling."); shared_data["lidar_track_mode_active"].value = False; break
            if not shared_data["lidar_track_mode_active"].value: continue

            az_deg, el_deg, dist_m, strength, ts = shared_data["refined_measurement"][:]; shared_data["refined_measurement_updated"].value = False
            if dist_m > 500: print("[EKF] Lost track signal. Disabling."); shared_data["lidar_track_mode_active"].value = False; continue

            z = np.array([np.deg2rad(az_deg), np.deg2rad(el_deg), dist_m])
            R = create_measurement_noise_matrix(strength)
            ekf.update_with_angle_wrapping(z, ekf.HJacobian, ekf.h, R)

            est_az, est_el = state_to_angles(ekf.x)
            shared_data["estimated_azimuth"].value = est_az; shared_data["estimated_elevation"].value = est_el; shared_data["ekf_confidence"].value = calculate_confidence(ekf.P)

        except Exception as e:
            print(f"[EKF] CRITICAL ERROR: {e}"); traceback.print_exc(); time.sleep(1)

    print("[EKF] Shutting down...")