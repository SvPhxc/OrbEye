import time
import numpy as np
import os

# --- Tracking Constants ---
GRID_STEP_DEGREES = 1.0
NUM_ACQUIRE_POINTS = 2
# --- NEW: Background Rejection Threshold ---
# A measurement must be at least this much closer than the background to be considered valid.
BACKGROUND_THRESHOLD_CM = 50.0


# ==============================================================================
# HELPER FUNCTIONS (Unchanged)
# ==============================================================================
def spherical_to_cartesian(az, el, dist):
    """Converts Spherical coordinates (az, el in degrees) to Cartesian (x, y, z)."""
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to Spherical (az, el in degrees)."""
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if dist == 0: return 0, 0, 0
    az_rad = np.arctan2(y, x)
    el_rad = np.arcsin(z / dist)
    return np.degrees(az_rad), np.degrees(el_rad), dist


# ==============================================================================
# KALMAN FILTER CLASS (Unchanged)
# ==============================================================================
class KalmanFilter:
    # ... (Kalman Filter class remains exactly the same as before) ...
    def __init__(self, dt, std_acc, std_meas):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.F = np.array(
            [[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        self.Q = np.eye(6) * std_acc ** 2
        self.R = np.eye(3) * std_meas ** 2
        self.P = np.eye(6) * 500

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


# ==============================================================================
# MAIN TRACKER LOGIC PROCESS (MODIFIED)
# ==============================================================================
class TrackerLogic:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.ekf = KalmanFilter(dt=0.1, std_acc=0.5, std_meas=0.1)
        self.state = "IDLE"
        self.background_map = None  # <-- NEW: To hold the lookup map
        self._load_background_data()  # <-- NEW: Load the data on startup

    def _load_background_data(self):  # <-- NEW METHOD
        """Loads and processes the background scan into a fast 2D lookup map."""
        bg_path = self.shared_data["background_path"].value
        try:
            print(f"[TrackerLogic] Attempting to load background data from '{bg_path}'...")
            raw_data = np.load(bg_path)
            # Create a map with a large default distance. Dims are [elevation][azimuth].
            # Using 1-degree resolution for the map.
            self.background_map = np.full((91, 361), 6000.0)  # Default to 60m

            for point in raw_data:
                az, el, dist, _ = point
                if dist > 0:  # Only use valid distance readings
                    # Use integer indices for fast lookup
                    az_idx = int(round(az)) % 360
                    el_idx = int(round(el))
                    # Store the minimum distance found for that angle
                    if dist < self.background_map[el_idx, az_idx]:
                        self.background_map[el_idx, az_idx] = dist

            print("[TrackerLogic] Background data loaded and processed successfully.")
        except FileNotFoundError:
            print(
                f"[TrackerLogic] WARNING: Background file '{bg_path}' not found. Tracker will run without background rejection.")
        except Exception as e:
            print(f"[TrackerLogic] ERROR loading background data: {e}")

    def run(self):
        print("[TrackerLogic] Process is running.")
        while not self.shared_data["shutdown"].value:
            # (State switching logic remains the same as before)
            # ...

            # --- STATE MACHINE ---
            if self.state == "TRACKING_LOOP":
                self.shared_data["ekf_running"].value = True

                # 1. PREDICT (Unchanged)
                predicted_state = self.ekf.predict()
                pred_pos = predicted_state[:3, 0]
                pred_az, pred_el, _ = cartesian_to_spherical(pred_pos[0], pred_pos[1], pred_pos[2])
                self.shared_data["predicted_azimuth"].value = pred_az
                self.shared_data["predicted_elevation"].value = pred_el

                # 2. REQUEST 3x3 GRID SCAN (Unchanged)
                # ...

                # 3. WAIT for scan to complete (Unchanged)
                # ...

                # 4. PROCESS results and UPDATE EKF <-- HEAVILY MODIFIED
                grid_results = np.array(self.shared_data["grid_scan_results"][:])

                valid_points = []
                for i in range(9):
                    measured_dist = grid_results[i]
                    # Basic validity check: must be a positive reading
                    if measured_dist <= 0:
                        continue

                    row, col = divmod(i, 3)
                    pan_offset = (col - 1) * GRID_STEP_DEGREES
                    tilt_offset = (row - 1) * GRID_STEP_DEGREES

                    point_az = (pred_az + pan_offset) % 360
                    point_el = max(0, min(90, pred_el + tilt_offset))

                    # --- BACKGROUND REJECTION LOGIC ---
                    is_valid_target = True  # Assume valid unless proven otherwise
                    if self.background_map is not None:
                        az_idx = int(round(point_az)) % 360
                        el_idx = int(round(point_el))

                        bg_dist = self.background_map[el_idx, az_idx]

                        # If the measured point is NOT significantly closer than the background, it's rejected.
                        if measured_dist >= (bg_dist - BACKGROUND_THRESHOLD_CM):
                            is_valid_target = False

                    if is_valid_target:
                        # Store the distance and the index of the valid point
                        valid_points.append((measured_dist, i))

                # --- UPDATE DECISION ---
                if not valid_points:
                    print("[TrackerLogic] All grid points rejected as background or invalid. Coasting.")
                    continue  # Skip the update step and proceed to the next prediction

                # If we have valid points, find the closest one among them
                best_dist, best_idx = min(valid_points)

                # Now proceed with the update using the validated best point
                row, col = divmod(best_idx, 3)
                pan_offset = (col - 1) * GRID_STEP_DEGREES
                tilt_offset = (row - 1) * GRID_STEP_DEGREES
                measured_az = pred_az + pan_offset
                measured_el = pred_el + tilt_offset

                print(f"[TrackerLogic] Best VALID measurement: {best_dist:.1f}cm. Updating EKF.")
                measurement_pos = spherical_to_cartesian(measured_az, measured_el, best_dist / 100.0)

                self.ekf.update(measurement_pos.reshape(3, 1))

            # ... (Other states like IDLE, AWAITING_ACQUISITION remain the same) ...

            time.sleep(0.05)
        print("[TrackerLogic] Process shut down.")


def run_tracker_logic(shared_data):
    tracker = TrackerLogic(shared_data)
    tracker.run()