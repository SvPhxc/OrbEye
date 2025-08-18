# tracking_logic.py

import time
import threading
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import math
from enum import Enum


class TrackingState(Enum):
    IDLE = 0
    ACQUIRING = 1
    TRACKING = 2
    DEBUG_MODE = 3
    REACTIVE_MODE = 4


class ClutterFilter:
    """
    A robust filter that rejects static background objects by first finding clutter
    in the same direction (az/el) and then checking if the new point is
    significantly closer than the known background object.
    """

    def __init__(self, background_file="background_data.npy", angular_tolerance=0.5, distance_margin_cm=70.0):
        """
        Initializes the filter with a 2D (azimuth, elevation) background map.

        Args:
            background_file (str): Path to the background scan data.
            angular_tolerance (float): The maximum angle (in degrees) to consider a point
                                       as being in the "same direction" as a background point.
            distance_margin_cm (float): A new point must be at least this much closer
                                        than a background object in the same direction to be
                                        considered a valid target.
        """
        self.angular_tolerance = angular_tolerance
        self.distance_margin_cm = distance_margin_cm
        self.background_tree = None
        self.background_data = None

        try:
            # Load background data [azimuth, elevation, distance_cm, strength]
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")

            # --- KEY CHANGE ---
            # Build k-d tree on directional coordinates ONLY [az, el] for fast angular search.
            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords)
            print("[ClutterFilter] 2D (directional) k-d tree built successfully.")

        except FileNotFoundError:
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found. Running without clutter filtering.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR loading background data: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """
        Checks if a measurement is a valid target or background clutter.

        Returns:
            bool: True if the target is valid, False if it's likely clutter.
        """
        if self.background_tree is None:
            return True  # No background data, so all targets are considered valid.

        query_point = np.array([azimuth, elevation])
        try:
            # Query the 2D tree to find the angularly closest background point.
            angular_dist, idx = self.background_tree.query(query_point, k=1)

            # 1. Is there a background object in this direction?
            if angular_dist < self.angular_tolerance:
                # 2. Get the distance of that background object.
                bg_distance = self.background_data[idx, 2]

                # 3. Is the new measurement significantly IN FRONT of the background?
                if distance < (bg_distance - self.distance_margin_cm):
                    return True  # Yes, it's a valid target in front of the background.
                else:
                    # No, it's at or behind the background object. It's clutter.
                    return False

            # If no background points are nearby in this direction, it's a valid target.
            return True

        except Exception:
            return True # Fail-safe: if the query fails, accept the measurement.






class KalmanFilter:
    def __init__(self, dt, process_variance, measurement_variance):
        """
        Initializes the Kalman Filter.
        :param dt: Time step
        :param process_variance: How much we trust the process model
        :param measurement_variance: How much we trust the measurement
        """
        self.dt = dt
        # State transition matrix
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Process covariance matrix
        self.Q = np.eye(4) * process_variance
        # Measurement covariance matrix
        self.R = np.eye(2) * measurement_variance
        # State estimate
        self.x = np.zeros((4, 1))
        # Error covariance matrix
        self.P = np.eye(4)

    def predict(self):
        """
        Predict the next state.
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        """
        Update the state with a new measurement.
        :param z: New measurement (az, el)
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P





import time
import math
from enum import Enum


# --- Helper objects for the class to be self-contained ---

class HandTrackerState(Enum):
    """Defines the possible states for the HandTracker."""
    IDLE = 0
    SCANNING = 1
    COASTING = 2
    KALMAN_TRACKING = 3



class HandTracker:
    """
    High-performance predictive tracker with velocity smoothing, dynamic rate adjustment,
    and a predictive "coasting" search mode for target reacquisition.
    Scan parameters (radius, points, speed) are dynamically adjusted based on target distance.
    """

    # Constants based on the TF Mini-S hardware specifications.
    # The Field of View (FOV) of the TF Mini-S is approximately 2 degrees.
    LIDAR_FOV = 2.0

    def __init__(self, timeout=1.0, coast_timeout=1.5,
                 prediction_factor=1, velocity_smoothing_factor=0.4):
        """
        Initializes the HandTracker.
        Note: scan_radius, scan_points, and time_per_waypoint are now dynamically
        calculated and will be set upon first target acquisition.
        """
        # Dynamic parameters are initialized but will be immediately overwritten
        # by _update_scan_parameters upon target acquisition.
        self.scan_radius = self.LIDAR_FOV / 2.0
        self.scan_points = 8
        self.time_per_waypoint = 0.03

        # Configuration for tracking behavior
        self.timeout = timeout
        self.coast_timeout = coast_timeout
        self.prediction_factor = prediction_factor
        self.velocity_smoothing_factor = velocity_smoothing_factor

        # Internal state variables
        self.state = HandTrackerState.IDLE
        self.best_point = {'az': 0, 'el': 0, 'strength': 0, 'dist': 0, 'time': 0}
        self.previous_best_point = None
        self.scan_path = []
        self.scan_index = 0
        self.last_waypoint_time = 0

        # State for smoothed velocity calculation
        self.smoothed_velocity = {'az': 0.0, 'el': 0.0}

        # State variables for the "coasting" (predictive search) mode
        self.coast_start_time = 0
        self.last_coast_update_time = 0
        self.coasting_target_pos = {'az': 0.0, 'el': 0.0}

        # Kalman Filter
        self.kf = KalmanFilter(dt=0.1, process_variance=1e-5, measurement_variance=1e-4)
        self.data_points = []
        self.last_kf_prediction_time = 0

    def reset(self):
        """Resets the tracker to its initial idle state."""
        self.state = HandTrackerState.IDLE
        self.best_point['strength'] = 0
        self.best_point['time'] = 0
        self.previous_best_point = None
        self.smoothed_velocity = {'az': 0.0, 'el': 0.0}
        self.coast_start_time = 0
        self.data_points = []
        print("[HandTracker] Reset.")

    def _update_scan_parameters(self, distance_m):
        """
        Dynamically adjusts scan points and speed based on the target's distance.
        This is the core of the distance-based adaptation.
        """
        # 1. SCAN RADIUS: Set to half the LiDAR's FOV.
        # This ensures the edge of the LiDAR's sensing cone passes through the
        # last known target position, maximizing the chance of a hit in a tight circle.
        self.scan_radius = (self.LIDAR_FOV * 5)

        # 2. SCAN POINTS: More points for closer targets, fewer for distant ones.
        # Closer targets have higher apparent velocity and benefit from a denser scan pattern.
        min_points, max_points = 6, 16
        # We assume a maximum effective tracking distance for this scaling logic (e.g., 12 meters).
        effective_dist_max = 12.0
        # Linearly interpolate the number of points based on distance.
        self.scan_points = int(max_points - (distance_m / effective_dist_max) * (max_points - min_points))
        # Clamp the value to stay within the defined min/max bounds.
        self.scan_points = max(min_points, min(max_points, self.scan_points))

        # 3. TIME PER WAYPOINT (Scan Speed): Scan faster for closer targets.
        # Linger longer on points for distant targets, which may have a weaker return signal.
        min_time, max_time = 0.02, 0.06  # in seconds
        self.time_per_waypoint = min_time + (distance_m / effective_dist_max) * (max_time - min_time)
        # Clamp the value to stay within the defined min/max bounds.
        self.time_per_waypoint = max(min_time, min(max_time, self.time_per_waypoint))

        print(f"[HandTracker] Distance: {distance_m:.2f}m -> "
              f"Radius: {self.scan_radius:.2f}°, "
              f"Points: {self.scan_points}, "
              f"Time/Point: {self.time_per_waypoint:.3f}s")

    def _generate_scan_path(self, center_az, center_el):
        """
        Generates a circular scan path using the current dynamic scan parameters.
        """
        self.scan_path = []
        if self.scan_points <= 0: return  # Safety check

        for i in range(self.scan_points):
            angle = (i / self.scan_points) * 2 * math.pi
            az_offset = self.scan_radius * math.cos(angle)
            el_offset = self.scan_radius * math.sin(angle)
            self.scan_path.append((center_az + az_offset, center_el + el_offset))

        norm_az = center_az % 360
        print(
            f"[HandTracker] Generated new {self.scan_points}-point scan path centered at Az={norm_az:.1f}, El={center_el:.1f}")

    def update(self, current_az, current_el, measurement, shared_data):
        """
        Main update loop for the tracker's state machine.
        """
        current_time = time.time()

        # STATE: IDLE - Waiting for a target
        if self.state == HandTrackerState.IDLE:
            if measurement:
                dist, strength = measurement
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                   'time': current_time}
                self.previous_best_point = self.best_point.copy()
                self.data_points.append((current_az, current_el, strength))

                # ** Dynamically set parameters based on the first detection **
                self._update_scan_parameters(dist)

                self._generate_scan_path(current_az, current_el)
                self.scan_index = 0
                self.last_waypoint_time = current_time
                print(f"[HandTracker] Acquired target. Starting scan at Az={current_az:.1f}, El={current_el:.1f}")

        # STATE: SCANNING - Actively tracking the target
        elif self.state == HandTrackerState.SCANNING:
            if len(self.data_points) > 8:
                # Basic accuracy check, a more sophisticated one can be implemented
                if self.best_point['strength'] > 100: # Example threshold
                    print("[HandTracker] Switching to Kalman Filter tracking.")
                    self.state = HandTrackerState.KALMAN_TRACKING
                    self.kf.x = np.array([[self.best_point['az']], [self.best_point['el']], [0], [0]]) # Initialize with current position
                    self.last_kf_prediction_time = current_time
                    return

            # Transition to COASTING if the target hasn't been seen for too long
            if current_time - self.best_point['time'] > self.timeout:
                print("[HandTracker] Target lost. Entering COASTING mode.")
                self.state = HandTrackerState.COASTING
                self.coast_start_time = current_time
                self.last_coast_update_time = current_time
                self.coasting_target_pos = {'az': self.best_point['az'], 'el': self.best_point['el']}
                return

            # If we have a measurement, check if it's better than our current best point
            if measurement:
                dist, strength = measurement
                self.data_points.append((current_az, current_el, strength))

                if strength > self.best_point['strength']:
                    self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                       'time': current_time}

            # Move to the next waypoint in the scan path if enough time has passed
            time_since_last_waypoint = current_time - self.last_waypoint_time
            if time_since_last_waypoint >= self.time_per_waypoint:
                waypoints_to_advance = max(1, int(time_since_last_waypoint / self.time_per_waypoint))
                self.last_waypoint_time = current_time
                for _ in range(waypoints_to_advance):
                    if not self.scan_path: continue  # Safety check

                    command_az, command_el = self.scan_path[self.scan_index]
                    command_motors_to_target(command_az, command_el, shared_data)
                    self.scan_index = (self.scan_index + 1)

                    # If a full scan cycle is complete, generate a new predicted path
                    if self.scan_index >= len(self.scan_path):
                        self.scan_index = 0
                        next_center_az, next_center_el = self.best_point['az'], self.best_point['el']

                        # Calculate smoothed velocity and predict the next center point
                        if self.previous_best_point and self.previous_best_point['strength'] > 0:
                            dt = self.best_point['time'] - self.previous_best_point['time']
                            if dt > 0:
                                raw_delta_az = self.best_point['az'] - self.previous_best_point['az']
                                # Handle azimuth wrap-around (e.g., from 359 to 1 degree)
                                if raw_delta_az > 180:
                                    raw_delta_az -= 360
                                elif raw_delta_az < -180:
                                    raw_delta_az += 360

                                raw_delta_el = self.best_point['el'] - self.previous_best_point['el']

                                s = self.velocity_smoothing_factor
                                self.smoothed_velocity['az'] = (s * self.smoothed_velocity['az']) + (
                                            (1 - s) * (raw_delta_az / dt))
                                self.smoothed_velocity['el'] = (s * self.smoothed_velocity['el']) + (
                                            (1 - s) * (raw_delta_el / dt))

                                # Predict based on one full scan cycle time
                                prediction_time = len(self.scan_path) * self.time_per_waypoint
                                next_center_az = self.best_point['az'] + (
                                            self.smoothed_velocity['az'] * prediction_time * self.prediction_factor)
                                next_center_el = self.best_point['el'] + (
                                            self.smoothed_velocity['el'] * prediction_time * self.prediction_factor)

                        self.previous_best_point = self.best_point.copy()

                        # ** Update scan parameters for the next cycle based on the latest distance **
                        self._update_scan_parameters(self.best_point['dist'])

                        self._generate_scan_path(next_center_az, next_center_el)
                        # Decay strength to prioritize finding a new, stronger signal
                        self.best_point['strength'] *= 0.8

        # STATE: COASTING - Target lost, predicting its path to reacquire
        elif self.state == HandTrackerState.COASTING:
            # If coasting for too long, give up and reset
            if current_time - self.coast_start_time > self.coast_timeout:
                print("[HandTracker] Coasting failed to reacquire target. Resetting.")
                self.reset()
                return

            # Success: Reacquired a target while coasting
            if measurement:
                dist, strength = measurement
                print("[HandTracker] Reacquired target during coast! Resuming scan.")
                self.state = HandTrackerState.SCANNING
                self.best_point = {'az': current_az, 'el': current_el, 'dist': dist, 'strength': strength,
                                   'time': current_time}

                # ** Update parameters based on reacquired distance **
                self._update_scan_parameters(dist)

                self._generate_scan_path(current_az, current_el)
                self.scan_index = 0
                self.last_waypoint_time = current_time
                return

            # Continue moving along the predicted path
            dt = current_time - self.last_coast_update_time
            if dt > 0:
                self.last_coast_update_time = current_time

                predicted_delta_az = self.smoothed_velocity['az'] * dt
                predicted_delta_el = self.smoothed_velocity['el'] * dt

                self.coasting_target_pos['az'] += predicted_delta_az
                self.coasting_target_pos['el'] += predicted_delta_el

                command_motors_to_target(self.coasting_target_pos['az'], self.coasting_target_pos['el'], shared_data)

        # STATE: KALMAN_TRACKING - Using the Kalman Filter for prediction
        elif self.state == HandTrackerState.KALMAN_TRACKING:
            if not measurement:
                print("[HandTracker] Target lost during Kalman tracking. Switching back to hand tracking.")
                self.state = HandTrackerState.SCANNING
                self.data_points = []
                return

            dist, strength = measurement
            # Update measurement noise based on strength (trust strong signals more)
            self.kf.R = np.eye(2) * (1e-1 / max(1, strength))
            self.kf.update(np.array([[current_az], [current_el]]))

            # Predict 0.5 seconds into the future
            if current_time - self.last_kf_prediction_time > 0.05: # Limit prediction rate
                predicted_state = self.kf.predict()
                future_time = 0.5
                future_az = predicted_state[0][0] + predicted_state[2][0] * future_time
                future_el = predicted_state[1][0] + predicted_state[3][0] * future_time
                command_motors_to_target(future_az, future_el, shared_data)
                self.last_kf_prediction_time = current_time

class ReactiveTracker:
    """Non-predictive tracker for immediate, reactive tracking of any target."""

    def __init__(self, smoothing_factor=0.4):
        self.smoothing_factor = smoothing_factor
        self.target_az = None
        self.target_el = None
        self.measurement_history = deque(maxlen=5)
        self.last_update_time = time.time()

    def update(self, azimuth, elevation, distance, strength):
        current_time = time.time()
        measurement = {'az': azimuth, 'el': elevation, 'dist': distance, 'strength': strength, 'time': current_time}
        self.measurement_history.append(measurement)
        adaptive_smoothing = self._calculate_adaptive_smoothing(strength)

        if self.target_az is None:
            self.target_az = azimuth
            self.target_el = elevation
        else:
            az_diff = azimuth - self.target_az
            if az_diff > 180:
                az_diff -= 360
            elif az_diff < -180:
                az_diff += 360
            self.target_az = self.target_az + (az_diff * adaptive_smoothing)
            self.target_el = (self.target_el * (1 - adaptive_smoothing) + elevation * adaptive_smoothing)
            self.target_az = self.target_az % 360

        self.last_update_time = current_time
        return self.target_az, self.target_el

    def _calculate_adaptive_smoothing(self, strength):
        base_smoothing = self.smoothing_factor
        if strength > 800:
            strength_factor = 1.5
        elif strength > 400:
            strength_factor = 1.2
        elif strength > 200:
            strength_factor = 0.8
        else:
            strength_factor = 0.5
        consistency_factor = self._check_measurement_consistency()
        adaptive_factor = base_smoothing * strength_factor * consistency_factor
        return max(0.1, min(1.0, adaptive_factor))

    def _check_measurement_consistency(self):
        if len(self.measurement_history) < 3: return 1.0
        recent_az = [m['az'] for m in list(self.measurement_history)[-3:]]
        recent_el = [m['el'] for m in list(self.measurement_history)[-3:]]
        az_diffs = []
        for i in range(1, len(recent_az)):
            diff = recent_az[i] - recent_az[i - 1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            az_diffs.append(abs(diff))
        el_diffs = [abs(recent_el[i] - recent_el[i - 1]) for i in range(1, len(recent_el))]
        if az_diffs and el_diffs:
            total_change = np.mean(az_diffs) + np.mean(el_diffs)
            if total_change < 2.0:
                return 1.2
            elif total_change < 5.0:
                return 1.0
            elif total_change < 10.0:
                return 0.8
            else:
                return 0.6
        return 1.0

    def get_current_target(self):
        if self.target_az is None: return None
        return self.target_az, self.target_el


def command_motors_to_target(azimuth, elevation, shared_data):
    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True


def run_tracking_logic(shared_data):
    """Main tracking logic process."""
    print("[TrackingLogic] Starting tracking logic process...")

    # Initialize components
    clutter_filter = ClutterFilter(shared_data.get("background_path", "background_data.npy").value)
    kf = KalmanFilter(dt=0.1, process_variance=1e-5, measurement_variance=1e-4)

    reactive_tracker = ReactiveTracker()
    hand_tracker = HandTracker()

    state = TrackingState.IDLE
    last_prediction_time = time.time()
    prediction_interval = 0.1

    shared_data["tracking_logic_ready"].value = True

    print("[TrackingLogic] Ready and running...")
    print("  - debug_mode=True: High-performance predictive hand tracking")
    print("  - reactive_mode=True: Simple reactive tracking")
    print("  - Normal mode: Advanced orbital tracking")

    while not shared_data["shutdown"].value:
        try:
            current_time = time.time()
            with shared_data["lidar_data"].get_lock():
                dist, strength, timestamp = shared_data["lidar_data"][:]
            current_az = shared_data["stepper_degrees"].value
            current_el = shared_data["servo_degrees"].value
            measurement_valid = (50.0 <= dist <= 600.0)

            if shared_data["debug_mode"].value:
                if state != TrackingState.DEBUG_MODE:
                    print("[TrackingLogic] Switching to DEBUG_MODE (Predictive Hand Tracking)")
                    state = TrackingState.DEBUG_MODE
                    hand_tracker.reset()
                is_valid_target = measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist,
                                                                                       strength)
                measurement_data = (dist, strength) if is_valid_target else None
                hand_tracker.update(current_az, current_el, measurement_data, shared_data)
                if hand_tracker.state == HandTrackerState.SCANNING:
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = hand_tracker.best_point['az']
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = hand_tracker.best_point['el']

            elif shared_data["reactive_mode"].value:
                # ... (rest of the logic remains the same)
                if state != TrackingState.REACTIVE_MODE:
                    print("[TrackingLogic] Switching to REACTIVE_MODE (Non-predictive tracking)")
                    state = TrackingState.REACTIVE_MODE
                    reactive_tracker = ReactiveTracker()
                if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                    target_az, target_el = reactive_tracker.update(current_az, current_el, dist, strength)
                    command_motors_to_target(target_az, target_el, shared_data)
                    with shared_data["predicted_azimuth"].get_lock():
                        shared_data["predicted_azimuth"].value = target_az
                    with shared_data["predicted_elevation"].get_lock():
                        shared_data["predicted_elevation"].value = target_el
            else:
                # ... (rest of the logic remains the same)
                if shared_data["acquire_points"].value:
                    if state != TrackingState.ACQUIRING:
                        print("[TrackingLogic] Switching to ACQUIRING mode")
                        state = TrackingState.ACQUIRING
                        acquirer = Acquirer()
                        shared_data["acquirer_status"].value = 1
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):
                        if acquirer.add_measurement(current_az, current_el, dist, current_time):
                            initial_state = acquirer.compute_initial_state()
                            if initial_state is not None:


                                shared_data["ekf_initialized"].value = True
                                shared_data["acquire_points"].value = False
                                shared_data["acquirer_status"].value = 0
                elif shared_data["lidar_track_mode_active"].value:
                    if state != TrackingState.TRACKING:
                        print("[TrackingLogic] Switching to TRACKING mode (Predictive)")
                        state = TrackingState.TRACKING
                    if measurement_valid and clutter_filter.is_valid_target(current_az, current_el, dist, strength):

                



            time.sleep(0.0005)

        except Exception as e:
            print(f"[TrackingLogic] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    print("[TrackingLogic] Shutting down...")