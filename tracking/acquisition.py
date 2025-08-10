# --- CORRECTED FILE: tracking/acquisition.py ---

import numpy as np
import math
import time
from astropy.time import Time
from motors.motor_controller import track_target
from tracking.tle_utils import get_tle_prediction

# --- Constants for the acquisition process ---
SETTLE_TIME_S = 0.05
TLE_SEARCH_WINDOW_DEG = 30.0
TLE_SEARCH_STEP_DEG = 5.0
REFINE_RADIUS_DEG = 2.0  # Increased radius slightly for more robustness
REFINE_STEP_DEG = 0.4
VELOCITY_ESTIMATE_DELAY_S = 0.5
PREDICTION_TIME_S = 0.7


def _get_and_consume_detection(shared_data):
    """
    Atomically checks for a detection, retrieves the data, and resets the flag.
    This prevents race conditions and ensures each detection is processed once.
    """
    if shared_data["satellite_detected"].value:
        with shared_data["satellite_points"].get_lock():
            # Copy data to a local dictionary
            detection = {
                'az': shared_data["satellite_points"][0],
                'el': shared_data["satellite_points"][1],
                'strength': shared_data["satellite_points"][2],
                'distance_m': shared_data["satellite_points"][3] / 100.0,
                'timestamp': time.time()  # Use current time as acquisition time
            }
        # Reset the flag to signal that the data has been consumed
        shared_data["satellite_detected"].value = False
        return detection
    return None


def _refine_target(pi, shared_data, movement_queue, rough_az, rough_el):
    """
    Performs a local 'hill-climbing' search to find the peak signal strength.
    """
    print(f"[ACQUIRE] Refining target around ({rough_az:.1f}°, {rough_el:.1f}°)...")
    best_point = None
    max_strength = -1

    for d_az in np.arange(-REFINE_RADIUS_DEG, REFINE_RADIUS_DEG + REFINE_STEP_DEG, REFINE_STEP_DEG):
        for d_el in np.arange(-REFINE_RADIUS_DEG, REFINE_RADIUS_DEG + REFINE_STEP_DEG, REFINE_STEP_DEG):
            if shared_data['shutdown'].value: return None
            target_az = (rough_az + d_az) % 360
            target_el = max(0, min(90, rough_el + d_el))

            track_target(pi, target_az, target_el, 0.0001, movement_queue, shared_data)
            time.sleep(SETTLE_TIME_S)

            with shared_data["lidar_data"].get_lock():
                strength = shared_data["lidar_data"][1]
                distance_cm = shared_data["lidar_data"][0]

            if strength > max_strength:
                max_strength = strength
                best_point = {
                    'az': shared_data["stepper_degrees"].value,
                    'el': shared_data["servo_degrees"].value,
                    'strength': strength,
                    'distance_m': distance_cm / 100.0,
                    'timestamp': time.time()
                }

    if best_point:
        print(
            f"[ACQUIRE] Refined to Str {best_point['strength']:.0f} @ ({best_point['az']:.1f}°, {best_point['el']:.1f}°)")
        track_target(pi, best_point['az'], best_point['el'], 0.0001, movement_queue, shared_data)
        time.sleep(SETTLE_TIME_S)
    return best_point


def run_acquisition_sequence(pi, shared_data, movement_queue, tle_data):
    """ Main orchestration function for acquiring a drone. """
    print("\n--- STARTING TLE-GUIDED ACQUISITION SEQUENCE ---")
    acquired_points = []

    # Ensure detection flag is clear before starting
    shared_data["satellite_detected"].value = False

    # --- PHASE 1: TLE-Guided Search for First Point ---
    print("[ACQUIRE] Phase 1: Searching for first point based on TLE...")
    predicted_az, predicted_el = get_tle_prediction(tle_data, Time.now())
    half_window = TLE_SEARCH_WINDOW_DEG / 2

    initial_detection = None
    for el in np.arange(max(0, predicted_el - half_window), min(90, predicted_el + half_window), TLE_SEARCH_STEP_DEG):
        for az in np.arange(predicted_az - half_window, predicted_az + half_window, TLE_SEARCH_STEP_DEG):
            if shared_data['shutdown'].value: return False
            track_target(pi, az, el, 0.0001, movement_queue, shared_data)
            time.sleep(SETTLE_TIME_S)
            initial_detection = _get_and_consume_detection(shared_data)
            if initial_detection:
                print(f"[ACQUIRE] Initial coarse detection via background subtraction!")
                break
        if initial_detection: break

    if not initial_detection:
        print("[ACQUIRE] Failed to find any target in the TLE search window.")
        return False

    point1 = _refine_target(pi, shared_data, movement_queue, initial_detection['az'], initial_detection['el'])
    if not point1:
        print("[ACQUIRE] Failed to refine first point.")
        return False
    acquired_points.append(point1)

    # --- PHASE 2: Acquire Point 2 for Velocity Estimate ---
    print(f"\n[ACQUIRE] Phase 2: Waiting {VELOCITY_ESTIMATE_DELAY_S}s before searching for second point...")
    time.sleep(VELOCITY_ESTIMATE_DELAY_S)

    # --- LOGIC FIX: Point back to P1's location and wait for a new detection ---
    print(f"[ACQUIRE] Re-pointing to last known location ({point1['az']:.1f}°, {point1['el']:.1f}°) to find P2.")
    track_target(pi, point1['az'], point1['el'], 0.0001, movement_queue, shared_data)

    point2_detection = None
    wait_start_time = time.time()
    while time.time() - wait_start_time < 2.0:  # Wait up to 2s for drone to pass through beam
        point2_detection = _get_and_consume_detection(shared_data)
        if point2_detection:
            print("[ACQUIRE] Second coarse detection acquired.")
            break
        time.sleep(0.01)

    if not point2_detection:
        print("[ACQUIRE] Failed to re-acquire target for second point.")
        return False

    point2 = _refine_target(pi, shared_data, movement_queue, point2_detection['az'], point2_detection['el'])
    if not point2 or (point2['timestamp'] - point1['timestamp'] < 0.2):
        print("[ACQUIRE] Failed to acquire a distinct second point.")
        return False
    acquired_points.append(point2)

    # --- PHASE 3: Predictive Acquisition of Point 3 ---
    print("\n[ACQUIRE] Phase 3: Predicting position for third point...")
    dt = point2['timestamp'] - point1['timestamp']
    delta_az = (point2['az'] - point1['az'] + 540) % 360 - 180
    vel_az = delta_az / dt
    vel_el = (point2['el'] - point1['el']) / dt
    print(f"[ACQUIRE] Estimated Velocity: {vel_az:.2f}°/s Az, {vel_el:.2f}°/s El")

    predicted_az_p3 = (point2['az'] + vel_az * PREDICTION_TIME_S) % 360
    predicted_el_p3 = max(0, min(90, point2['el'] + vel_el * PREDICTION_TIME_S))

    print(f"[ACQUIRE] Moving to predicted location: ({predicted_az_p3:.1f}°, {predicted_el_p3:.1f}°)")
    track_target(pi, predicted_az_p3, predicted_el_p3, 0.0001, movement_queue, shared_data)

    point3_detection = None
    wait_start_time = time.time()
    while time.time() - wait_start_time < 2.0:
        point3_detection = _get_and_consume_detection(shared_data)
        if point3_detection:
            print("[ACQUIRE] Third coarse detection acquired at predicted location.")
            break
        time.sleep(0.01)

    if not point3_detection:
        print("[ACQUIRE] No detection at predicted location. Attempting refinement anyway.")
        point3 = _refine_target(pi, shared_data, movement_queue, predicted_az_p3, predicted_el_p3)
    else:
        point3 = _refine_target(pi, shared_data, movement_queue, point3_detection['az'], point3_detection['el'])

    if not point3:
        print("[ACQUIRE] Failed to acquire third point.")
        return False
    acquired_points.append(point3)

    # --- PHASE 4: Handoff to EKF ---
    print("\n[ACQUIRE] Success! Populating buffer with 3 points for EKF.")
    points_buffer = shared_data["points_buffer"]
    with shared_data["points_count"].get_lock():
        for i, point in enumerate(acquired_points):
            base_idx = i * 5  # 5 values per point now
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']
            points_buffer[base_idx + 4] = point['timestamp']  # CRITICAL: Save the timestamp
            print(
                f"  Point {i + 1}: Az={point['az']:.1f}, El={point['el']:.1f}, Dist={point['distance_m']:.2f}m, Time={point['timestamp']:.2f}")

        shared_data["points_count"].value = len(acquired_points)

    return True


def run_manual_acquisition_sequence(pi, shared_data, movement_queue):
    """
    A simplified acquisition sequence for debug mode (e.g., tracking a hand).
    It assumes the user is already pointing the LiDAR at the target.
    """
    print("\n--- STARTING MANUAL ACQUISITION SEQUENCE (DEBUG MODE) ---")
    acquired_points = []

    # User should be pointing at the target already. We just need to refine.
    current_az = shared_data['stepper_degrees'].value
    current_el = shared_data['servo_degrees'].value

    print("[ACQUIRE-DBG] Acquiring first point...")
    point1 = _refine_target(pi, shared_data, movement_queue, current_az, current_el)
    if not point1: return False
    acquired_points.append(point1)

    print("[ACQUIRE-DBG] Acquiring second point after a short delay...")
    time.sleep(0.7)  # A longer delay for slower hand movements
    point2 = _refine_target(pi, shared_data, movement_queue, point1['az'], point1['el'])
    if not point2 or (point2['timestamp'] - point1['timestamp'] < 0.2): return False
    acquired_points.append(point2)

    print("[ACQUIRE-DBG] Acquiring third point...")
    time.sleep(0.7)
    point3 = _refine_target(pi, shared_data, movement_queue, point2['az'], point2['el'])
    if not point3: return False
    acquired_points.append(point3)

    # --- Handoff to EKF ---
    print("\n[ACQUIRE-DBG] Success! Populating buffer with 3 points for EKF.")
    points_buffer = shared_data["points_buffer"]
    with shared_data["points_count"].get_lock():
        for i, point in enumerate(acquired_points):
            base_idx = i * 5
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']
            points_buffer[base_idx + 4] = point['timestamp']
        shared_data["points_count"].value = len(acquired_points)

    return True