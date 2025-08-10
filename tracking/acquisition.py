# ==============================================================================
# tracking/acquisition.py (MODIFIED)
# ------------------------------------------------------------------------------
# Key Fixes:
# - Removed the unnecessary `track_target_func` parameter from all function
#   definitions.
# - All functions now correctly use the `track_target` function imported
#   directly from `motors.motor_utils`.
# ==============================================================================

import numpy as np
import math
import time
from astropy.time import Time
from motors.motor_utils import track_target  # CORRECTED: This is the source of truth
from tracking.tle_utils import get_tle_prediction, parse_tle_file

# --- Constants ---
SETTLE_TIME_S = 0.05
TLE_SEARCH_WINDOW_DEG = 30.0
TLE_SEARCH_STEP_DEG = 5.0
REFINE_RADIUS_DEG = 2.0
REFINE_STEP_DEG = 0.4
VELOCITY_ESTIMATE_DELAY_S = 0.5
PREDICTION_TIME_S = 0.7


def _get_and_consume_detection(shared_data):
    """Atomically checks for a detection and resets the flag."""
    if shared_data["satellite_detected"].value:
        with shared_data["satellite_points"].get_lock():
            detection = {
                'az': shared_data["satellite_points"][0],
                'el': shared_data["satellite_points"][1],
                'strength': shared_data["satellite_points"][2],
                'distance_m': shared_data["satellite_points"][3] / 100.0,
                'timestamp': time.time()
            }
        shared_data["satellite_detected"].value = False
        return detection
    return None


def _refine_target(pi, shared_data, movement_queue, rough_az, rough_el):
    """Performs a local 'hill-climbing' search to find the peak signal strength."""
    print(f"[ACQUIRE] Refining target around ({rough_az:.1f}°, {rough_el:.1f}°)...")
    best_point = None
    max_strength = -1

    for d_az in np.arange(-REFINE_RADIUS_DEG, REFINE_RADIUS_DEG + REFINE_STEP_DEG, REFINE_STEP_DEG):
        for d_el in np.arange(-REFINE_RADIUS_DEG, REFINE_RADIUS_DEG + REFINE_STEP_DEG, REFINE_STEP_DEG):
            if shared_data['shutdown'].value: return None
            target_az = (rough_az + d_az) % 360
            target_el = max(0, min(90, rough_el + d_el))

            # Uses the imported track_target function
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
    """Main orchestration function for acquiring a drone using TLE data."""
    print("\n--- STARTING TLE-GUIDED ACQUISITION SEQUENCE ---")
    acquired_points = []
    shared_data["satellite_detected"].value = False

    predicted_az, predicted_el = get_tle_prediction(tle_data, Time.now())
    half_window = TLE_SEARCH_WINDOW_DEG / 2

    initial_detection = None
    for el in np.arange(max(0, predicted_el - half_window), min(90, predicted_el + half_window), TLE_SEARCH_STEP_DEG):
        for az in np.arange(predicted_az - half_window, predicted_az + half_window, TLE_SEARCH_STEP_DEG):
            if shared_data['shutdown'].value: return False
            track_target(pi, az, el, 0.0001, movement_queue, shared_data)
            time.sleep(SETTLE_TIME_S)
            initial_detection = _get_and_consume_detection(shared_data)
            if initial_detection: break
        if initial_detection: break

    if not initial_detection: return False
    point1 = _refine_target(pi, shared_data, movement_queue, initial_detection['az'], initial_detection['el'])
    if not point1: return False
    acquired_points.append(point1)

    time.sleep(VELOCITY_ESTIMATE_DELAY_S)
    track_target(pi, point1['az'], point1['el'], 0.0001, movement_queue, shared_data)

    point2_detection = None
    wait_start_time = time.time()
    while time.time() - wait_start_time < 2.0:
        point2_detection = _get_and_consume_detection(shared_data)
        if point2_detection: break
        time.sleep(0.01)

    if not point2_detection: return False
    point2 = _refine_target(pi, shared_data, movement_queue, point2_detection['az'], point2_detection['el'])
    if not point2 or (point2['timestamp'] - point1['timestamp'] < 0.2): return False
    acquired_points.append(point2)

    dt = point2['timestamp'] - point1['timestamp']
    delta_az = (point2['az'] - point1['az'] + 540) % 360 - 180
    vel_az = delta_az / dt
    vel_el = (point2['el'] - point1['el']) / dt
    predicted_az_p3 = (point2['az'] + vel_az * PREDICTION_TIME_S) % 360
    predicted_el_p3 = max(0, min(90, point2['el'] + vel_el * PREDICTION_TIME_S))
    track_target(pi, predicted_az_p3, predicted_el_p3, 0.0001, movement_queue, shared_data)

    point3_detection = None
    wait_start_time = time.time()
    while time.time() - wait_start_time < 2.0:
        point3_detection = _get_and_consume_detection(shared_data)
        if point3_detection: break
        time.sleep(0.01)

    point3 = _refine_target(pi, shared_data, movement_queue, point3_detection['az'],
                            point3_detection['el']) if point3_detection else _refine_target(pi, shared_data,
                                                                                            movement_queue,
                                                                                            predicted_az_p3,
                                                                                            predicted_el_p3)
    if not point3: return False
    acquired_points.append(point3)

    points_buffer = shared_data["points_buffer"]
    with shared_data["points_count"].get_lock():
        for i, point in enumerate(acquired_points):
            base_idx = i * 5
            points_buffer[base_idx:base_idx + 5] = [point['az'], point['el'], point['distance_m'], point['strength'],
                                                    point['timestamp']]
        shared_data["points_count"].value = len(acquired_points)
    return True


def run_manual_acquisition_sequence(pi, shared_data, movement_queue):
    """Simplified acquisition for debug mode (e.g., tracking a hand)."""
    print("\n--- STARTING MANUAL ACQUISITION SEQUENCE (DEBUG MODE) ---")
    acquired_points = []

    current_az = shared_data['stepper_degrees'].value
    current_el = shared_data['servo_degrees'].value

    print("[ACQUIRE-DBG] Acquiring first point...")
    point1 = _refine_target(pi, shared_data, movement_queue, current_az, current_el)
    if not point1: return False
    acquired_points.append(point1)

    print("[ACQUIRE-DBG] Acquiring second point after a short delay...")
    time.sleep(0.7)
    point2 = _refine_target(pi, shared_data, movement_queue, point1['az'], point1['el'])
    if not point2 or (point2['timestamp'] - point1['timestamp'] < 0.2): return False
    acquired_points.append(point2)

    print("[ACQUIRE-DBG] Acquiring third point...")
    time.sleep(0.7)
    point3 = _refine_target(pi, shared_data, movement_queue, point2['az'], point2['el'])
    if not point3: return False
    acquired_points.append(point3)

    points_buffer = shared_data["points_buffer"]
    with shared_data["points_count"].get_lock():
        for i, point in enumerate(acquired_points):
            base_idx = i * 5
            points_buffer[base_idx:base_idx + 5] = [point['az'], point['el'], point['distance_m'], point['strength'],
                                                    point['timestamp']]
        shared_data["points_count"].value = len(acquired_points)
    return True