import numpy as np
import math
import time
from astropy.time import Time
from tracking.tle_utils import get_tle_prediction

# --- Constants for the acquisition process ---
SETTLE_TIME_S = 0.05
TLE_SEARCH_WINDOW_DEG = 30.0
TLE_SEARCH_STEP_DEG = 5.0
REFINE_RADIUS_DEG = 2.0
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


def _refine_target(pi, shared_data, movement_queue, rough_az, rough_el, track_target_func):
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

            track_target_func(pi, target_az, target_el, 0.0001, movement_queue, shared_data)
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
        track_target_func(pi, best_point['az'], best_point['el'], 0.0001, movement_queue, shared_data)
        time.sleep(SETTLE_TIME_S)
    return best_point


def run_acquisition_sequence(pi, shared_data, movement_queue, tle_data, track_target_func):
    """ Main orchestration function for acquiring a drone. """
    print("\n--- STARTING TLE-GUIDED ACQUISITION SEQUENCE ---")
    acquired_points = []

    shared_data["satellite_detected"].value = False

    # --- PHASE 1: TLE-Guided Search for First Point ---
    print("[ACQUIRE] Phase 1: Searching for first point based on TLE...")
    predicted_az, predicted_el = get_tle_prediction(tle_data, Time.now())
    half_window = TLE_SEARCH_WINDOW_DEG / 2

    initial_detection = None
    for el in np.arange(max(0, predicted_el - half_window), min(90, predicted_el + half_window), TLE_SEARCH_STEP_DEG):
        for az in np.arange(predicted_az - half_window, predicted_az + half_window, TLE_SEARCH_STEP_DEG):
            if shared_data['shutdown'].value: return False
            track_target_func(pi, az, el, 0.0001, movement_queue, shared_data)
            time.sleep(SETTLE_TIME_S)
            initial_detection = _get_and_consume_detection(shared_data)
            if initial_detection:
                print(f"[ACQUIRE] Initial coarse detection via background subtraction!")
                break
        if initial_detection: break

    if not initial_detection:
        print("[ACQUIRE] Failed to find any target in the TLE search window.")
        return False

    point1 = _refine_target(pi, shared_data, movement_queue, initial_detection['az'], initial_detection['el'],
                            track_target_func)
    if not point1:
        print("[ACQUIRE] Failed to refine first point.")
        return False
    acquired_points.append(point1)

    # --- PHASE 2: Acquire Point 2 for Velocity Estimate ---
    print(f"\n[ACQUIRE] Phase 2: Waiting {VELOCITY_ESTIMATE_DELAY_S}s before searching for second point...")
    time.sleep(VELOCITY_ESTIMATE_DELAY_S)

    print(f"[ACQUIRE] Re-pointing to last known location ({point1['az']:.1f}°, {point1['el']:.1f}°) to find P2.")
    track_target_func(pi, point1['az'], point1['el'], 0.0001, movement_queue, shared_data)

    point2_detection = None
    wait_start_time = time.time()
    while time.time() - wait_start_time < 2.0:
        point2_detection = _get_and_consume_detection(shared_data)
        if point2_detection:
            print("[ACQUIRE] Second coarse detection acquired.")
            break
        time.sleep(0.01)

    if not point2_detection:
        print("[ACQUIRE] Failed to re-acquire target for second point.")
        return False

    point2 = _refine_target(pi, shared_data, movement_queue, point2_detection['az'], point2_detection['el'], track_ta