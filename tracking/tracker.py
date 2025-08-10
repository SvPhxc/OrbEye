# ==============================================================================
# tracking/tracker.py (MODIFIED)
# ------------------------------------------------------------------------------
# Imports its low-level dependencies from `motor_utils` to prevent
# circular imports.
# ==============================================================================

import numpy as np
import math
import time
from motors.motor_utils import track_target  # CORRECTED IMPORT

SEARCH_RADIUS_DEG = 2.5
SEARCH_STEP_DEG = 0.5
DWELL_TIME_S = 0.02


def active_track_target(pi, shared_data, movement_queue):
    """
    Performs a hybrid tracking routine.
    """
    center_az = shared_data["predicted_azimuth"].value
    center_el = shared_data["predicted_elevation"].value

    best_point = None
    max_strength = -1

    for d_az in np.arange(-SEARCH_RADIUS_DEG, SEARCH_RADIUS_DEG + SEARCH_STEP_DEG, SEARCH_STEP_DEG):
        for d_el in np.arange(-SEARCH_RADIUS_DEG, SEARCH_RADIUS_DEG + SEARCH_STEP_DEG, SEARCH_STEP_DEG):
            if shared_data['shutdown'].value or not shared_data['ekf_running'].value:
                return

            target_az = (center_az + d_az) % 360
            target_el = max(0, min(90, center_el + d_el))

            track_target(pi, target_az, target_el, 0.0001, movement_queue, shared_data)
            time.sleep(DWELL_TIME_S)

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
                }

    min_strength_threshold = 1000 if shared_data["debug_mode"].value else 4000

    if best_point and best_point['strength'] > min_strength_threshold:
        track_target(pi, best_point['az'], best_point['el'], 0.0001, movement_queue, shared_data)

        sp = shared_data["satellite_points"]
        with sp.get_lock():
            sp[0] = best_point['az']
            sp[1] = best_point['el']
            sp[2] = best_point['strength']
            sp[3] = best_point['distance_m'] * 100.0

        shared_data["satellite_detected"].value = True