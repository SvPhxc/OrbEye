# ==============================================================================
# tracking/tracker.py
# ------------------------------------------------------------------------------
# This module contains the active tracking logic. Instead of passively pointing
# where the EKF predicts, it actively searches around the prediction to find
# the point of maximum signal strength, ensuring a solid lock on the target.
# ==============================================================================

import numpy as np
import math
import time
from motors.motor_controller import track_target  # Uses the low-level move function

# --- Constants for the active search ---
# How far from the EKF prediction to search, in degrees.
SEARCH_RADIUS_DEG = 2.5
# The step size of the search pattern. Smaller is more thorough but slower.
SEARCH_STEP_DEG = 0.5
# How long to wait at each point in the search grid for the sensor to update.
DWELL_TIME_S = 0.02


def active_track_target(pi, shared_data, movement_queue):
    """
    Performs a hybrid tracking routine.
    1. Gets the EKF's prediction as a starting point.
    2. Conducts a local search to find the real target peak.
    3. If a peak is found, it updates shared memory so the EKF can consume
       this high-quality measurement.
    """
    # Get the EKF's strategic prediction
    center_az = shared_data["predicted_azimuth"].value
    center_el = shared_data["predicted_elevation"].value

    best_point = None
    max_strength = -1

    # Perform a local grid search around the predicted center
    for d_az in np.arange(-SEARCH_RADIUS_DEG, SEARCH_RADIUS_DEG + SEARCH_STEP_DEG, SEARCH_STEP_DEG):
        for d_el in np.arange(-SEARCH_RADIUS_DEG, SEARCH_RADIUS_DEG + SEARCH_STEP_DEG, SEARCH_STEP_DEG):
            if shared_data['shutdown'].value or not shared_data['ekf_running'].value:
                return  # Exit immediately if tracking is stopped

            # Calculate the next point in our search pattern
            target_az = (center_az + d_az) % 360
            target_el = max(0, min(90, center_el + d_el))

            # Use the low-level motor command to move to the search point
            track_target(pi, target_az, target_el, 0.0001, movement_queue, shared_data)
            time.sleep(DWELL_TIME_S)

            # Check the live LiDAR reading at this point
            with shared_data["lidar_data"].get_lock():
                strength = shared_data["lidar_data"][1]
                distance_cm = shared_data["lidar_data"][0]

            # If this point is better than what we've seen, save it
            if strength > max_strength:
                max_strength = strength
                best_point = {
                    'az': shared_data["stepper_degrees"].value,
                    'el': shared_data["servo_degrees"].value,
                    'strength': strength,
                    'distance_m': distance_cm / 100.0,
                }

    # After the search, if we found a reasonably strong signal, we have a lock.
    # For debug mode, 1000 is a good threshold. For a drone, it might be higher.
    min_strength_threshold = 1000 if shared_data["debug_mode"].value else 4000

    if best_point and best_point['strength'] > min_strength_threshold:
        # Move the turret to the best point found to stay centered
        track_target(pi, best_point['az'], best_point['el'], 0.0001, movement_queue, shared_data)

        # --- This is the crucial feedback loop ---
        # Provide this high-quality, locked-on measurement to the EKF.
        # We use the existing `satellite_points` buffer for this communication.
        sp = shared_data["satellite_points"]
        with sp.get_lock():
            sp[0] = best_point['az']
            sp[1] = best_point['el']
            sp[2] = best_point['strength']
            sp[3] = best_point['distance_m'] * 100.0  # Convert back to cm

        # Signal to the EKF that a valid measurement is ready for consumption.
        shared_data["satellite_detected"].value = True

        # print(f"[ACTIVE TRACK] Peak Found: Str {best_point['strength']:.0f}") # Optional: for debugging
    else:
        # If the search found nothing, it means we likely lost the target.
        # We don't send anything to the EKF, so it will continue to coast on its
        # own predictions (Kalman 'predict' step) until we find the target again.
        pass