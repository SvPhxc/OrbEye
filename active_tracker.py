# active_tracker.py

import time
import math
import numpy as np

# --- Constants for the active search pattern ---
HUNT_RADIUS_DEG = 1.5  # How far from the prediction to search
HUNT_STEP_DEG = 0.75  # The density of the search grid
HUNT_LOOP_DELAY_S = 0.05  # Time to wait between each hunt cycle


def _normalize_az(az):
    return az % 360.0


def _clamp_tilt(el):
    return max(0.0, min(90.0, el))


def _read_lidar(shared):
    arr = shared["lidar_data"]
    with arr.get_lock():
        return float(arr[0]), float(arr[1]), float(arr[2])


def _wait_for_fresh_lidar(shared, old_ts, timeout_s=0.2):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        d, s, ts = _read_lidar(shared)
        if ts > old_ts:
            return d, s, ts
        time.sleep(0.005)
    return None, None, None


def _move_and_wait(shared, target_az, target_el, timeout_s=1.0):
    if shared["shutdown"].value: return False

    shared["target_azimuth"].value = target_az
    shared["target_elevation"].value = target_el
    shared["target_reached"].value = False
    shared["go_to_target"].value = True

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        if shared["target_reached"].value:
            return True
        time.sleep(0.01)

    shared["go_to_target"].value = False
    return False


def _generate_hunt_grid(center_az, center_el):
    """Generates a small spiral grid for the local search."""
    yield center_az, _clamp_tilt(center_el)  # Check center first

    num_steps = int(HUNT_RADIUS_DEG / HUNT_STEP_DEG)
    for i in range(1, num_steps + 1):
        r = i * HUNT_STEP_DEG
        # Check 4 cardinal points at radius r
        yield _normalize_az(center_az), _clamp_tilt(center_el + r)
        yield _normalize_az(center_az + r), _clamp_tilt(center_el)
        yield _normalize_az(center_az), _clamp_tilt(center_el - r)
        yield _normalize_az(center_az - r), _clamp_tilt(center_el)


def hunt_for_peak_strength(shared):
    """
    Takes the EKF prediction, searches locally for the best signal,
    and publishes it as a high-confidence "satellite_point".
    """
    # Get the EKF's prediction as the center of our search
    center_az = shared["predicted_azimuth"].value
    center_el = shared["predicted_elevation"].value

    best_sample = None

    grid = _generate_hunt_grid(center_az, center_el)

    for az, el in grid:
        if shared["shutdown"].value: break

        if not _move_and_wait(shared, az, el): continue

        _, _, old_ts = _read_lidar(shared)
        dist_cm, strength, ts = _wait_for_fresh_lidar(shared, old_ts)

        if ts is None: continue  # No fresh reading

        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 4000

        if (min_m <= dist_cm / 100.0 <= max_m) and (strength >= min_strength):
            if best_sample is None or strength > best_sample['strength']:
                best_sample = {
                    'az': shared["stepper_degrees"].value,
                    'el': shared["servo_degrees"].value,
                    'dist_cm': dist_cm,
                    'strength': strength,
                    'timestamp': ts
                }

    # If we found a good point, publish it for the EKF
    if best_sample:
        sp = shared["satellite_points"]
        with sp.get_lock():
            sp[0] = best_sample['az']
            sp[1] = best_sample['el']
            sp[2] = best_sample['dist_cm']
            sp[3] = best_sample['strength']
            sp[4] = best_sample['timestamp']
        shared["satellite_detected"].value = True


def run_active_tracker(shared_data):
    """The main loop for the active tracking process."""
    print("[ActiveTracker] Process started.")
    while not shared_data["shutdown"].value:
        # This process only runs when HF tracking is active AND the EKF is initialized
        if shared_data["lidar_track_mode_active"].value and shared_data["ekf_initialized"].value:
            hunt_for_peak_strength(shared_data)
            time.sleep(HUNT_LOOP_DELAY_S)
        else:
            # If not active, sleep longer to reduce CPU usage
            time.sleep(0.2)

    print("[ActiveTracker] Process shutting down.")