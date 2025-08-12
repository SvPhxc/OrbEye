# active_tracker.py

import time
import math
import numpy as np

# --- Constants for the active search pattern ---
HUNT_RADIUS_DEG = 1.0  # Smaller radius for a tighter 3x3 grid
HUNT_STEP_DEG = 1.0  # Step size defines the grid spacing
HUNT_LOOP_DELAY_S = 0.05


def _normalize_az(az):
    return az % 360.0


def _clamp_tilt(el):
    return max(0.0, min(90.0, el))


def _read_lidar(shared):
    with shared["lidar_data"].get_lock():
        return float(shared["lidar_data"][0]), float(shared["lidar_data"][1]), float(shared["lidar_data"][2])


def _wait_for_fresh_lidar(shared, old_ts, timeout_s=0.2):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        d, s, ts = _read_lidar(shared)
        if ts > old_ts: return d, s, ts
        time.sleep(0.005)
    return None, None, None


def _move_and_wait(shared, target_az, target_el, timeout_s=0.5):
    if shared["shutdown"].value: return False
    shared["target_azimuth"].value = target_az;
    shared["target_elevation"].value = target_el
    shared["target_reached"].value = False;
    shared["go_to_target"].value = True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        if shared["target_reached"].value: return True
        time.sleep(0.01)
    shared["go_to_target"].value = False
    return False


def _generate_hunt_grid(center_az, center_el):
    """Generates a 3x3 grid of 9 points centered on the prediction."""
    for dy in [-HUNT_STEP_DEG, 0, HUNT_STEP_DEG]:
        for dx in [-HUNT_STEP_DEG, 0, HUNT_STEP_DEG]:
            yield _normalize_az(center_az + dx), _clamp_tilt(center_el + dy)


def hunt_for_peak_strength(shared):
    """
    Performs a 9-point grid scan, aggregates the results into a single
    high-quality measurement, and publishes it for the EKF.
    """
    center_az = shared["predicted_azimuth"].value
    center_el = shared["predicted_elevation"].value

    valid_points = []

    grid = _generate_hunt_grid(center_az, center_el)

    for az, el in grid:
        if shared["shutdown"].value: break
        if not _move_and_wait(shared, az, el): continue

        old_ts = _read_lidar(shared)
        dist_cm, strength, ts = _wait_for_fresh_lidar(shared, old_ts)
        if ts is None: continue

        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 4000

        if (min_m <= dist_cm / 100.0 <= max_m) and (strength >= min_strength):
            valid_points.append({
                'az': shared["stepper_degrees"].value, 'el': shared["servo_degrees"].value,
                'dist_cm': dist_cm, 'strength': strength, 'timestamp': ts
            })

    # --- Aggregate the results ---
    if not valid_points: return

    # Calculate a strength-weighted average of the positions
    total_strength = sum(p['strength'] for p in valid_points)
    if total_strength == 0: return

    # Use a circular-aware average for azimuth
    s_sum = sum(p['strength'] * np.sin(np.deg2rad(p['az'])) for p in valid_points)
    c_sum = sum(p['strength'] * np.cos(np.deg2rad(p['az'])) for p in valid_points)
    avg_az = np.rad2deg(np.arctan2(s_sum, c_sum)) % 360

    # Standard weighted average for other values
    avg_el = sum(p['strength'] * p['el'] for p in valid_points) / total_strength
    avg_dist = sum(p['strength'] * p['dist_cm'] for p in valid_points) / total_strength
    avg_strength = sum(p['strength'] for p in valid_points) / len(valid_points)
    avg_ts = sum(p['timestamp'] for p in valid_points) / len(valid_points)

    # Publish this new high-quality "meta-point" for the EKF
    with shared["satellite_points"].get_lock():
        sp = shared["satellite_points"]
        sp[0] = avg_az;
        sp[1] = avg_el;
        sp[2] = avg_dist
        sp[3] = avg_strength;
        sp[4] = avg_ts
    shared["satellite_detected"].value = True


def run_active_tracker(shared_data):
    """Main loop for the active tracking process."""
    print("[ActiveTracker] Process started.")
    while not shared_data["shutdown"].value:
        if shared_data["lidar_track_mode_active"].value and shared_data["ekf_initialized"].value:
            hunt_for_peak_strength(shared_data)
            time.sleep(HUNT_LOOP_DELAY_S)
        else:
            time.sleep(0.2)
    print("[ActiveTracker] Process shutting down.")