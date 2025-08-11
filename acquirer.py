# acquirer.py

import time
import math
import numpy as np

# Tunable parameters
COARSE_HALF_SPAN_DEG = 15.0
COARSE_STEP_DEG = 3.0
REFINE_RADIUS_DEG = 2.0
REFINE_STEP_DEG = 0.5
DWELL_TIME_S = 0.05
MAX_ACQUISITION_TIME_S = 45.0
MIN_POINT_SEPARATION_S = 0.2
MAX_REFINEMENT_ATTEMPTS = 3
POSE_REACHED_TIMEOUT_S = 2.0


def _shortest_angular_delta(target, current):
    return ((target - current + 540.0) % 360.0) - 180.0


def _normalize_az(az):
    return az % 360.0


def _clamp_tilt(el):
    return max(0.0, min(90.0, el))


def _read_lidar(shared):
    arr = shared["lidar_data"]
    with arr.get_lock():
        d, s, ts = float(arr[0]), float(arr[1]), float(arr[2])
    return d, s, ts


def _populate_points_buffer(shared, points):
    buf = shared["points_buffer"]
    count = shared["points_count"]
    with count.get_lock(), buf.get_lock():
        for i, p in enumerate(points):
            base = i * 4
            buf[base + 0] = float(p['az'])
            buf[base + 1] = float(p['el'])
            buf[base + 2] = float(p['distance_m'])
            buf[base + 3] = float(p['strength'])
        count.value = len(points)


def _wait_for_fresh_lidar(shared, old_ts, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        d, s, ts = _read_lidar(shared)
        if ts > old_ts:
            az = float(shared["stepper_degrees"].value)
            el = float(shared["servo_degrees"].value)
            return {'az': az, 'el': el, 'distance_m': d / 100.0, 'strength': s, 'timestamp': ts}
        time.sleep(0.005)
    return None


def _move_and_wait(shared, target_az, target_el):
    """Commands the hardware controller to move and waits for completion."""
    if shared["shutdown"].value: return False

    shared["target_azimuth"].value = target_az
    shared["target_elevation"].value = target_el
    shared["target_reached"].value = False
    shared["go_to_target"].value = True

    deadline = time.monotonic() + POSE_REACHED_TIMEOUT_S
    while time.monotonic() < deadline and not shared["shutdown"].value:
        if shared["target_reached"].value:
            time.sleep(DWELL_TIME_S)  # Dwell to let LiDAR stabilize
            return True
        time.sleep(0.01)

    # If it times out, stop the movement
    shared["go_to_target"].value = False
    print(f"[Acquirer] WARN: Timeout waiting to reach pose ({target_az:.1f}, {target_el:.1f})")
    return False


def _generate_spiral_grid(center_az, center_el, half_span_deg, step_deg):
    """Yields (az, el) positions in an outward spiral."""
    yield center_az, _clamp_tilt(center_el)
    max_offset_steps = int(math.ceil(half_span_deg / step_deg))
    for layer in range(1, max_offset_steps + 1):
        # Top edge
        for dx in range(-layer, layer + 1):
            yield _normalize_az(center_az + dx * step_deg), _clamp_tilt(center_el + layer * step_deg)
        # Right edge
        for dy in range(layer - 1, -layer - 1, -1):
            yield _normalize_az(center_az + layer * step_deg), _clamp_tilt(center_el + dy * step_deg)
        # Bottom edge
        for dx in range(layer - 1, -layer - 1, -1):
            yield _normalize_az(center_az + dx * step_deg), _clamp_tilt(center_el - layer * step_deg)
        # Left edge
        for dy in range(-layer + 1, layer):
            yield _normalize_az(center_az - layer * step_deg), _clamp_tilt(center_el + dy * step_deg)


def _refine_target_local(shared, rough_az, rough_el):
    """Scans a small grid around a point to find the strongest LiDAR return."""
    best_sample = None
    grid = _generate_spiral_grid(rough_az, rough_el, REFINE_RADIUS_DEG, REFINE_STEP_DEG)

    for az, el in grid:
        if shared["shutdown"].value: return None
        if not _move_and_wait(shared, az, el): continue

        _, _, old_ts = _read_lidar(shared)
        sample = _wait_for_fresh_lidar(shared, old_ts, 0.2)
        if sample is None: continue

        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 5000

        if (min_m <= sample['distance_m'] <= max_m) and (sample['strength'] >= min_strength):
            if best_sample is None or sample['strength'] > best_sample['strength']:
                best_sample = sample

    if best_sample:
        _move_and_wait(shared, best_sample['az'], best_sample['el'])
    return best_sample


def acquire_three_points(shared):
    """Main acquisition logic."""
    print("[Acquirer] Starting 3-point acquisition...")
    shared["acquirer_status"].value = 1  # 1 = Acquiring

    # For now, we seed the search at the current position.
    # Later, this could come from TLE data.
    seed_az = shared["stepper_degrees"].value
    seed_el = shared["servo_degrees"].value

    points = []

    # --- Find First Point ---
    print("[Acquirer] Searching for first point...")
    coarse_grid = _generate_spiral_grid(seed_az, seed_el, COARSE_HALF_SPAN_DEG, COARSE_STEP_DEG)
    initial_detection = None
    for az, el in coarse_grid:
        if shared["shutdown"].value: break
        if not _move_and_wait(shared, az, el): continue

        dist, strength, _ = _read_lidar(shared)
        min_m, max_m = shared["lidar_acceptance_range"]
        min_strength = 1000 if shared["debug_mode"].value else 5000

        if (min_m <= dist / 100.0 <= max_m) and (strength >= min_strength):
            print(f"[Acquirer] Coarse detection found! Strength: {strength}")
            initial_detection = {'az': az, 'el': el}
            break

    if not initial_detection:
        print("[Acquirer] Coarse search failed. Aborting.")
        shared["acquirer_status"].value = 3  # 3 = Failed
        return

    # --- Refine First Point ---
    print("[Acquirer] Refining point 1...")
    p1 = _refine_target_local(shared, initial_detection['az'], initial_detection['el'])
    if not p1:
        print("[Acquirer] Failed to refine point 1. Aborting.")
        shared["acquirer_status"].value = 3
        return
    points.append(p1)
    print(f"[Acquirer] Point 1 locked: Str {p1['strength']:.0f} @ ({p1['az']:.1f}, {p1['el']:.1f})")

    # --- Get Second and Third Points ---
    for i in range(2, 4):
        time.sleep(MIN_POINT_SEPARATION_S)
        print(f"[Acquirer] Refining point {i}...")
        p_next = _refine_target_local(shared, points[-1]['az'], points[-1]['el'])
        if not p_next:
            print(f"[Acquirer] Failed to refine point {i}. Aborting.")
            shared["acquirer_status"].value = 3
            return
        points.append(p_next)
        print(f"[Acquirer] Point {i} locked: Str {p_next['strength']:.0f} @ ({p_next['az']:.1f}, {p_next['el']:.1f})")

    # --- Success ---
    _populate_points_buffer(shared, points)
    print("[Acquirer] Successfully acquired 3 points. Handing over to EKF.")
    shared["acquirer_status"].value = 2  # 2 = Success
    shared["ekf_start"].value = True  # Signal the EKF to initialize


def run_acquirer(shared_data):
    """The main loop for the acquirer process."""
    print("[Acquirer] Process started.")
    while not shared_data["shutdown"].value:
        if shared_data["acquire_points"].value:
            # Acknowledge the trigger and set it to false so we don't re-run
            shared_data["acquire_points"].value = False
            # Run the main acquisition logic
            acquire_three_points(shared_data)

        time.sleep(0.1)  # Check for the trigger flag 10 times a second

    print("[Acquirer] Process shutting down.")