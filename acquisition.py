# File: acquisition.py

import math
import numpy as np

# --- Tunable Parameters for Acquisition ---
COARSE_HALF_SPAN_DEG = 15.0
COARSE_STEP_DEG = 3.0
REFINE_RADIUS_DEG = 2.0
REFINE_STEP_DEG = 0.4
MIN_POINT_SEPARATION_S = 0.2

def _normalize_az(az):
    return az % 360.0

def _clamp_tilt(el):
    return max(0.0, min(90.0, el))

def generate_spiral_search_path(center_az, center_el):
    max_offset_steps = int(math.ceil(COARSE_HALF_SPAN_DEG / COARSE_STEP_DEG))
    yield _normalize_az(center_az), _clamp_tilt(center_el)
    for layer in range(1, max_offset_steps + 1):
        for dx in range(-layer, layer + 1): yield _normalize_az(center_az + dx * COARSE_STEP_DEG), _clamp_tilt(center_el + layer * COARSE_STEP_DEG)
        for dy in range(layer - 1, -layer - 1, -1): yield _normalize_az(center_az + layer * COARSE_STEP_DEG), _clamp_tilt(center_el + dy * COARSE_STEP_DEG)
        for dx in range(layer - 1, -layer - 1, -1): yield _normalize_az(center_az + dx * COARSE_STEP_DEG), _clamp_tilt(center_el - layer * COARSE_STEP_DEG)
        for dy in range(-layer + 1, layer): yield _normalize_az(center_az - layer * COARSE_STEP_DEG), _clamp_tilt(center_el + dy * COARSE_STEP_DEG)

def generate_refinement_path(center_az, center_el):
    offsets = np.arange(-REFINE_RADIUS_DEG, REFINE_RADIUS_DEG + 1e-6, REFINE_STEP_DEG)
    for d_el in offsets:
        for d_az in offsets:
            yield _normalize_az(center_az + d_az), _clamp_tilt(center_el + d_el)

def check_lidar_for_target(dist_cm, strength, shared_data):
    min_m, max_m = shared_data["lidar_acceptance_range"]
    min_strength = 1000 if shared_data["debug_mode"].value else 5000
    if not (min_m <= dist_cm / 100.0 <= max_m): return False
    if strength < min_strength: return False
    return True

def populate_points_buffer(shared_data, points):
    buf = shared_data["points_buffer"]
    count = shared_data["points_count"]
    with count.get_lock():
        for i, p in enumerate(points):
            base = i * 5
            buf[base:base+5] = [float(p['az']), float(p['el']), float(p['distance_m']), float(p['strength']), float(p['timestamp'])]
        count.value = len(points)
        print(f"[Acquisition] Populated EKF buffer with {len(points)} points.")