"""
Enhanced detection algorithm for drone/satellite tracking with improved
background subtraction and multi-criteria validation.
"""

import numpy as np
import time
from collections import deque


def enhanced_detect_satellite(current_strength, current_range_cm, az_deg, el_deg,
                              shared_data, bg_index, detection_history=None):
    """
    Enhanced satellite detection with multiple validation criteria and
    temporal consistency checking.
    """
    if detection_history is None:
        detection_history = deque(maxlen=10)

    az = int(round(az_deg)) % 360
    el = int(round(el_deg))

    # Integration functions for your existing LiDAR handler


def run_lidar_with_enhanced_detection(shared_data, port="/dev/serial0", baudrate=115200):
    """
    Enhanced version of your run_lidar function with improved detection algorithms.
    """
    from LiDAR.lidar_handler import read_tfmini_data
    import serial

    # Bind shared arrays/flags
    lidar_sh = shared_data["lidar_data"]
    stepper_deg = shared_data["stepper_degrees"]
    servo_deg = shared_data["servo_degrees"]
    lidar_range_sh = shared_data.get("lidar_acceptance_range", None)

    # Enhanced detection state
    detection_history = deque(maxlen=20)
    validation_history = deque(maxlen=10)

    # Background handling
    background_array = np.empty((0, 4))
    bg_index = {}
    bg_loaded_ts = 0.0

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Enhanced LiDAR processor started...")

            while not shared_data["shutdown"].value:
                # Read TFmini frame
                distance_cm, strength = read_tfmini_data(ser)

                if distance_cm is not None and strength is not None:
                    ts = time.time()
                    with lidar_sh.get_lock():
                        lidar_sh[0] = float(distance_cm)
                        lidar_sh[1] = float(strength)
                        lidar_sh[2] = ts

                time.sleep(0.01)

                # Refresh background index
                if shared_data.get("background_ready", Value('b', False)).value and (time.time() - bg_loaded_ts > 1.0):
                    bg_index = build_enhanced_bg_index(shared_data["background_path"], quality_filter=True)
                    bg_loaded_ts = time.time()
                    print(f"[TFmini] Enhanced background index: {len(bg_index)} cells")
                    with shared_data["background_ready"].get_lock():
                        shared_data["background_ready"].value = False

                # Get current readings
                with lidar_sh.get_lock():
                    distance_cm = float(lidar_sh[0])
                    strength = float(lidar_sh[1])
                    ts = float(lidar_sh[2])

                az = float(stepper_deg.value)
                el = float(servo_deg.value)

                # Enhanced validation and detection during acquisition or tracking
                if shared_data.get("acquire_points", Value('b', False)).value or \
                        shared_data.get("ekf_running", Value('b', False)).value:

                    # Enhanced validation
                    is_valid = validate_lidar_data_enhanced(distance_cm, strength, shared_data, validation_history)

                    if is_valid and bg_index:
                        # Enhanced detection
                        is_detected, detection_score = enhanced_detect_satellite(
                            strength, distance_cm, az, el, shared_data, bg_index, detection_history
                        )

                        # Store additional detection metadata
                        if is_detected:
                            # You can add more sophisticated point storage here
                            pass

                # Background scan accumulation (unchanged)
                if shared_data.get("scan_trigger", Value('b', False)).value:
                    background_array = save_background_enhanced(background_array, lidar_sh, az, el)

                # 3-point acquisition handling (enhanced)
                if shared_data.get("acquire_points", Value('b', False)).value and shared_data.get("satellite_detected",
                                                                                                  Value('b',
                                                                                                        False)).value:
                    # Store high-quality detection points
                    az_val = shared_data["satellite_points"][0]
                    el_val = shared_data["satellite_points"][1]
                    str_val = shared_data["satellite_points"][2]
                    dist_m = shared_data["satellite_points"][3] / 100.0

                    # Add quality scoring for point selection
                    with shared_data["points_count"].get_lock():
                        k = shared_data["points_count"].value
                        if k < 3:
                            base = 4 * k
                            pb = shared_data["points_buffer"]
                            pb[base + 0] = float(az_val)
                            pb[base + 1] = float(el_val)
                            pb[base + 2] = float(dist_m)
                            pb[base + 3] = float(str_val)
                            shared_data["points_count"].value = k + 1

                            print(f"[TFmini] Stored point {k + 1}/3: "
                                  f"({az_val:.1f}°, {el_val:.1f}°) "
                                  f"dist={dist_m:.1f}m strength={str_val:.0f}")

                    shared_data["satellite_detected"].value = False

                # Save background when requested
                if shared_data.get("save_background", Value('b', False)).value:
                    np.save(shared_data["background_path"], background_array)
                    shared_data["background_ready"].value = True
                    print(f"[TFmini] Background saved: {len(background_array)} points")
                    shared_data["save_background"].value = False

    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")
    except Exception as e:
        print(f"[TFmini] Error: {e}")


def save_background_enhanced(background_array, lidar_data, stepper, servo):
    """
    Enhanced background saving with better data organization.
    """
    az = int(round(stepper)) % 360
    el = int(round(servo))

    if not (0 <= el < 90):
        return background_array

    # Grid index
    pos = el * 360 + az

    # Add timestamp for data freshness tracking
    timestamp = time.time()
    new_row = np.array([[pos, lidar_data[0], lidar_data[1], timestamp]])

    return np.append(background_array, new_row, axis=0)


def adaptive_threshold_detection(current_strength, bg_strength, current_range, bg_range):
    """
    Adaptive threshold detection that adjusts based on background characteristics.
    """
    # Base thresholds
    strength_threshold = 3000
    range_threshold = 50  # cm

    # Adaptive adjustments based on background
    if bg_strength > 10000:  # High background strength
        strength_threshold = max(5000, bg_strength * 0.3)
    elif bg_strength < 3000:  # Low background strength  
        strength_threshold = 2000

    if bg_range > 1000:  # Distant background
        range_threshold = 100
    elif bg_range < 500:  # Close background
        range_threshold = 30

    # Check thresholds
    strength_diff = abs(current_strength - bg_strength)
    range_diff = abs(current_range - bg_range)

    return (strength_diff > strength_threshold and
            range_diff > range_threshold)


# Quality assessment for acquired points
def assess_point_quality(point_data, bg_index=None):
    """
    Assess the quality of a detected point for EKF initialization.
    Returns a quality score from 0.0 to 1.0.
    """
    az, el, dist_m, strength = point_data
    quality_score = 0.0

    # Strength quality (0-0.3)
    if strength > 15000:
        quality_score += 0.3
    elif strength > 10000:
        quality_score += 0.2
    elif strength > 5000:
        quality_score += 0.1

    # Distance quality (0-0.2) - prefer mid-range distances
    if 5.0 <= dist_m <= 8.0:  # Optimal range
        quality_score += 0.2
    elif 3.0 <= dist_m <= 12.0:  # Acceptable range
        quality_score += 0.1

    # Background contrast (0-0.3)
    if bg_index:
        az_int, el_int = int(round(az)) % 360, int(round(el))
        bg_data = bg_index.get((az_int, el_int))
        if bg_data:
            bg_strength, bg_range_cm = bg_data
            strength_contrast = strength / max(bg_strength, 1000)
            range_contrast = abs(dist_m * 100 - bg_range_cm) / max(bg_range_cm, 100)

            if strength_contrast > 2.0:
                quality_score += 0.2
            elif strength_contrast > 1.5:
                quality_score += 0.1

            if range_contrast > 0.5:
                quality_score += 0.1

    # Position quality (0-0.2) - avoid extreme elevations
    if 20 <= el <= 70:  # Good elevation range
        quality_score += 0.2
    elif 10 <= el <= 80:  # Acceptable range
        quality_score += 0.1

    return min(quality_score, 1.0)


# Point selection for EKF initialization
def select_best_three_points(candidate_points, max_candidates=10):
    """
    Select the best 3 points from candidates for EKF initialization.
    Considers quality, spatial distribution, and temporal separation.
    """
    if len(candidate_points) < 3:
        return candidate_points

    # Sort by quality score
    sorted_candidates = sorted(candidate_points,
                               key=lambda p: p.get('quality', 0.0),
                               reverse=True)

    selected = []

    # Always take the highest quality point
    selected.append(sorted_candidates[0])

    # For second point, prefer good spatial separation
    for candidate in sorted_candidates[1:]:
        if len(selected) >= 2:
            break

        # Check spatial separation from already selected points
        min_separation = float('inf')
        for sel_point in selected:
            separation = np.sqrt((candidate['az'] - sel_point['az']) ** 2 +
                                 (candidate['el'] - sel_point['el']) ** 2)
            min_separation = min(min_separation, separation)

        # Require minimum 2° separation
        if min_separation > 2.0:
            selected.append(candidate)

    # For third point, consider temporal separation as well
    for candidate in sorted_candidates:
        if len(selected) >= 3:
            break

        if candidate in selected:
            continue

        # Check both spatial and temporal separation
        good_separation = True
        for sel_point in selected:
            spatial_sep = np.sqrt((candidate['az'] - sel_point['az']) ** 2 +
                                  (candidate['el'] - sel_point['el']) ** 2)
            temporal_sep = abs(candidate.get('timestamp', 0) -
                               sel_point.get('timestamp', 0))

            if spatial_sep < 1.5 and temporal_sep < 0.5:
                good_separation = False
                break

        if good_separation:
            selected.append(candidate)

    return selected[:3]
    Get
    background
    reference
    bg_data = bg_index.get((az, el))
    if not bg_data:
        return False, 0.0  # No background reference available

    bg_strength, bg_range_cm = bg_data

    # Multi-criteria detection scoring
    detection_score = 0.0
    reasons = []

    # Criterion 1: Strength difference
    strength_diff = abs(current_strength - bg_strength)
    strength_ratio = current_strength / max(bg_strength, 1000)  # Avoid division by zero

    if strength_diff > 3000:  # Significant strength change
        detection_score += min(strength_diff / 1000.0, 5.0)  # Cap at 5 points
        reasons.append(f"strength_diff={strength_diff:.0f}")

    if strength_ratio > 1.5:  # Much brighter than background
        detection_score += min(strength_ratio, 3.0)  # Cap at 3 points
        reasons.append(f"strength_ratio={strength_ratio:.2f}")

    # Criterion 2: Range difference  
    range_diff = abs(current_range_cm - bg_range_cm)
    range_ratio = abs(current_range_cm - bg_range_cm) / max(bg_range_cm, 100)

    if range_diff > 100:  # More than 1m difference
        detection_score += min(range_diff / 100.0, 3.0)  # Cap at 3 points
        reasons.append(f"range_diff={range_diff:.0f}cm")

    # Criterion 3: Expected drone characteristics
    drone_range_score = 0.0
    if 300 <= current_range_cm <= 1200:  # 3-12m expected range
        drone_range_score = 2.0
        if 500 <= current_range_cm <= 800:  # Optimal range 5-8m
            drone_range_score = 3.0
        detection_score += drone_range_score
        reasons.append(f"good_range")

    # Criterion 4: Minimum strength threshold
    if current_strength < 5000:
        detection_score *= 0.5  # Penalize weak signals
        reasons.append("weak_signal_penalty")

    # Criterion 5: Temporal consistency
    timestamp = time.time()
    detection_history.append({
        'timestamp': timestamp,
        'az': az_deg,
        'el': el_deg,
        'strength': current_strength,
        'range_cm': current_range_cm,
        'score': detection_score
    })

    # Check for consistent detections in recent history
    recent_detections = [d for d in detection_history if timestamp - d['timestamp'] < 2.0]
    if len(recent_detections) >= 3:
        avg_recent_score = np.mean([d['score'] for d in recent_detections])
        if avg_recent_score > 3.0:
            detection_score += 1.0  # Bonus for consistent detection
            reasons.append("temporal_consistency")

    # Final decision threshold
    detection_threshold = 4.0  # Adjust this based on testing
    is_detected = detection_score >= detection_threshold

    if is_detected:
        print(f"[DETECT] Detection at ({az_deg:.1f}°, {el_deg:.1f}°) "
              f"Score: {detection_score:.2f} Reasons: {', '.join(reasons)}")

        # Store detection in shared memory
        sp = shared_data["satellite_points"]
        sp[0], sp[1], sp[2], sp[3] = az_deg, el_deg, current_strength, current_range_cm
        shared_data["satellite_detected"].value = True

    return is_detected, detection_score


def validate_lidar_data_enhanced(distance_cm, strength, shared_data, history=None):
    """
    Enhanced LiDAR data validation with temporal filtering.
    """
    if history is None:
        history = deque(maxlen=5)

    # Basic validity checks
    if distance_cm in (-1, -2, -4) or strength < 100:
        return False

    # Range check for drone detection
    lidar_range = shared_data.get("lidar_acceptance_range")
    if lidar_range:
        min_range_cm = lidar_range[0] * 100
        max_range_cm = lidar_range[1] * 100
        if not (min_range_cm <= distance_cm <= max_range_cm):
            return False
    else:
        # Default range check
        if not (300 <= distance_cm <= 1200):  # 3-12m
            return False

    # Minimum strength threshold
    if strength < 2000:  # Lowered threshold for initial detection
        return False

    # Temporal consistency check
    current_time = time.time()
    history.append({
        'timestamp': current_time,
        'distance': distance_cm,
        'strength': strength
    })

    # Remove old entries
    while history and current_time - history[0]['timestamp'] > 1.0:
        history.popleft()

    # Check for reasonable stability
    if len(history) >= 3:
        distances = [h['distance'] for h in history]
        strengths = [h['strength'] for h in history]

        # Check if measurements are reasonably stable
        dist_std = np.std(distances)
        strength_std = np.std(strengths)

        # If readings are too erratic, might be noise
        if dist_std > 200 or strength_std > 5000:  # Adjust thresholds as needed
            return False

    return True


def build_enhanced_bg_index(path, quality_filter=True):
    """
    Enhanced background index building with quality filtering.
    """
    import numpy as np

    idx = {}

    try:
        bg = np.load(path)
    except Exception as e:
        print(f"[DETECT] Could not load background file '{path}': {e}")
        return idx

    # Handle different data formats
    if bg.ndim == 1 and bg.size % 4 == 0:
        bg = bg.reshape((-1, 4))

    # Group measurements by position for quality filtering
    position_groups = {}

    for row in bg:
        if len(row) < 3:
            continue

        pos = int(row[0])
        dist_cm = float(row[1])
        strength = float(row[2])

        # Decode position
        az = pos % 360
        el = pos // 360

        # Basic sanity filters
        if not (0 <= az < 360 and 0 <= el < 90):
            continue
        if not (10.0 <= dist_cm <= 2000.0):
            continue
        if not np.isfinite(dist_cm) or not np.isfinite(strength):
            continue

        # Group by position
        pos_key = (az, el)
        if pos_key not in position_groups:
            position_groups[pos_key] = []
        position_groups[pos_key].append({'distance': dist_cm, 'strength': strength})

    # Process each position group
    for pos_key, measurements in position_groups.items():
        if not measurements:
            continue

        if quality_filter and len(measurements) > 1:
            # Use median values for more robust background
            distances = [m['distance'] for m in measurements]
            strengths = [m['strength'] for m in measurements]

            # Remove obvious outliers (simple approach)
            dist_median = np.median(distances)
            strength_median = np.median(strengths)

            filtered_measurements = []
            for m in measurements:
                if (abs(m['distance'] - dist_median) < 100 and  # Within 1m
                        abs(m['strength'] - strength_median) < 2000):  # Within reasonable strength range
                    filtered_measurements.append(m)

            if filtered_measurements:
                # Use mean of filtered measurements
                avg_distance = np.mean([m['distance'] for m in filtered_measurements])
                avg_strength = np.mean([m['strength'] for m in filtered_measurements])
                idx[pos_key] = (avg_strength, avg_distance)
            else:
                # Fall back to median if all filtered out
                idx[pos_key] = (strength_median, dist_median)
        else:
            # Single measurement or no filtering
            m = measurements[0]
            idx[pos_key] = (m['strength'], m['distance'])

    print(f"[DETECT] Enhanced background index built: {len(idx)} cells")
    return idx

#