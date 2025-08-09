"""
Enhanced 3-Point Acquisition Strategy for Drone Tracking Initialization

This module enhances the existing spiral_acquire_three function without changing names.
"""

import numpy as np
import math
import time
from collections import deque

# --- HELPER FUNCTIONS MOVED TO THE TOP ---

def check_current_detection(shared_data):
    """Check if current LiDAR reading indicates a valid detection"""
    with shared_data["lidar_data"].get_lock():
        distance_cm = shared_data["lidar_data"][0]
        strength = shared_data["lidar_data"][1]

    current_az = shared_data["stepper_degrees"].value
    current_el = shared_data["servo_degrees"].value

    # Check if within expected drone range
    # NOTE: This range (100-1200 cm) should match your drone's expected distance.
    if not (100 <= distance_cm <= 1200):
        return None

    # This is a basic strength check. The value might need tuning.
    if strength < 1000:
        return None

    return {
        'az': current_az,
        'el': current_el,
        'distance_m': distance_cm / 100.0,
        'strength': strength
    }

def optimize_point_locally(pi, shared_data, movement_queue, rough_point, search_radius=1.0):
    """Fine-tune a point by searching around it"""
    # This import is fine here as it's locally scoped
    from motors.motor_controller import track_target

    best_point = rough_point.copy()
    center_az = rough_point['az']
    center_el = rough_point['el']

    for daz in np.arange(-search_radius, search_radius + 0.1, 0.2):
        for del_el in np.arange(-search_radius, search_radius + 0.1, 0.2):
            if shared_data['shutdown'].value:
                break

            test_az = center_az + daz
            test_el = max(0, min(90, center_el + del_el))

            track_target(pi, test_az, test_el, 0.0001, movement_queue, shared_data)
            time.sleep(0.03) # Dwell time for sensor to update

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > best_point['strength']:
                best_point = detection.copy()
                best_point['timestamp'] = time.time()

    return best_point

def select_best_second_point(detections, point1):
    """Select best second point considering strength and separation"""
    if not detections:
        return None

    best_detection = None
    best_score = -1

    for detection in detections:
        # Calculate angular separation
        dist = math.sqrt((detection['az'] - point1['az'])**2 +
                        (detection['el'] - point1['el'])**2)

        # Calculate time separation
        time_sep = abs(detection['timestamp'] - point1['timestamp'])

        # Scoring logic: prioritize points that are further away in space and time
        score = detection['strength']
        if dist > 2.0:  # Reward points that are clearly separate
            score += 1000
        if time_sep > 0.2: # Reward points with a good time delta for velocity calculation
            score += 500

        if score > best_score:
            best_score = score
            best_detection = detection

    return best_detection


# --- MAIN ACQUISITION FUNCTION ---

def enhanced_spiral_acquire_three(pi, shared_data, movement_queue):
    """
    Enhanced version that replaces the existing spiral_acquire_three function.
    Keeps the same function signature but implements the 3-phase strategy:
    1. Find first point with highest strength
    2. Spiral search for second point with velocity estimation
    3. Predictive positioning for third point
    """
    print("\n--- STARTING ENHANCED 3-POINT ACQUISITION ---")

    # Import is locally scoped, so it's fine here
    from motors.motor_controller import track_target

    # Reset acquisition state
    shared_data["points_count"].value = 0

    # Storage for acquired points
    acquired_points = []

    # Search parameters
    min_strength_threshold = 2000
    spiral_radius_start = 1.0
    spiral_radius_max = 8.0
    spiral_step = 2
    scan_dwell_time = 0.05

    try:
        # Phase 1: Find first point with highest strength
        print("[3PT] Phase 1: Finding first high-strength point...")

        start_az = shared_data["stepper_degrees"].value
        start_el = shared_data["servo_degrees"].value

        best_point = None
        search_radius = 5.0

        # Concentric search around starting area
        for radius in np.arange(0.5, search_radius, 0.5):
            for angle in np.arange(0, 360, 15):
                if shared_data['shutdown'].value:
                    return False

                search_az = start_az + radius * math.cos(math.radians(angle))
                search_el = max(0, min(90, start_el + radius * math.sin(math.radians(angle))))

                track_target(pi, search_az, search_el, 0.0001, movement_queue, shared_data)
                time.sleep(scan_dwell_time)

                detection = check_current_detection(shared_data)
                if detection and detection['strength'] > min_strength_threshold:
                    if not best_point or detection['strength'] > best_point['strength']:
                        best_point = detection.copy()
                        best_point['timestamp'] = time.time()
                        print(f"[3PT] New best first point: {best_point['strength']} @ "
                              f"({best_point['az']:.1f}°, {best_point['el']:.1f}°)")

        if not best_point:
            print("[3PT] Failed to find first point")
            return False

        best_point = optimize_point_locally(pi, shared_data, movement_queue, best_point)
        acquired_points.append(best_point)
        print(f"[3PT] Acquired Point 1: {best_point['strength']} @ ({best_point['az']:.1f}°, {best_point['el']:.1f}°)")

        # Phase 2: Spiral search for second point
        print("[3PT] Phase 2: Spiral search for second point...")
        center_az, center_el = best_point['az'], best_point['el']
        spiral_detections = []
        angle, radius = 0.0, spiral_radius_start

        while radius < spiral_radius_max and not shared_data['shutdown'].value and len(spiral_detections) < 5:
            spiral_az = center_az + radius * math.cos(math.radians(angle))
            spiral_el = max(0, min(90, center_el + radius * math.sin(math.radians(angle))))

            track_target(pi, spiral_az, spiral_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                dist_from_p1 = math.sqrt((detection['az'] - best_point['az'])**2 + (detection['el'] - best_point['el'])**2)
                if dist_from_p1 > 1.0:
                    detection['timestamp'] = time.time()
                    spiral_detections.append(detection)
                    print(f"[3PT] Spiral candidate: {detection['strength']} @ ({detection['az']:.1f}°, {detection['el']:.1f}°)")

            angle += 15.0
            radius += spiral_step * (angle / 360.0)

        point2 = select_best_second_point(spiral_detections, best_point) if spiral_detections else None
        if not point2:
            print("[3PT] Failed to find suitable second point")
            return False

        acquired_points.append(point2)
        print(f"[3PT] Acquired Point 2: {point2['strength']} @ ({point2['az']:.1f}°, {point2['el']:.1f}°)")

        # Phase 3: Predictive positioning for third point
        print("[3PT] Phase 3: Predictive positioning for third point...")
        dt = point2['timestamp'] - best_point['timestamp']
        if dt <= 1e-3: dt = 0.1

        vel_az = (point2['az'] - best_point['az']) / dt
        vel_el = (point2['el'] - best_point['el']) / dt

        if abs(vel_az) > 180: vel_az -= 360 * np.sign(vel_az)

        MAX_VEL_DEG_S = 100.0
        vel_az = np.clip(vel_az, -MAX_VEL_DEG_S, MAX_VEL_DEG_S)
        vel_el = np.clip(vel_el, -MAX_VEL_DEG_S, MAX_VEL_DEG_S)
        print(f"[3PT] Estimated (clipped) velocity: {vel_az:.2f}°/s az, {vel_el:.2f}°/s el")

        prediction_time = 0.5
        predicted_az = point2['az'] + vel_az * prediction_time
        predicted_el = max(0, min(90, point2['el'] + vel_el * prediction_time))
        print(f"[3PT] Predicted position for P3: ({predicted_az:.1f}°, {predicted_el:.1f}°)")

        search_positions = []
        search_box_radius, search_step = 4.0, 2.0
        search_positions.append((predicted_az, predicted_el))

        for r in np.arange(search_step, search_box_radius + 1e-6, search_step):
            for angle_deg in np.arange(0, 360, 45):
                d_az, d_el = r * math.cos(math.radians(angle_deg)), r * math.sin(math.radians(angle_deg))
                search_positions.append((predicted_az + d_az, predicted_el + d_el))

        best_point3 = None
        for i, (pred_az, pred_el) in enumerate(search_positions):
            if shared_data['shutdown'].value: break

            print(f"[3PT] Searching P3 at ({pred_az:.1f}, {pred_el:.1f}) [Step {i+1}/{len(search_positions)}]")
            track_target(pi, pred_az, pred_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time * 2)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                detection['timestamp'] = time.time()
                if not best_point3 or detection['strength'] > best_point3['strength']:
                    best_point3 = detection.copy()
                    print(f"[3PT] Predictive candidate: {best_point3['strength']} @ ({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")
                    if best_point3['strength'] > min_strength_threshold * 1.5:
                        break

        if not best_point3:
            print("[3PT] Failed to find third point with predictive search")
            return False

        best_point3 = optimize_point_locally(pi, shared_data, movement_queue, best_point3)
        acquired_points.append(best_point3)
        print(f"[3PT] Acquired Point 3: {best_point3['strength']} @ ({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")

        # Store final points
        points_buffer = shared_data["points_buffer"]
        for i, point in enumerate(acquired_points[:3]):
            base_idx = i * 4
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']

        shared_data["points_count"].value = len(acquired_points)
        print(f"[3PT] Successfully acquired {len(acquired_points)} points.")
        return True

    except Exception as e:
        # It's good practice to print the exception to know what went wrong
        import traceback
        print(f"[3PT] An exception occurred during acquisition: {e}")
        traceback.print_exc()
        return False