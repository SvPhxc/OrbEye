"""
Enhanced 3-Point Acquisition Strategy for Drone Tracking Initialization

This module enhances the existing spiral_acquire_three function without changing names.
"""

import numpy as np
import math
import time
from collections import deque

def enhanced_spiral_acquire_three(pi, shared_data, movement_queue):
    """
    Enhanced version that replaces the existing spiral_acquire_three function.
    Keeps the same function signature but implements the 3-phase strategy:
    1. Find first point with highest strength
    2. Spiral search for second point with velocity estimation
    3. Predictive positioning for third point
    """
    print("\n--- STARTING ENHANCED 3-POINT ACQUISITION ---")

    # Import your existing functions
    from motors.motor_controller import track_target

    # Reset acquisition state
    shared_data["points_count"].value = 0

    # Storage for candidate points
    candidate_points = []
    acquired_points = []

    # Search parameters
    min_strength_threshold = 8000
    spiral_radius_start = 1.0
    spiral_radius_max = 8.0
    spiral_step = 0.3
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

                # Check current detection
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

        # Optimize first point locally
        best_point = optimize_point_locally(pi, shared_data, movement_queue, best_point)
        acquired_points.append(best_point)

        # Phase 2: Spiral search for second point
        print("[3PT] Phase 2: Spiral search for second point...")

        center_az = best_point['az']
        center_el = best_point['el']
        spiral_detections = []

        angle = 0.0
        radius = spiral_radius_start

        while radius < spiral_radius_max and not shared_data['shutdown'].value:
            spiral_az = center_az + radius * math.cos(math.radians(angle))
            spiral_el = max(0, min(90, center_el + radius * math.sin(math.radians(angle))))

            track_target(pi, spiral_az, spiral_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                detection['timestamp'] = time.time()
                detection['spiral_angle'] = angle
                detection['spiral_radius'] = radius
                spiral_detections.append(detection)

                print(f"[3PT] Spiral detection: {detection['strength']} @ "
                      f"({detection['az']:.1f}°, {detection['el']:.1f}°)")

            angle += 15.0
            radius += spiral_step * (angle / 360.0)

        # Select best second point
        if len(spiral_detections) >= 1:
            point2 = select_best_second_point(spiral_detections, best_point)
            if point2:
                acquired_points.append(point2)
            else:
                print("[3PT] Failed to find suitable second point")
                return False
        else:
            print("[3PT] No detections in spiral search")
            return False

        # Phase 3: Predictive positioning for third point
        print("[3PT] Phase 3: Predictive positioning for third point...")

        # Calculate velocity from first two points
        dt = point2['timestamp'] - best_point['timestamp']
        if dt <= 0:
            dt = 0.1

        vel_az = (point2['az'] - best_point['az']) / dt
        vel_el = (point2['el'] - best_point['el']) / dt

        # Handle azimuth wraparound
        if abs(vel_az) > 180:
            vel_az = vel_az - 360 * np.sign(vel_az)

        print(f"[3PT] Estimated velocity: {vel_az:.2f}°/s az, {vel_el:.2f}°/s el")

        # Predict future positions
        prediction_time = 0.5
        predicted_az = point2['az'] + vel_az * prediction_time
        predicted_el = max(0, min(90, point2['el'] + vel_el * prediction_time))

        # Search around predicted position
        search_positions = [
            (predicted_az, predicted_el),
            (predicted_az + vel_az * 0.3, predicted_el + vel_el * 0.3),
            (predicted_az + vel_az * 0.7, predicted_el + vel_el * 0.7),
            (predicted_az - 1.0, predicted_el),
            (predicted_az + 1.0, predicted_el),
            (predicted_az, max(0, predicted_el - 1.0)),
            (predicted_az, min(90, predicted_el + 1.0)),
        ]

        best_point3 = None

        for pred_az, pred_el in search_positions:
            if shared_data['shutdown'].value:
                break

            track_target(pi, pred_az, pred_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time * 2)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                detection['timestamp'] = time.time()
                if not best_point3 or detection['strength'] > best_point3['strength']:
                    best_point3 = detection.copy()
                    print(f"[3PT] Predictive detection: {best_point3['strength']} @ "
                          f"({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")

        if best_point3:
            best_point3 = optimize_point_locally(pi, shared_data, movement_queue, best_point3)
            acquired_points.append(best_point3)
        else:
            print("[3PT] Failed to find third point")
            return False

        # Store final points in shared memory using your existing format
        points_buffer = shared_data["points_buffer"]

        for i, point in enumerate(acquired_points[:3]):
            base_idx = i * 4
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']

        shared_data["points_count"].value = len(acquired_points)

        print(f"[3PT] Successfully acquired {len(acquired_points)} points with strengths: " +
              ", ".join([str(int(p['strength'])) for p in acquired_points]))

        return True

    except Exception as e:
        print(f"[3PT] Error during acquisition: {e}")
        return False

def check_current_detection(shared_data):
    """Check if current LiDAR reading indicates a valid detection"""
    with shared_data["lidar_data"].get_lock():
        distance_cm = shared_data["lidar_data"][0]
        strength = shared_data["lidar_data"][1]

    current_az = shared_data["stepper_degrees"].value
    current_el = shared_data["servo_degrees"].value

    # Check if within expected drone range
    if not (300 <= distance_cm <= 1200):
        return None

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
            time.sleep(0.03)

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
        dist = math.sqrt((detection['az'] - point1['az'])**2 +
                        (detection['el'] - point1['el'])**2)

        time_sep = abs(detection['timestamp'] - point1['timestamp'])

        score = detection['strength']
        if dist > 2.0:
            score += 1000
        if time_sep > 0.2:
            score += 500

        if score > best_score:
            best_score = score
            best_detection = detection

    return best_detection