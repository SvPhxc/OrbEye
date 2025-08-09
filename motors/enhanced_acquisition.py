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
        shared_data["points_count"].value = 1
        print(f"[3PT] Acquired Point 1: {best_point['strength']} @ ({best_point['az']:.1f}°, {best_point['el']:.1f}°)")


        # Phase 2: Spiral search for second point
        print("[3PT] Phase 2: Spiral search for second point...")

        center_az = best_point['az']
        center_el = best_point['el']
        spiral_detections = []

        angle = 0.0
        radius = spiral_radius_start

        while radius < spiral_radius_max and not shared_data['shutdown'].value and len(spiral_detections) < 5:
            spiral_az = center_az + radius * math.cos(math.radians(angle))
            spiral_el = max(0, min(90, center_el + radius * math.sin(math.radians(angle))))

            track_target(pi, spiral_az, spiral_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                # Ensure point is distinct from the first one
                dist_from_p1 = math.sqrt((detection['az'] - best_point['az'])**2 + (detection['el'] - best_point['el'])**2)
                if dist_from_p1 > 1.0: # Only consider points at least 1 degree away
                    detection['timestamp'] = time.time()
                    spiral_detections.append(detection)
                    print(f"[3PT] Spiral candidate: {detection['strength']} @ "
                          f"({detection['az']:.1f}°, {detection['el']:.1f}°)")

            angle += 15.0
            radius += spiral_step * (angle / 360.0)

        # Select best second point
        point2 = None
        if len(spiral_detections) >= 1:
            point2 = select_best_second_point(spiral_detections, best_point)
            if point2:
                acquired_points.append(point2)
                shared_data["points_count"].value = 2
                print(f"[3PT] Acquired Point 2: {point2['strength']} @ ({point2['az']:.1f}°, {point2['el']:.1f}°)")

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
        if dt <= 1e-3: # Avoid division by zero or near-zero
            dt = 0.1

        vel_az = (point2['az'] - best_point['az']) / dt
        vel_el = (point2['el'] - best_point['el']) / dt

        # Handle azimuth wraparound
        if abs(vel_az) > 180:
            vel_az = vel_az - 360 * np.sign(vel_az)
            
        # --- NEW: Clip the velocity to a reasonable range to prevent outliers ---
        MAX_VEL_DEG_S = 100.0 # Sanity limit: 100 degrees per second
        vel_az = np.clip(vel_az, -MAX_VEL_DEG_S, MAX_VEL_DEG_S)
        vel_el = np.clip(vel_el, -MAX_VEL_DEG_S, MAX_VEL_DEG_S)

        print(f"[3PT] Estimated (clipped) velocity: {vel_az:.2f}°/s az, {vel_el:.2f}°/s el")

        # Predict future position
        prediction_time = 0.5
        predicted_az = point2['az'] + vel_az * prediction_time
        predicted_el = max(0, min(90, point2['el'] + vel_el * prediction_time))

        print(f"[3PT] Predicted position: ({predicted_az:.1f}°, {predicted_el:.1f}°)")

        # --- NEW: Use a robust, fixed-size box search instead of a velocity-scaled pattern ---
        search_positions = []
        search_box_radius = 4.0  # Search an 8x8 degree box
        search_step = 2.0        # With a 2 degree step size

        # Start at the predicted point
        search_positions.append((predicted_az, predicted_el))
        
        # Create a grid around the predicted point
        for r in np.arange(search_step, search_box_radius + 1e-6, search_step):
            for angle_deg in np.arange(0, 360, 45):
                d_az = r * math.cos(math.radians(angle_deg))
                d_el = r * math.sin(math.radians(angle_deg))
                search_positions.append((predicted_az + d_az, predicted_el + d_el))

        best_point3 = None
        for i, (pred_az, pred_el) in enumerate(search_positions):
            if shared_data['shutdown'].value:
                break
            
            print(f"[3PT] Searching for P3 at ({pred_az:.1f}, {pred_el:.1f}) [Step {i+1}/{len(search_positions)}]")
            track_target(pi, pred_az, pred_el, 0.0001, movement_queue, shared_data)
            time.sleep(scan_dwell_time * 2)

            detection = check_current_detection(shared_data)
            if detection and detection['strength'] > min_strength_threshold:
                detection['timestamp'] = time.time()
                if not best_point3 or detection['strength'] > best_point3['strength']:
                    best_point3 = detection.copy()
                    print(f"[3PT] Predictive candidate: {best_point3['strength']} @ "
                          f"({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")
                    # Optimization: If we find a strong signal, we can stop searching
                    if best_point3['strength'] > min_strength_threshold * 1.5:
                        break 

        if best_point3:
            best_point3 = optimize_point_locally(pi, shared_data, movement_queue, best_point3)
            acquired_points.append(best_point3)
            print(f"[3PT] Acquired Point 3: {best_point3['strength']} @ ({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")
        else:
            print("[3PT] Failed to find third point with predictive search")
            return False

        # Store final points in shared memory
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