"""
Enhanced 3-Point Acquisition Strategy for Drone Tracking Initialization

This module implements a sophisticated approach to acquire 3 high-quality points
for Kalman filter initialization, focusing on highest strength readings and
predicting drone movement direction.
"""

import numpy as np
import math
import time
from collections import deque


class ThreePointAcquisition:
    def __init__(self, shared_data, movement_queue, pi):
        self.shared_data = shared_data
        self.movement_queue = movement_queue
        self.pi = pi

        # Point storage with metadata
        self.candidate_points = []
        self.acquired_points = []

        # Search parameters
        self.min_strength_threshold = 8000  # Higher threshold for "good" points
        self.spiral_radius_start = 1.0  # degrees
        self.spiral_radius_max = 8.0  # degrees
        self.spiral_step = 0.3  # degrees per step
        self.scan_dwell_time = 0.05  # seconds per position

        # Movement prediction
        self.velocity_samples = deque(maxlen=5)
        self.last_detection_time = None

    def acquire_three_points(self):
        """
        Main acquisition sequence:
        1. Initial detection and strength optimization
        2. Spiral search with velocity estimation
        3. Predictive positioning for third point
        """
        print("[3PT] Starting 3-point acquisition sequence...")

        # Reset state
        self.candidate_points.clear()
        self.acquired_points.clear()
        self.shared_data["points_count"].value = 0

        try:
            # Phase 1: Find and optimize first point
            point1 = self.find_first_point()
            if not point1:
                print("[3PT] Failed to find first point")
                return False

            # Phase 2: Spiral search for second point with motion estimation
            point2 = self.find_second_point_with_spiral(point1)
            if not point2:
                print("[3PT] Failed to find second point")
                return False

            # Phase 3: Predictive positioning for third point
            point3 = self.find_third_point_predictive(point1, point2)
            if not point3:
                print("[3PT] Failed to find third point")
                return False

            # Store final points in shared memory
            self.store_final_points([point1, point2, point3])
            print(f"[3PT] Successfully acquired 3 points with strengths: "
                  f"{point1['strength']}, {point2['strength']}, {point3['strength']}")

            return True

        except Exception as e:
            print(f"[3PT] Error during acquisition: {e}")
            return False

    def find_first_point(self):
        """
        Find the first point by scanning around initial TLE prediction area
        and optimizing for highest strength reading.
        """
        print("[3PT] Phase 1: Finding first high-strength point...")

        # Start from TLE-predicted area (you can enhance this with your TLE data)
        start_az = self.shared_data["stepper_degrees"].value
        start_el = self.shared_data["servo_degrees"].value

        best_point = None
        search_radius = 5.0  # degrees

        # Concentric search around predicted area
        for radius in np.arange(0.5, search_radius, 0.5):
            for angle in np.arange(0, 360, 15):  # 24 positions per ring
                if self.shared_data['shutdown'].value:
                    return None

                # Calculate search position
                search_az = start_az + radius * math.cos(math.radians(angle))
                search_el = max(0, min(90, start_el + radius * math.sin(math.radians(angle))))

                # Move to position and measure
                self.track_target(search_az, search_el)
                time.sleep(self.scan_dwell_time)

                # Check for detection
                detection = self.check_current_detection()
                if detection and detection['strength'] > self.min_strength_threshold:
                    if not best_point or detection['strength'] > best_point['strength']:
                        best_point = detection.copy()
                        best_point['timestamp'] = time.time()
                        print(f"[3PT] New best first point: {best_point['strength']} @ "
                              f"({best_point['az']:.1f}°, {best_point['el']:.1f}°)")

        if best_point:
            # Optimize around best point with finer resolution
            best_point = self.optimize_point_locally(best_point)
            self.acquired_points.append(best_point)
            return best_point

        return None

    def find_second_point_with_spiral(self, point1):
        """
        Spiral search around first point to find second point and estimate velocity.
        """
        print("[3PT] Phase 2: Spiral search for second point...")

        start_time = time.time()
        center_az = point1['az']
        center_el = point1['el']

        spiral_detections = []

        # Archimedean spiral search
        angle = 0.0
        radius = self.spiral_radius_start

        while radius < self.spiral_radius_max and not self.shared_data['shutdown'].value:
            # Calculate spiral position
            spiral_az = center_az + radius * math.cos(math.radians(angle))
            spiral_el = max(0, min(90, center_el + radius * math.sin(math.radians(angle))))

            self.track_target(spiral_az, spiral_el)
            time.sleep(self.scan_dwell_time)

            # Check for detection
            detection = self.check_current_detection()
            if detection and detection['strength'] > self.min_strength_threshold:
                detection['timestamp'] = time.time()
                detection['spiral_angle'] = angle
                detection['spiral_radius'] = radius
                spiral_detections.append(detection)

                print(f"[3PT] Spiral detection: {detection['strength']} @ "
                      f"({detection['az']:.1f}°, {detection['el']:.1f}°)")

            # Update spiral parameters
            angle += 15.0  # degrees
            radius += self.spiral_step * (angle / 360.0)  # Expand as we spiral

        # Analyze detections to find best second point and estimate movement
        if len(spiral_detections) >= 2:
            point2 = self.select_best_second_point(spiral_detections, point1)
            if point2:
                self.acquired_points.append(point2)
                self.estimate_velocity_from_detections(spiral_detections)
                return point2

        return None

    def find_third_point_predictive(self, point1, point2):
        """
        Use velocity estimation to predict where drone will be and position for third point.
        """
        print("[3PT] Phase 3: Predictive positioning for third point...")

        # Calculate velocity vector from first two points
        dt = point2['timestamp'] - point1['timestamp']
        if dt <= 0:
            dt = 0.1  # fallback

        # Angular velocities (deg/s)
        vel_az = (point2['az'] - point1['az']) / dt
        vel_el = (point2['el'] - point1['el']) / dt

        # Handle azimuth wraparound
        if abs(vel_az) > 180:
            vel_az = vel_az - 360 * np.sign(vel_az)

        print(f"[3PT] Estimated velocity: {vel_az:.2f}°/s az, {vel_el:.2f}°/s el")

        # Predict future positions
        prediction_time = 0.5  # seconds ahead
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
            if self.shared_data['shutdown'].value:
                break

            self.track_target(pred_az, pred_el)
            time.sleep(self.scan_dwell_time * 2)  # Wait longer for prediction

            detection = self.check_current_detection()
            if detection and detection['strength'] > self.min_strength_threshold:
                detection['timestamp'] = time.time()
                if not best_point3 or detection['strength'] > best_point3['strength']:
                    best_point3 = detection.copy()
                    print(f"[3PT] Predictive detection: {best_point3['strength']} @ "
                          f"({best_point3['az']:.1f}°, {best_point3['el']:.1f}°)")

        if best_point3:
            # Fine-tune the third point
            best_point3 = self.optimize_point_locally(best_point3)
            self.acquired_points.append(best_point3)
            return best_point3

        return None

    def optimize_point_locally(self, rough_point, search_radius=1.0):
        """
        Fine-tune a point by searching in small increments around it.
        """
        best_point = rough_point.copy()
        center_az = rough_point['az']
        center_el = rough_point['el']

        # Fine grid search
        for daz in np.arange(-search_radius, search_radius + 0.1, 0.2):
            for del_el in np.arange(-search_radius, search_radius + 0.1, 0.2):
                if self.shared_data['shutdown'].value:
                    break

                test_az = center_az + daz
                test_el = max(0, min(90, center_el + del_el))

                self.track_target(test_az, test_el)
                time.sleep(0.03)  # Quick measurement

                detection = self.check_current_detection()
                if detection and detection['strength'] > best_point['strength']:
                    best_point = detection.copy()
                    best_point['timestamp'] = time.time()

        return best_point

    def select_best_second_point(self, detections, point1):
        """
        Select best second point considering strength and temporal/spatial separation.
        """
        if not detections:
            return None

        # Score each detection
        best_detection = None
        best_score = -1

        for detection in detections:
            # Distance from first point
            dist = math.sqrt((detection['az'] - point1['az']) ** 2 +
                             (detection['el'] - point1['el']) ** 2)

            # Time separation
            time_sep = abs(detection['timestamp'] - point1['timestamp'])

            # Scoring: strength + spatial separation bonus
            score = detection['strength']
            if dist > 2.0:  # Prefer points at least 2° apart
                score += 1000
            if time_sep > 0.2:  # Prefer temporally separated points
                score += 500

            if score > best_score:
                best_score = score
                best_detection = detection

        return best_detection

    def estimate_velocity_from_detections(self, detections):
        """
        Estimate velocity from multiple detection points.
        """
        if len(detections) < 2:
            return

        # Sort by timestamp
        detections.sort(key=lambda x: x['timestamp'])

        velocities = []
        for i in range(1, len(detections)):
            dt = detections[i]['timestamp'] - detections[i - 1]['timestamp']
            if dt > 0:
                vel_az = (detections[i]['az'] - detections[i - 1]['az']) / dt
                vel_el = (detections[i]['el'] - detections[i - 1]['el']) / dt
                velocities.append((vel_az, vel_el))

        if velocities:
            avg_vel_az = np.mean([v[0] for v in velocities])
            avg_vel_el = np.mean([v[1] for v in velocities])
            print(f"[3PT] Average velocity: {avg_vel_az:.2f}°/s az, {avg_vel_el:.2f}°/s el")

    def check_current_detection(self):
        """
        Check if current LiDAR reading indicates a valid drone detection.
        """
        # Read current lidar data
        with self.shared_data["lidar_data"].get_lock():
            distance_cm = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]

        # Get current mount position
        current_az = self.shared_data["stepper_degrees"].value
        current_el = self.shared_data["servo_degrees"].value

        # Check if within expected drone range (3-12m as specified)
        if not (300 <= distance_cm <= 1200):
            return None

        # Check if strength is reasonable
        if strength < 1000:  # Basic threshold
            return None

        return {
            'az': current_az,
            'el': current_el,
            'distance_m': distance_cm / 100.0,
            'strength': strength
        }

    def track_target(self, azimuth, elevation):
        """
        Move the pan-tilt system to target position.
        """
        from motors.motor_controller import track_target
        track_target(self.pi, azimuth, elevation, 0.0001, self.movement_queue, self.shared_data)

    def store_final_points(self, points):
        """
        Store the three acquired points in shared memory for EKF initialization.
        """
        points_buffer = self.shared_data["points_buffer"]

        for i, point in enumerate(points[:3]):
            base_idx = i * 4
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']

        self.shared_data["points_count"].value = len(points)
        print(f"[3PT] Stored {len(points)} points in shared memory")


# Integration function to replace the existing spiral_acquire_three
def enhanced_spiral_acquire_three(pi, shared_data, movement_queue):
    """
    Enhanced replacement for the existing spiral_acquire_three function.
    """
    print("\n--- STARTING ENHANCED 3-POINT ACQUISITION ---")

    # Create acquisition instance
    acquisition = ThreePointAcquisition(shared_data, movement_queue, pi)

    # Run the acquisition sequence
    success = acquisition.acquire_three_points()

    if success:
        print("--- 3-POINT ACQUISITION COMPLETED SUCCESSFULLY ---")
        return True
    else:
        print("--- 3-POINT ACQUISITION FAILED ---")
        return False