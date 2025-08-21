#!/usr/bin/env python3
"""
CircularDroneTracker: Real-time tracking of a drone in circular orbit
Uses predict-and-wait intercept strategy with continuous orbital refinement
Implements adaptive arc scanning to handle 2° LiDAR FOV constraint
"""

from enum import Enum
import numpy as np
import math
import time
from collections import deque
import traceback


class TrackerState(Enum):
    IDLE = 0
    SEARCHING = 1
    CONFIRMING_DIRECTION = 2
    CALCULATING_PLANE = 3
    TRACKING = 4
    LOST = 5


class CircularDroneTracker:
    """
    Tracks a drone moving in a circular orbit at 2m distance.
    Drone angular velocity: 18 deg/s (20s per orbit)
    LiDAR FOV: 2° - requires arc scanning for reliable detection
    """

    def __init__(self, shared_data, prediction_time_sec=0.5):
        self.shared_data = shared_data
        self.state = TrackerState.IDLE

        # Control flags from shared_data
        self.active = False

        # Drone parameters
        self.TARGET_DISTANCE_CM = 200.0  # 2 meters in cm
        self.DISTANCE_TOLERANCE_CM = 20.0  # ±20cm tolerance
        self.ANGULAR_VELOCITY_DEG = 18.0  # degrees per second
        self.MIN_STRENGTH = 500  # Minimum LiDAR strength threshold
        self.LIDAR_FOV_DEG = 2.0  # LiDAR field of view

        # Prediction parameters
        self.prediction_time = prediction_time_sec
        self.prediction_angle = self.ANGULAR_VELOCITY_DEG * prediction_time_sec

        # Arc scan parameters (adaptive)
        self.arc_radius_initial = 90.0  # Initial arc scan radius in degrees
        self.arc_radius_min = 1.5  # Minimum arc radius (slightly larger than FOV/2)
        self.arc_radius_current = self.arc_radius_initial
        self.arc_scan_points = 5  # Number of points in arc scan

        # Tracking state
        self.orbit_points = deque(maxlen=20)  # Rolling buffer of confirmed points
        self.last_confirmed_point = None  # (az, el, dist, time)
        self.last_confirmed_3d = None
        self.orbital_normal = None  # Normal vector to orbital plane
        self.orbital_center = np.array([0, 0, 0])  # Assume sensor at origin

        # Orbital refinement
        self.refinement_interval = 5  # Refine plane every N points
        self.points_since_refinement = 0
        self.orbital_confidence = 0.0  # 0-1 confidence in orbital model

        # Search parameters
        self.initial_heading = 0
        self.initial_inclination = -1
        self.heading_deviation = 0.0

        # Performance tracking
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.max_misses = 5

        print("[CircularDroneTracker] Initialized with {:.1f}° FOV, prediction: {:.2f}s".format(
            self.LIDAR_FOV_DEG, prediction_time_sec))

    def command_motors_to_target(self, azimuth, elevation):
        """Command motors to move to target position"""
        azimuth = float(azimuth % 360.0)  # Normalize azimuth
        elevation = float(np.clip(elevation, 0, 90))  # Clamp elevation

        try:
            self.shared_data["target_azimuth"].value = azimuth
            self.shared_data["target_elevation"].value = elevation
            self.shared_data["go_to_target"].value = True

            # Wait briefly for movement to start
            time.sleep(0.01)
        except Exception as e:
            print("[Tracker] Error commanding motors: {}".format(e))

    def wait_for_position(self, target_az, target_el, timeout=1.0):
        """Wait for motors to reach target position"""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            az_error = abs((current_az - target_az + 180) % 360 - 180)
            el_error = abs(current_el - target_el)

            if az_error < 0.5 and el_error < 0.5:
                return True

            # Check if target_reached flag is available and set
            target_reached = self.shared_data.get("target_reached")
            if target_reached and target_reached.value:
                return True

            time.sleep(0.002)
        return False

    def read_current_state(self):
        """Read current motor positions and LiDAR data"""
        current_az = self.shared_data["stepper_degrees"].value
        current_el = self.shared_data["servo_degrees"].value

        with self.shared_data["lidar_data"].get_lock():
            dist = float(self.shared_data["lidar_data"][0])
            strength = float(self.shared_data["lidar_data"][1])
            timestamp = float(self.shared_data["lidar_data"][2])

        return current_az, current_el, dist, strength, timestamp

    def is_valid_measurement(self, dist, strength):
        """Check if LiDAR measurement is valid drone detection"""
        min_dist = self.TARGET_DISTANCE_CM - self.DISTANCE_TOLERANCE_CM
        max_dist = self.TARGET_DISTANCE_CM + self.DISTANCE_TOLERANCE_CM
        return (min_dist < dist < max_dist) and (strength >= self.MIN_STRENGTH)

    def spherical_to_cartesian(self, az_deg, el_deg, dist_cm):
        """Convert spherical coordinates to 3D Cartesian"""
        az_rad = math.radians(az_deg)
        el_rad = math.radians(el_deg)
        r = dist_cm / 100.0  # Convert to meters

        x = r * math.cos(el_rad) * math.cos(az_rad)
        y = r * math.cos(el_rad) * math.sin(az_rad)
        z = r * math.sin(el_rad)

        # Always return as a properly shaped numpy array
        return np.array([x, y, z], dtype=np.float64)

    def cartesian_to_spherical(self, point_3d):
        """Convert 3D Cartesian to spherical coordinates"""
        # Ensure point is a numpy array
        point_3d = np.array(point_3d).flatten()

        if point_3d.shape[0] != 3:
            print("[Tracker] ERROR: Invalid 3D point shape in cartesian_to_spherical")
            return 0, 0, 0

        x, y, z = point_3d[0], point_3d[1], point_3d[2]
        r = np.linalg.norm(point_3d)

        if r < 0.01:  # Near origin
            return 0, 0, 0

        el_rad = math.asin(np.clip(z / r, -1, 1))
        az_rad = math.atan2(y, x)

        az_deg = math.degrees(az_rad) % 360.0
        el_deg = math.degrees(el_rad)
        dist_cm = r * 100.0

        return az_deg, el_deg, dist_cm

    def rotate_vector_rodrigues(self, v, k, theta_rad):
        """Rotate vector v around axis k by angle theta using Rodrigues' formula"""
        # Ensure vectors are numpy arrays with correct shape
        v = np.array(v).flatten()
        k = np.array(k).flatten()

        # Validate dimensions
        if v.shape[0] != 3 or k.shape[0] != 3:
            print("[Tracker] ERROR: Invalid vector dimensions - v:{}, k:{}".format(v.shape, k.shape))
            return v

        # Normalize axis vector
        k_norm = np.linalg.norm(k)
        if k_norm < 1e-10:
            print("[Tracker] ERROR: Zero-length rotation axis")
            return v

        k = k / k_norm

        # Rodrigues' rotation formula
        v_rot = (v * math.cos(theta_rad) +
                 np.cross(k, v) * math.sin(theta_rad) +
                 k * np.dot(k, v) * (1 - math.cos(theta_rad)))

        return v_rot

    def refine_orbital_plane(self):
        """Refine orbital plane estimation using accumulated points"""
        if len(self.orbit_points) < 3:
            return

        # Convert recent points to 3D
        points_3d = []
        for pt in self.orbit_points:
            p3d = self.spherical_to_cartesian(pt[0], pt[1], pt[2])
            points_3d.append(p3d)

        points_3d = np.array(points_3d)

        # Validate shape
        if points_3d.shape[1] != 3:
            print("[Tracker] ERROR: Invalid 3D points shape: {}".format(points_3d.shape))
            return

        # Use SVD to find best-fit plane through origin
        # The points should lie on a circle, so we find the plane that minimizes
        # the sum of squared distances
        try:
            U, s, Vt = np.linalg.svd(points_3d.T, full_matrices=False)

            # The normal is the singular vector with smallest singular value
            new_normal = Vt[-1, :]

            # Ensure it's a proper 3D vector
            new_normal = np.array(new_normal).flatten()
            if new_normal.shape[0] != 3:
                print("[Tracker] ERROR: SVD produced invalid normal vector")
                return

            # Ensure consistency of normal direction
            if self.orbital_normal is not None:
                if np.dot(new_normal, self.orbital_normal) < 0:
                    new_normal = -new_normal

            # Smooth update using weighted average
            if self.orbital_normal is not None:
                alpha = 0.3  # Smoothing factor
                self.orbital_normal = (1 - alpha) * self.orbital_normal + alpha * new_normal
                self.orbital_normal = self.orbital_normal / np.linalg.norm(self.orbital_normal)
            else:
                self.orbital_normal = new_normal / np.linalg.norm(new_normal)

            # Ensure orbital_normal is properly shaped
            self.orbital_normal = np.array(self.orbital_normal).flatten()

            # Update confidence based on fit quality
            # Use the ratio of singular values as a measure of planarity
            if s[2] > 0:
                planarity = 1.0 - (s[2] / s[0])  # Close to 1 means good plane fit
                self.orbital_confidence = 0.7 * self.orbital_confidence + 0.3 * planarity

            # Update arc scan radius based on confidence
            self.arc_radius_current = self.arc_radius_initial * (1 - self.orbital_confidence * 0.7)
            self.arc_radius_current = max(self.arc_radius_current, self.arc_radius_min)

            inclination_deg = math.degrees(math.acos(np.clip(abs(self.orbital_normal[2]), 0, 1)))
            print("[Tracker] Plane refined - Inclination: {:.1f}°, Confidence: {:.2f}, Arc: {:.1f}°".format(
                inclination_deg, self.orbital_confidence, self.arc_radius_current))

        except Exception as e:
            print("[Tracker] ERROR in plane refinement: {}".format(e))
            import traceback
            traceback.print_exc()

    def perform_arc_scan(self, center_az, center_el, radius_deg):
        """Perform arc scan around predicted position to handle FOV limitation"""
        best_measurement = None
        best_strength = 0

        # Generate arc points
        for i in range(self.arc_scan_points):
            if not self.active or self.shared_data["shutdown"].value:
                break

            # Create arc pattern
            if i == 0:
                # Center point
                scan_az = center_az
                scan_el = center_el
            else:
                # Arc points
                angle = (2 * math.pi * (i - 1)) / (self.arc_scan_points - 1)
                scan_az = (center_az + radius_deg * math.cos(angle)) % 360.0
                scan_el = np.clip(center_el + radius_deg * math.sin(angle) * 0.5, 0, 90)

            # Move to scan point
            self.command_motors_to_target(scan_az, scan_el)

            # Wait for position with short timeout
            if not self.wait_for_position(scan_az, scan_el, timeout=0.1):
                continue

            # Take multiple readings for reliability
            for _ in range(3):
                current_az, current_el, dist, strength, timestamp = self.read_current_state()

                if self.is_valid_measurement(dist, strength):
                    if strength > best_strength:
                        best_measurement = (current_az, current_el, dist, strength, timestamp)
                        best_strength = strength

                time.sleep(0.002)  # 500Hz polling

        return best_measurement

    def start_tracking(self, initial_heading=0, heading_deviation=00.0, initial_inclination=-1):
        """Initialize search parameters and start tracking"""
        self.initial_heading = initial_heading
        self.heading_deviation = heading_deviation
        self.initial_inclination = initial_inclination
        self.state = TrackerState.SEARCHING
        self.active = True

        # Reset tracking state
        self.orbit_points.clear()
        self.orbital_normal = None
        self.orbital_confidence = 0.0
        self.arc_radius_current = self.arc_radius_initial

        print("[Tracker] Starting - Heading: {}, Inclination: {}".format(
            initial_heading if initial_heading != -1 else "UNKNOWN",
            initial_inclination if initial_inclination != -1 else "UNKNOWN"))

    def stop_tracking(self):
        """Stop tracking"""
        self.active = False
        self.state = TrackerState.IDLE
        print("[Tracker] Stopped")

    def state_searching(self):
        """Search for first drone detection with arc scanning"""
        # Determine search center
        if self.initial_heading != -1:
            # Search around known heading
            center_az = self.initial_heading
            center_el = 45 if self.initial_inclination == -1 else self.initial_inclination
        else:
            # Sweep search
            sweep_time = time.time() % 10.0
            center_az = (sweep_time * 36.0) % 360.0
            center_el = 30.0

        # Perform arc scan
        measurement = self.perform_arc_scan(center_az, center_el, self.heading_deviation / 2)

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))
            print("[Tracker] First point found at ({:.1f}°, {:.1f}°) dist={:.0f}cm".format(
                az, el, dist))
            self.state = TrackerState.CONFIRMING_DIRECTION

    def state_confirming_direction(self):
        """Find second point to determine direction"""
        if len(self.orbit_points) == 0:
            self.state = TrackerState.SEARCHING
            return

        first_point = self.orbit_points[-1]

        # Wait briefly for drone to move
        time_elapsed = time.time() - first_point[3]
        if time_elapsed < 0.3:
            time.sleep(0.3 - time_elapsed)

        # Predict next position (drone moves right)
        predicted_az = (first_point[0] + self.ANGULAR_VELOCITY_DEG * 0.5) % 360.0
        predicted_el = first_point[1]

        # Arc scan with larger radius for second point
        measurement = self.perform_arc_scan(predicted_az, predicted_el, 8.0)

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))
            print("[Tracker] Second point found at ({:.1f}°, {:.1f}°)".format(az, el))
            self.state = TrackerState.CALCULATING_PLANE

    def state_calculating_plane(self):
        """Calculate initial orbital plane from points"""
        if len(self.orbit_points) < 2:
            self.state = TrackerState.SEARCHING
            return

        # Get more points for better plane estimation
        if len(self.orbit_points) < 4:
            # Collect more points
            last_point = self.orbit_points[-1]
            predicted_az = (last_point[0] + self.ANGULAR_VELOCITY_DEG * 0.3) % 360.0
            predicted_el = last_point[1]

            measurement = self.perform_arc_scan(predicted_az, predicted_el, 6.0)

            if measurement:
                az, el, dist, strength, timestamp = measurement
                self.orbit_points.append((az, el, dist, time.time()))
                print("[Tracker] Point {} collected at ({:.1f}°, {:.1f}°)".format(
                    len(self.orbit_points), az, el))
            return

        # Calculate orbital plane
        self.refine_orbital_plane()

        if self.orbital_normal is not None:
            # Initialize tracking state
            last_point = self.orbit_points[-1]
            self.last_confirmed_point = last_point

            # Ensure last_confirmed_3d is properly set as a numpy array
            self.last_confirmed_3d = np.array(self.spherical_to_cartesian(
                last_point[0], last_point[1], last_point[2])).flatten()

            # Validate dimensions
            if self.last_confirmed_3d.shape[0] != 3:
                print("[Tracker] ERROR: Invalid 3D position shape")
                self.state = TrackerState.SEARCHING
                return

            print("[Tracker] Entering tracking mode")
            self.state = TrackerState.TRACKING
        else:
            print("[Tracker] Failed to calculate plane, restarting")
            self.state = TrackerState.SEARCHING

    def state_tracking(self):
        """Main tracking loop with predict-and-intercept using arc scanning"""
        if self.last_confirmed_3d is None or self.orbital_normal is None:
            self.state = TrackerState.SEARCHING
            return

        # Validate vector dimensions before rotation
        if not isinstance(self.last_confirmed_3d, np.ndarray):
            self.last_confirmed_3d = np.array(self.last_confirmed_3d).flatten()
        if not isinstance(self.orbital_normal, np.ndarray):
            self.orbital_normal = np.array(self.orbital_normal).flatten()

        if self.last_confirmed_3d.shape[0] != 3 or self.orbital_normal.shape[0] != 3:
            print("[Tracker] ERROR: Invalid vector dimensions in tracking")
            self.state = TrackerState.SEARCHING
            return

        # PREDICT: Calculate intercept point
        rotation_angle_rad = math.radians(self.prediction_angle)
        predicted_3d = self.rotate_vector_rodrigues(
            self.last_confirmed_3d,
            self.orbital_normal,
            rotation_angle_rad
        )

        # Convert to spherical coordinates
        pred_az, pred_el, pred_dist = self.cartesian_to_spherical(predicted_3d)

        # INTERCEPT: Perform arc scan around predicted position
        measurement = self.perform_arc_scan(pred_az, pred_el, self.arc_radius_current)

        if measurement:
            # Success - update tracking
            az, el, dist, strength, timestamp = measurement

            # Add to orbit points for continuous refinement
            self.orbit_points.append((az, el, dist, time.time()))
            self.last_confirmed_point = (az, el, dist, time.time())

            # Ensure last_confirmed_3d is properly updated as numpy array
            self.last_confirmed_3d = np.array(self.spherical_to_cartesian(az, el, dist)).flatten()

            self.consecutive_hits += 1
            self.consecutive_misses = 0
            self.points_since_refinement += 1

            # Refine orbital plane periodically
            if self.points_since_refinement >= self.refinement_interval:
                self.refine_orbital_plane()
                self.points_since_refinement = 0

            # Update confidence
            self.orbital_confidence = min(1.0, self.orbital_confidence + 0.05)

            print("[Tracker] Hit #{} at ({:.1f}°, {:.1f}°) - Arc: {:.1f}°, Conf: {:.2f}".format(
                self.consecutive_hits, az, el, self.arc_radius_current, self.orbital_confidence))

            # Update shared satellite points for visualization
            try:
                with self.shared_data["satellite_points"].get_lock():
                    self.shared_data["satellite_points"][0] = az
                    self.shared_data["satellite_points"][1] = el
                    self.shared_data["satellite_points"][2] = dist
                    self.shared_data["satellite_points"][3] = strength
                    self.shared_data["satellite_points"][4] = timestamp
            except:
                pass

            # Record to tracking history if enabled
            if self.shared_data.get("record_tle_points") and self.shared_data["record_tle_points"].value:
                try:
                    self.shared_data["tracking_history"].append([az, el, dist, strength, timestamp])
                except:
                    pass
        else:
            # Miss - expand search
            self.consecutive_misses += 1
            self.consecutive_hits = 0
            self.orbital_confidence = max(0.0, self.orbital_confidence - 0.1)

            # Increase arc radius for next attempt
            self.arc_radius_current = min(
                self.arc_radius_current * 1.2,
                self.arc_radius_initial * 2
            )

            print("[Tracker] Miss #{} - Expanding arc to {:.1f}°".format(
                self.consecutive_misses, self.arc_radius_current))

            if self.consecutive_misses >= self.max_misses:
                print("[Tracker] Lost tracking, entering recovery")
                self.state = TrackerState.LOST

    def state_lost(self):
        """Try to reacquire target with expanding search"""
        if self.last_confirmed_point is None:
            self.state = TrackerState.SEARCHING
            return

        # Predict where drone should be now based on last known position
        time_elapsed = time.time() - self.last_confirmed_point[3]
        predicted_angle = self.ANGULAR_VELOCITY_DEG * time_elapsed

        # Search in expanding spiral
        search_radius = 10.0 + 5.0 * (self.consecutive_misses - self.max_misses)

        predicted_az = (self.last_confirmed_point[0] + predicted_angle) % 360.0
        predicted_el = self.last_confirmed_point[1]

        measurement = self.perform_arc_scan(predicted_az, predicted_el, search_radius)

        if measurement:
            print("[Tracker] Reacquired target!")
            az, el, dist, strength, timestamp = measurement

            # Reset tracking
            self.orbit_points.append((az, el, dist, time.time()))
            self.last_confirmed_point = (az, el, dist, time.time())

            # Ensure last_confirmed_3d is properly set as numpy array
            self.last_confirmed_3d = np.array(self.spherical_to_cartesian(az, el, dist)).flatten()

            self.consecutive_misses = 0
            self.arc_radius_current = self.arc_radius_initial
            self.state = TrackerState.TRACKING
        elif self.consecutive_misses > self.max_misses + 10:
            print("[Tracker] Reacquisition failed, restarting search")
            self.state = TrackerState.SEARCHING
            self.consecutive_misses = 0

    def update(self):
        """Main update loop - call this repeatedly"""
        # Check control flags
        if self.shared_data["shutdown"].value:
            return False

        # Check start/stop control
        tracker_active = self.shared_data.get("circular_tracker_active")
        if tracker_active and tracker_active.value:
            if not self.active:
                # Start tracking - safely get values from shared_data
                try:
                    heading = self.shared_data.get("heading")
                    if heading is not None:
                        heading = heading.value if hasattr(heading, 'value') else float(heading)
                    else:
                        heading = -1

                    inclination = self.shared_data.get("inclination")
                    if inclination is not None:
                        inclination = inclination.value if hasattr(inclination, 'value') else float(inclination)
                    else:
                        inclination = -1

                    self.start_tracking(heading, 30.0, inclination)
                except Exception as e:
                    print("[Tracker] Error reading initial parameters: {}".format(e))
                    self.start_tracking()
        else:
            if self.active:
                self.stop_tracking()
            time.sleep(0.1)
            return True

        if not self.active:
            time.sleep(0.1)
            return True

        # Execute state machine
        try:
            if self.state == TrackerState.IDLE:
                time.sleep(0.1)
            elif self.state == TrackerState.SEARCHING:
                self.state_searching()
            elif self.state == TrackerState.CONFIRMING_DIRECTION:
                self.state_confirming_direction()
            elif self.state == TrackerState.CALCULATING_PLANE:
                self.state_calculating_plane()
            elif self.state == TrackerState.TRACKING:
                self.state_tracking()
            elif self.state == TrackerState.LOST:
                self.state_lost()
        except Exception as e:
            print("[Tracker] Error in state {}: {}".format(self.state, e))
            traceback.print_exc()
            self.state = TrackerState.SEARCHING

        return True

    def run(self):
        """Continuous tracking loop"""
        print("[CircularDroneTracker] Started - waiting for activation")

        while self.update():
            # Small delay to prevent CPU overload
            time.sleep(0.001)

        print("[CircularDroneTracker] Stopped")


# Integration function for your system
def run_circular_tracker(shared_data):
    """Run the circular drone tracker as a separate process"""
    try:
        # Ensure control flag exists
        if "circular_tracker_active" not in shared_data:
            from multiprocessing import Value
            shared_data["circular_tracker_active"] = Value('b', False)

        tracker = CircularDroneTracker(shared_data, prediction_time_sec=0.5)
        tracker.run()
    except Exception as e:
        print("[CircularTracker Process] Fatal error: {}".format(e))
        traceback.print_exc()


# Example of how to control the tracker from your main program
def start_circular_tracking(shared_data, heading=-1, inclination=-1):
    """Start the circular tracker with given parameters"""
    try:
        if "heading" in shared_data:
            shared_data["heading"].value = float(heading)
        if "inclination" in shared_data:
            shared_data["inclination"].value = float(inclination)

        shared_data["circular_tracker_active"].value = True
        print("[Main] Circular tracking activated")
    except Exception as e:
        print("[Main] Error starting circular tracking: {}".format(e))


def stop_circular_tracking(shared_data):
    """Stop the circular tracker"""
    try:
        shared_data["circular_tracker_active"].value = False
        print("[Main] Circular tracking deactivated")
    except Exception as e:
        print("[Main] Error stopping circular tracking: {}".format(e))