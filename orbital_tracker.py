#!/usr/bin/env python3
"""
CircularDroneTracker: Real-time tracking of a drone in circular orbit
Uses predict-and-wait intercept strategy with continuous orbital refinement
Implements adaptive arc scanning to handle 2° LiDAR FOV constraint

Key Features:
- Clutter filtering to reject background objects
- Wide hemisphere search when inclination is unknown
- Adaptive arc scanning that shrinks as confidence increases
- Continuous orbital plane refinement using SVD
- Fallback for systems without scipy
"""

from enum import Enum
import numpy as np
import math
import time
from collections import deque
import traceback

# Try to import scipy for KD-tree, fall back to simple implementation if not available
try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[CircularDroneTracker] scipy not available, using simple clutter filter")


class ClutterFilter:
    """
    Clutter filter for background rejection with fallback for systems without scipy.
    """

    def __init__(self, background_file="background_scan.npy"):
        self.angular_tolerance = 1.0  # degrees
        self.distance_margin_cm = 50.0  # cm closer than background
        self.background_tree = None
        self.background_data = None
        self._cache = {}

        try:
            self.background_data = np.load(background_file)
            print("[ClutterFilter] Loaded {} background points.".format(len(self.background_data)))

            if HAS_SCIPY:
                # Build KD-tree for fast lookups
                coords = self.background_data[:, [0, 1]]  # az, el columns
                self.background_tree = cKDTree(coords, leafsize=16)
                self.bg_distances = self.background_data[:, 2]  # distance column
            else:
                # Simple implementation without scipy
                self.bg_azimuths = self.background_data[:, 0]
                self.bg_elevations = self.background_data[:, 1]
                self.bg_distances = self.background_data[:, 2]

        except FileNotFoundError:
            print("[ClutterFilter] WARNING: Background file '{}' not found.".format(background_file))
        except Exception as e:
            print("[ClutterFilter] ERROR: {}".format(e))

    def is_foreground(self, azimuth, elevation, distance, strength):
        """Check if measurement is foreground (not background clutter)"""
        if self.background_data is None:
            return True  # No background data, accept all

        # Check cache first
        cache_key = (int(azimuth * 10), int(elevation * 10))
        if cache_key in self._cache:
            bg_dist = self._cache[cache_key]
        else:
            if HAS_SCIPY and self.background_tree is not None:
                # Fast KD-tree query
                query_point = np.array([azimuth, elevation])
                try:
                    angular_dist, idx = self.background_tree.query(query_point, k=1)

                    if angular_dist < self.angular_tolerance:
                        bg_dist = self.bg_distances[idx]
                    else:
                        bg_dist = float('inf')  # No background at this angle

                except Exception as e:
                    print("[ClutterFilter] Query error: {}".format(e))
                    return True
            else:
                # Simple brute-force search
                az_diff = np.abs(self.bg_azimuths - azimuth)
                # Handle wraparound at 360 degrees
                az_diff = np.minimum(az_diff, 360 - az_diff)
                el_diff = np.abs(self.bg_elevations - elevation)

                # Combined angular distance
                angular_dist = np.sqrt(az_diff ** 2 + el_diff ** 2)
                min_idx = np.argmin(angular_dist)

                if angular_dist[min_idx] < self.angular_tolerance:
                    bg_dist = self.bg_distances[min_idx]
                else:
                    bg_dist = float('inf')

            # Cache result
            if len(self._cache) < 10000:
                self._cache[cache_key] = bg_dist

        # Object is foreground if significantly closer than background
        return distance < (bg_dist - self.distance_margin_cm)


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

    def __init__(self, shared_data, prediction_time_sec=0.5, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.state = TrackerState.IDLE

        # Control flags from shared_data
        self.active = False

        # Initialize clutter filter for background rejection
        self.clutter_filter = ClutterFilter(background_file)

        # Drone parameters
        self.TARGET_DISTANCE_CM = 200.0  # 2 meters in cm
        self.DISTANCE_TOLERANCE_CM = 20.0  # ±20cm tolerance
        self.ANGULAR_VELOCITY_DEG = 18.0  # degrees per second
        self.MIN_STRENGTH = 100  # Minimum LiDAR strength threshold
        self.LIDAR_FOV_DEG = 2.0  # LiDAR field of view

        # Prediction parameters
        self.prediction_time = prediction_time_sec
        self.prediction_angle = self.ANGULAR_VELOCITY_DEG * prediction_time_sec

        # Arc scan parameters (adaptive)
        self.arc_radius_initial = 5.0  # Initial arc scan radius in degrees
        self.arc_radius_min = 1.5  # Minimum arc radius (slightly larger than FOV/2)
        self.arc_radius_current = self.arc_radius_initial
        self.arc_scan_points = 5  # Default number of points in arc scan
        self.arc_scan_points_wide = 9  # More points for wide searches

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
        self.initial_heading = -1
        self.initial_inclination = -1
        self.heading_deviation = 30.0

        # Performance tracking
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.max_misses = 5

        print("[CircularDroneTracker] Initialized with {:.1f}° FOV, prediction: {:.2f}s".format(
            self.LIDAR_FOV_DEG, prediction_time_sec))
        print("[CircularDroneTracker] Clutter filter: {}".format(
            "Enabled" if self.clutter_filter.background_data is not None else "Disabled"))

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

    def is_valid_measurement(self, dist, strength, az=None, el=None):
        """Check if LiDAR measurement is valid drone detection"""
        # Check distance and strength
        min_dist = self.TARGET_DISTANCE_CM - self.DISTANCE_TOLERANCE_CM
        max_dist = self.TARGET_DISTANCE_CM + self.DISTANCE_TOLERANCE_CM

        if not (min_dist < dist < max_dist):
            return False

        if strength < self.MIN_STRENGTH:
            return False

        # Check against background clutter if position is provided
        if az is not None and el is not None and self.clutter_filter:
            if not self.clutter_filter.is_foreground(az, el, dist, strength):
                return False

        return True

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

    def perform_arc_scan(self, center_az, center_el, radius_deg, radius_el_deg=None, num_points=None):
        """
        Perform arc scan around predicted position to handle FOV limitation.
        radius_deg: azimuth search radius
        radius_el_deg: elevation search radius (if None, uses radius_deg * 0.5)
        num_points: number of scan points (if None, uses self.arc_scan_points)
        """
        best_measurement = None
        best_strength = 0

        if radius_el_deg is None:
            radius_el_deg = radius_deg * 0.5

        if num_points is None:
            num_points = self.arc_scan_points

        # Generate arc points
        for i in range(num_points):
            if not self.active or self.shared_data["shutdown"].value:
                break

            # Create arc pattern
            if i == 0:
                # Center point
                scan_az = center_az
                scan_el = center_el
            else:
                # Arc points distributed around circle
                angle = (2 * math.pi * (i - 1)) / (num_points - 1)
                scan_az = (center_az + radius_deg * math.cos(angle)) % 360.0
                scan_el = np.clip(center_el + radius_el_deg * math.sin(angle), 0, 90)

            # Move to scan point
            self.command_motors_to_target(scan_az, scan_el)

            # Wait for position with short timeout
            if not self.wait_for_position(scan_az, scan_el, timeout=0.1):
                continue

            # Take multiple readings for reliability
            for _ in range(3):
                current_az, current_el, dist, strength, timestamp = self.read_current_state()

                # Check if valid with clutter filtering
                if self.is_valid_measurement(dist, strength, current_az, current_el):
                    if strength > best_strength:
                        best_measurement = (current_az, current_el, dist, strength, timestamp)
                        best_strength = strength

                time.sleep(0.002)  # 500Hz polling

        return best_measurement

    def start_tracking(self, initial_heading=-1, heading_deviation=30.0, initial_inclination=-1):
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
            # Search around known heading with normal scan points
            center_az = self.initial_heading
            center_el = 45 if self.initial_inclination == -1 else self.initial_inclination
            num_points = self.arc_scan_points
        else:
            # Sweep search - use more points for wider coverage
            sweep_time = time.time() % 10.0
            center_az = (sweep_time * 36.0) % 360.0
            center_el = 30.0
            num_points = self.arc_scan_points_wide

        # Perform arc scan with both azimuth and elevation radius
        measurement = self.perform_arc_scan(center_az, center_el,
                                            self.heading_deviation / 2,
                                            self.heading_deviation / 3,
                                            num_points)

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))
            print("[Tracker] First point found at ({:.1f}°, {:.1f}°) dist={:.0f}cm str={:.0f}".format(
                az, el, dist, strength))
            self.state = TrackerState.CONFIRMING_DIRECTION

    def state_confirming_direction(self):
        """Find second point to determine direction and start defining orbital plane"""
        if len(self.orbit_points) == 0:
            self.state = TrackerState.SEARCHING
            return

        first_point = self.orbit_points[-1]

        # Wait for drone to move (0.5 seconds = 9 degrees of movement)
        time_elapsed = time.time() - first_point[3]
        wait_time = 0.5
        if time_elapsed < wait_time:
            time.sleep(wait_time - time_elapsed)

        # Predict next position - drone moves "right" at 18 deg/s
        predicted_az = (first_point[0] + self.ANGULAR_VELOCITY_DEG * wait_time) % 360.0

        if self.initial_inclination != -1:
            # Inclination known - narrow search
            predicted_el = first_point[1]
            # Small arc scan since we know roughly where to look
            measurement = self.perform_arc_scan(predicted_az, predicted_el, 8.0, 4.0)
        else:
            # Inclination unknown - must search wide area
            # The drone could be at any elevation depending on orbital plane
            print("[Tracker] Unknown inclination - performing wide hemisphere search")

            # Search strategy: Cover the right hemisphere where drone should be
            # Center the search 9 degrees to the right of first point
            best_measurement = None
            best_strength = 0

            # Search in a grid pattern covering possible positions
            # Azimuth: ±15° around predicted position (covers timing uncertainty)
            # Elevation: ±30° from first point (covers most orbital inclinations)
            az_offsets = [0, 5, -5, 10, -10, 15, -15]
            el_offsets = [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30]

            # Quick grid search
            for el_offset in el_offsets:
                search_el = np.clip(first_point[1] + el_offset, 5, 85)

                for az_offset in az_offsets:
                    if not self.active or self.shared_data["shutdown"].value:
                        break

                    scan_az = (predicted_az + az_offset) % 360.0

                    # Move and check
                    self.command_motors_to_target(scan_az, search_el)
                    if self.wait_for_position(scan_az, search_el, timeout=0.05):

                        # Take readings
                        for _ in range(2):  # Fewer readings for speed
                            current_az, current_el, dist, strength, timestamp = self.read_current_state()

                            # Check if valid with clutter filtering
                            if self.is_valid_measurement(dist, strength, current_az, current_el):
                                if strength > best_strength:
                                    best_measurement = (current_az, current_el, dist, strength, timestamp)
                                    best_strength = strength
                                    print("[Tracker] Found candidate at ({:.1f}°, {:.1f}°) str={}".format(
                                        current_az, current_el, strength))

                            time.sleep(0.002)

                # If we found a strong signal, can stop early
                if best_strength > self.MIN_STRENGTH * 3:
                    print("[Tracker] Strong signal found, ending search early")
                    break

            measurement = best_measurement

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))

            # Calculate approximate inclination from first two points
            az_change = (az - first_point[0] + 180) % 360 - 180
            el_change = el - first_point[1]

            if abs(az_change) > 0.1:
                # Estimate the orbital plane inclination
                # If elevation increases as azimuth increases, plane is tilted
                approx_inclination = math.degrees(math.atan2(el_change, abs(az_change)))
                print("[Tracker] Second point found at ({:.1f}°, {:.1f}°)".format(az, el))
                print("[Tracker] Elevation change: {:.1f}°, Azimuth change: {:.1f}°".format(
                    el_change, az_change))
                print("[Tracker] Estimated orbital tilt: {:.1f}°".format(approx_inclination))

                # Store this estimate for future use
                if self.initial_inclination == -1:
                    self.initial_inclination = first_point[1] + approx_inclination * 5
            else:
                print("[Tracker] Second point found at ({:.1f}°, {:.1f}°)".format(az, el))

            self.state = TrackerState.CALCULATING_PLANE
        else:
            print("[Tracker] Failed to find second point, returning to search")
            self.orbit_points.clear()
            self.state = TrackerState.SEARCHING

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

            # Use moderate search radius since we're still learning the plane
            measurement = self.perform_arc_scan(predicted_az, predicted_el, 6.0, 4.0)

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
        # Use adaptive radius based on confidence
        elevation_radius = self.arc_radius_current * 0.7  # Slightly smaller for elevation
        measurement = self.perform_arc_scan(pred_az, pred_el,
                                            self.arc_radius_current,
                                            elevation_radius)

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

        # Search in expanding pattern
        search_radius = 10.0 + 5.0 * (self.consecutive_misses - self.max_misses)
        search_radius = min(search_radius, 45.0)  # Cap at 45 degrees

        predicted_az = (self.last_confirmed_point[0] + predicted_angle) % 360.0
        predicted_el = self.last_confirmed_point[1]

        # Use more scan points for wider searches
        num_points = self.arc_scan_points_wide if search_radius > 20 else self.arc_scan_points

        # Wide search in both azimuth and elevation
        measurement = self.perform_arc_scan(predicted_az, predicted_el,
                                            search_radius,
                                            search_radius * 0.8,
                                            num_points)

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
                        heading = 0

                    inclination = self.shared_data.get("inclination")
                    if inclination is not None:
                        inclination = inclination.value if hasattr(inclination, 'value') else float(inclination)
                    else:
                        inclination = 0

                    self.start_tracking(heading, 0.0, inclination)
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
def run_circular_tracker(shared_data, background_file="background_scan.npy"):
    """Run the circular drone tracker as a separate process"""
    try:
        # Ensure control flag exists
        if "circular_tracker_active" not in shared_data:
            from multiprocessing import Value
            shared_data["circular_tracker_active"] = Value('b', False)

        tracker = CircularDroneTracker(shared_data,
                                       prediction_time_sec=1,
                                       background_file=background_file)
        tracker.run()
    except Exception as e:
        print("[CircularTracker Process] Fatal error: {}".format(e))
        traceback.print_exc()


# Example of how to control the tracker from your main program
def start_circular_tracking(shared_data, heading=0, inclination=0):
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