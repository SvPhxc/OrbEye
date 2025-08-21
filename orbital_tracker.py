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
- Wait-and-scan strategy to find strongest signal (accounts for drone dimensions)
- Fallback for systems without scipy

Tracking Strategy:
1. SEARCHING: Find first strong signal from drone
2. CONFIRMING_DIRECTION: Wait for drone to move, find second point with wide search
3. CALCULATING_PLANE: Collect 5+ well-spaced points to define orbital plane
4. TRACKING: Predict intercept point, wait for drone arrival, track strongest signal
5. LOST: Expanding recovery search if tracking fails
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
            if len(self._cache) < 100000:
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
        self.TARGET_DISTANCE_CM = 100.0  # 2 meters in cm
        self.DISTANCE_TOLERANCE_CM = 20.0  # ±20cm tolerance
        self.ANGULAR_VELOCITY_DEG = 18.0  # degrees per second
        self.MIN_STRENGTH = 600  # Minimum LiDAR strength threshold
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
        self.refinement_interval = 3  # Refine plane every N points
        self.points_since_refinement = 0
        self.orbital_confidence = 0.0  # 0-1 confidence in orbital model

        # Search parameters
        self.initial_heading = 0
        self.initial_inclination = 30
        self.heading_deviation = 0.0

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

    def wait_and_scan_for_target(self, center_az, center_el, wait_time=0.5, scan_radius=2.0):
        """
        Move to position and wait for drone to arrive, continuously scanning.
        Returns the strongest measurement during the wait period.
        """
        # Move to intercept position
        self.command_motors_to_target(center_az, center_el)

        # Wait for motors to reach position
        if not self.wait_for_position(center_az, center_el, timeout=0.2):
            print("[Tracker] Failed to reach position ({:.1f}°, {:.1f}°)".format(center_az, center_el))
            return None

        print("[Tracker] Waiting at ({:.1f}°, {:.1f}°) for {:.2f}s".format(
            center_az, center_el, wait_time))

        # Now wait and continuously scan for the target
        start_wait = time.time()
        best_measurement = None
        best_strength = 0
        measurements_collected = []

        while (time.time() - start_wait) < wait_time:
            if not self.active or self.shared_data["shutdown"].value:
                break

            # Small spiral pattern around center while waiting
            elapsed = time.time() - start_wait
            spiral_angle = elapsed * math.pi * 4  # 2 rotations during wait
            spiral_radius = min(scan_radius * (elapsed / wait_time), scan_radius)

            # Calculate scan position
            scan_az = (center_az + spiral_radius * math.cos(spiral_angle)) % 360.0
            scan_el = np.clip(center_el + spiral_radius * math.sin(spiral_angle) * 0.5, 0, 90)

            # Move to scan position (small movement)
            self.command_motors_to_target(scan_az, scan_el)
            time.sleep(0.01)  # Brief settling time

            # Take multiple readings
            for _ in range(5):  # More readings while waiting
                current_az, current_el, dist, strength, timestamp = self.read_current_state()

                # Check if valid with clutter filtering
                if self.is_valid_measurement(dist, strength, current_az, current_el):
                    measurement = (current_az, current_el, dist, strength, timestamp)
                    measurements_collected.append(measurement)

                    # Track the strongest signal
                    if strength > best_strength:
                        best_measurement = measurement
                        best_strength = strength
                        print("[Tracker]   Found signal: str={:.0f} at ({:.1f}°, {:.1f}°)".format(
                            strength, current_az, current_el))

                time.sleep(0.002)  # 500Hz polling

        if best_measurement:
            print("[Tracker] Best signal during wait: str={:.0f}".format(best_strength))
        else:
            print("[Tracker] No valid target found during {:.2f}s wait".format(wait_time))

        return best_measurement

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
            print("[Tracker] Need at least 3 points for plane calculation")
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

        # Check if points are well-distributed (not all clustered)
        distances = []
        for i in range(len(points_3d) - 1):
            dist = np.linalg.norm(points_3d[i + 1] - points_3d[i])
            distances.append(dist)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist < 0.1:  # Points too close together (< 10cm spacing)
            print("[Tracker] Points too close for reliable plane calculation")
            return

        # Method 1: For well-spaced points, use SVD
        if len(points_3d) >= 4 and min_dist > 0.05:
            try:
                # Center the points
                center = np.mean(points_3d, axis=0)
                centered = points_3d - center

                # SVD to find best-fit plane
                U, s, Vt = np.linalg.svd(centered.T, full_matrices=False)

                # The normal is the singular vector with smallest singular value
                new_normal = Vt[-1, :]

                # Check quality of fit
                if s[2] > 0:
                    planarity = 1.0 - (s[2] / s[0])
                else:
                    planarity = 1.0

                print("[Tracker] SVD plane fit - Planarity: {:.3f}".format(planarity))

            except Exception as e:
                print("[Tracker] SVD failed: {}, using cross product".format(e))
                # Fallback to cross product method
                new_normal = self._calculate_normal_from_cross_product(points_3d)
        else:
            # Method 2: For fewer points, use cross product of well-separated vectors
            new_normal = self._calculate_normal_from_cross_product(points_3d)

        if new_normal is None:
            return

        # Ensure it's a proper 3D vector
        new_normal = np.array(new_normal).flatten()
        if new_normal.shape[0] != 3:
            print("[Tracker] ERROR: Invalid normal vector shape")
            return

        # Normalize
        norm = np.linalg.norm(new_normal)
        if norm < 1e-10:
            print("[Tracker] ERROR: Zero-length normal vector")
            return
        new_normal = new_normal / norm

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
            self.orbital_normal = new_normal

        # Ensure orbital_normal is properly shaped
        self.orbital_normal = np.array(self.orbital_normal).flatten()

        # Calculate inclination from normal vector
        # The inclination is the angle between the normal and the Z-axis
        z_component = abs(self.orbital_normal[2])
        inclination_deg = math.degrees(math.acos(np.clip(z_component, 0, 1)))

        # Update confidence
        if len(points_3d) >= 4:
            self.orbital_confidence = min(1.0, self.orbital_confidence + 0.1)

        # Update arc scan radius based on confidence
        self.arc_radius_current = self.arc_radius_initial * (1 - self.orbital_confidence * 0.7)
        self.arc_radius_current = max(self.arc_radius_current, self.arc_radius_min)

        print("[Tracker] Plane refined - Inclination: {:.1f}°, Normal: [{:.3f}, {:.3f}, {:.3f}], Conf: {:.2f}".format(
            inclination_deg, self.orbital_normal[0], self.orbital_normal[1],
            self.orbital_normal[2], self.orbital_confidence))

    def _calculate_normal_from_cross_product(self, points_3d):
        """Calculate normal vector using cross product of well-separated points"""
        if len(points_3d) < 2:
            return None

        # Find the two most separated points
        max_dist = 0
        best_pair = (0, 1)
        for i in range(len(points_3d)):
            for j in range(i + 1, len(points_3d)):
                dist = np.linalg.norm(points_3d[j] - points_3d[i])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)

        if max_dist < 0.1:  # Too close
            print("[Tracker] Points too close for cross product")
            return None

        # Use the best separated pair
        v1 = points_3d[best_pair[0]]
        v2 = points_3d[best_pair[1]]

        # Cross product gives normal to plane containing origin and two points
        normal = np.cross(v1, v2)

        return normal

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
        self.consecutive_hits = 0
        self.consecutive_misses = 0

        print("[Tracker] ========== STARTING CIRCULAR DRONE TRACKER ==========")
        print("[Tracker] Target: 2m distance, 18°/s angular velocity")
        print("[Tracker] Initial heading: {}".format(
            "{:.1f}°".format(initial_heading) if initial_heading != -1 else "UNKNOWN (will search)"))
        print("[Tracker] Initial inclination: {}".format(
            "{:.1f}°".format(initial_inclination) if initial_inclination != -1 else "UNKNOWN (will determine)"))
        print("[Tracker] Search deviation: ±{:.1f}°".format(heading_deviation / 2))
        print("[Tracker] Clutter filter: {}".format(
            "Active" if self.clutter_filter.background_data is not None else "Disabled"))
        print("[Tracker] =======================================================")
        print("[Tracker] Entering SEARCHING state")

    def stop_tracking(self):
        """Stop tracking and clear state"""
        self.active = False
        self.state = TrackerState.IDLE

        # Clear tracking data
        self.orbit_points.clear()
        self.last_confirmed_point = None
        self.last_confirmed_3d = None
        self.orbital_normal = None
        self.orbital_confidence = 0.0
        self.consecutive_hits = 0
        self.consecutive_misses = 0

        # Clear shared satellite points
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
        except:
            pass

        print("[Tracker] ========== STOPPED ==========")
        print("[Tracker] Tracking data cleared")

    def state_searching(self):
        """Search for first drone detection, looking for strongest signal"""
        # Determine search center
        if self.initial_heading != -1:
            # Search around known heading
            center_az = self.initial_heading
            center_el = 45 if self.initial_inclination == -1 else self.initial_inclination
            search_radius = self.heading_deviation / 2

            print("[Tracker] Searching near heading {:.1f}° at elevation {:.1f}°".format(
                center_az, center_el))
        else:
            # Sweep search
            sweep_time = time.time() % 10.0
            center_az = (sweep_time * 36.0) % 360.0
            center_el = 30.0
            search_radius = 20.0  # Wider search when heading unknown

            if int(sweep_time) % 2 == 0:  # Print every 2 seconds
                print("[Tracker] Sweep searching at {:.1f}°".format(center_az))

        # Use wait-and-scan to find the strongest signal in the area
        measurement = self.wait_and_scan_for_target(center_az, center_el,
                                                    wait_time=0.3,
                                                    scan_radius=search_radius)

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))
            print("[Tracker] First point found at ({:.1f}°, {:.1f}°) dist={:.0f}cm str={:.0f}".format(
                az, el, dist, strength))
            print("[Tracker] Moving to CONFIRMING_DIRECTION state")
            self.state = TrackerState.CONFIRMING_DIRECTION

    def state_confirming_direction(self):
        """Find second point after waiting for drone to move significantly"""
        if len(self.orbit_points) == 0:
            self.state = TrackerState.SEARCHING
            return

        first_point = self.orbit_points[-1]

        # Wait longer for drone to move (1 second = 18 degrees of movement)
        # This gives better separation for calculating the orbital plane
        time_elapsed = time.time() - first_point[3]
        wait_time = 1.0  # Increased from 0.5 to get better point separation
        if time_elapsed < wait_time:
            print("[Tracker] Waiting {:.2f}s for drone to move...".format(wait_time - time_elapsed))
            time.sleep(wait_time - time_elapsed)

        # Predict next position - drone moves "right" at 18 deg/s
        predicted_az = (first_point[0] + self.ANGULAR_VELOCITY_DEG * wait_time) % 360.0

        if self.initial_inclination != -1:
            # Inclination known - can predict elevation change
            predicted_el = first_point[1]  # Will refine this with actual inclination

            # Wait at predicted position for drone to arrive
            measurement = self.wait_and_scan_for_target(predicted_az, predicted_el,
                                                        wait_time=0.5, scan_radius=3.0)
        else:
            # Inclination unknown - must search wide area
            print("[Tracker] Unknown inclination - searching hemisphere for second point")

            # The drone has moved 18° in azimuth, but could be at any elevation
            # Search strategy: Try multiple intercept points at different elevations
            best_measurement = None
            best_strength = 0

            # Search elevations from -30° to +30° relative to first point
            elevation_steps = 11  # Check 11 different elevations
            for i in range(elevation_steps):
                el_offset = -30 + (60 * i / (elevation_steps - 1))
                search_el = np.clip(first_point[1] + el_offset, 5, 85)

                # Also vary azimuth slightly to account for timing uncertainty
                for az_offset in [0, -3, 3]:  # ±3° azimuth variation
                    scan_az = (predicted_az + az_offset) % 360.0

                    print("[Tracker] Checking ({:.1f}°, {:.1f}°)...".format(scan_az, search_el))

                    # Move and wait briefly
                    self.command_motors_to_target(scan_az, search_el)
                    if self.wait_for_position(scan_az, search_el, timeout=0.05):

                        # Take several readings to find strongest signal
                        for _ in range(5):
                            current_az, current_el, dist, strength, timestamp = self.read_current_state()

                            if self.is_valid_measurement(dist, strength, current_az, current_el):
                                if strength > best_strength:
                                    best_measurement = (current_az, current_el, dist, strength, timestamp)
                                    best_strength = strength
                                    print("[Tracker]   Found signal str={:.0f}".format(strength))

                            time.sleep(0.002)

                    # If we found a very strong signal, can stop searching
                    if best_strength > self.MIN_STRENGTH * 5:
                        print("[Tracker] Strong signal found, ending search")
                        break

                if best_strength > self.MIN_STRENGTH * 5:
                    break

            measurement = best_measurement

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))

            # Calculate motion vector from first to second point
            az_change = (az - first_point[0] + 180) % 360 - 180
            el_change = el - first_point[1]
            distance_2d = math.sqrt(az_change ** 2 + el_change ** 2)

            print("[Tracker] Second point found at ({:.1f}°, {:.1f}°) str={:.0f}".format(
                az, el, strength))
            print("[Tracker] Motion: Δaz={:.1f}°, Δel={:.1f}°, distance={:.1f}°".format(
                az_change, el_change, distance_2d))

            if distance_2d < 5.0:
                print("[Tracker] Points too close, need more separation")
                # Keep first point, try again with longer wait
                self.orbit_points.pop()  # Remove second point
                return

            self.state = TrackerState.CALCULATING_PLANE
        else:
            print("[Tracker] Failed to find second point, returning to search")
            self.orbit_points.clear()
            self.state = TrackerState.SEARCHING

    def state_calculating_plane(self):
        """Collect enough well-spaced points to reliably calculate orbital plane"""
        if len(self.orbit_points) < 2:
            self.state = TrackerState.SEARCHING
            return

        # Need at least 5 well-spaced points for reliable plane calculation
        min_points_needed = 5

        if len(self.orbit_points) < min_points_needed:
            # Collect more points
            last_point = self.orbit_points[-1]

            # Wait for drone to move further
            time_since_last = time.time() - last_point[3]
            wait_time = 0.5  # Wait 0.5s between points (9° separation)
            if time_since_last < wait_time:
                time.sleep(wait_time - time_since_last)

            # Predict next position
            predicted_az = (last_point[0] + self.ANGULAR_VELOCITY_DEG * wait_time) % 360.0
            predicted_el = last_point[1]

            # If we have an initial inclination estimate, use it
            if len(self.orbit_points) >= 2:
                # Estimate elevation change from previous points
                p1 = self.orbit_points[-2]
                p2 = self.orbit_points[-1]
                el_rate = (p2[1] - p1[1]) / ((p2[3] - p1[3]) if p2[3] != p1[3] else 1.0)
                predicted_el = np.clip(last_point[1] + el_rate * wait_time, 0, 90)

            print("[Tracker] Collecting point {} for plane calculation".format(len(self.orbit_points) + 1))

            # Use wait-and-scan to find the drone
            measurement = self.wait_and_scan_for_target(predicted_az, predicted_el,
                                                        wait_time=0.3,
                                                        scan_radius=4.0)

            if measurement:
                az, el, dist, strength, timestamp = measurement
                self.orbit_points.append((az, el, dist, time.time()))
                print("[Tracker] Point {} collected at ({:.1f}°, {:.1f}°) str={:.0f}".format(
                    len(self.orbit_points), az, el, strength))
            else:
                print("[Tracker] Failed to find point {}, trying again".format(len(self.orbit_points) + 1))
            return

        # We have enough points, calculate orbital plane
        print("[Tracker] Calculating orbital plane from {} points".format(len(self.orbit_points)))
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

            print("[Tracker] Orbital plane established, entering tracking mode")
            self.state = TrackerState.TRACKING
        else:
            print("[Tracker] Failed to calculate plane, need better points")
            # Remove oldest point and try again
            if len(self.orbit_points) > 2:
                self.orbit_points.popleft()
            else:
                # Start over
                self.orbit_points.clear()
                self.state = TrackerState.SEARCHING

    def state_tracking(self):
        """Main tracking loop using predict-and-wait intercept strategy"""
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

        # PREDICT: Calculate intercept point (where drone will be in prediction_time seconds)
        rotation_angle_rad = math.radians(self.prediction_angle)
        predicted_3d = self.rotate_vector_rodrigues(
            self.last_confirmed_3d,
            self.orbital_normal,
            rotation_angle_rad
        )

        # Convert to spherical coordinates
        pred_az, pred_el, pred_dist = self.cartesian_to_spherical(predicted_3d)

        print("[Tracker] Predicting intercept at ({:.1f}°, {:.1f}°)".format(pred_az, pred_el))

        # WAIT AND INTERCEPT: Move to predicted position and wait for drone
        # Use adaptive scan radius based on confidence
        scan_radius = self.arc_radius_current
        measurement = self.wait_and_scan_for_target(pred_az, pred_el,
                                                    wait_time=self.prediction_time,
                                                    scan_radius=scan_radius)

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

            # Update confidence and shrink arc radius
            self.orbital_confidence = min(1.0, self.orbital_confidence + 0.05)
            self.arc_radius_current = self.arc_radius_initial * (1 - self.orbital_confidence * 0.7)
            self.arc_radius_current = max(self.arc_radius_current, self.arc_radius_min)

            print("[Tracker] Intercepted #{} at ({:.1f}°, {:.1f}°) str={:.0f} - Arc: {:.1f}°, Conf: {:.2f}".format(
                self.consecutive_hits, az, el, strength, self.arc_radius_current, self.orbital_confidence))

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
            self.orbital_confidence = max(0.0, self.orbital_confidence - 0.15)

            # Increase arc radius for next attempt
            self.arc_radius_current = min(
                self.arc_radius_current * 1.5,
                self.arc_radius_initial * 3
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

        # Cap prediction at one full orbit
        if predicted_angle > 360:
            predicted_angle = predicted_angle % 360

        # Search radius expands with each miss
        search_radius = 10.0 + 5.0 * (self.consecutive_misses - self.max_misses)
        search_radius = min(search_radius, 45.0)  # Cap at 45 degrees

        predicted_az = (self.last_confirmed_point[0] + predicted_angle) % 360.0
        predicted_el = self.last_confirmed_point[1]

        print("[Tracker] Recovery search at ({:.1f}°, {:.1f}°) radius={:.1f}°".format(
            predicted_az, predicted_el, search_radius))

        # Use wait-and-scan with expanding radius
        measurement = self.wait_and_scan_for_target(predicted_az, predicted_el,
                                                    wait_time=1,
                                                    scan_radius=search_radius)

        if measurement:
            print("[Tracker] TARGET REACQUIRED!")
            az, el, dist, strength, timestamp = measurement

            # Reset tracking
            self.orbit_points.append((az, el, dist, time.time()))
            self.last_confirmed_point = (az, el, dist, time.time())

            # Ensure last_confirmed_3d is properly set as numpy array
            self.last_confirmed_3d = np.array(self.spherical_to_cartesian(az, el, dist)).flatten()

            self.consecutive_misses = 0
            self.arc_radius_current = self.arc_radius_initial

            print("[Tracker] Returning to TRACKING state")
            self.state = TrackerState.TRACKING
        else:
            self.consecutive_misses += 1

            if self.consecutive_misses > self.max_misses + 10:
                print("[Tracker] Reacquisition failed after {} attempts, restarting search".format(
                    self.consecutive_misses))
                self.orbit_points.clear()
                self.orbital_normal = None
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
                        inclination =0

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