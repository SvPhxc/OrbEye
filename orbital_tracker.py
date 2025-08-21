#!/usr/bin/env python3
"""
CircularDroneTracker: Real-time tracking of a drone in circular orbit
Uses predict-and-wait intercept strategy with continuous orbital refinement

Key Concepts:
- The drone orbits in a great circle at 2m from the sensor
- Inclination = angle of the line when plotting azimuth (x) vs elevation (y)
- Elevation = base_elevation + (azimuth_change * tan(inclination))
- The tracker finds this linear relationship and uses it for prediction

Features:
- Clutter filtering to reject background objects
- Line scanning along expected orbital paths
- Adaptive search that narrows as confidence increases
- Wait-and-scan strategy to find strongest signal (accounts for drone dimensions)
- Inclination deviation handling for uncertain initial estimates

Tracking Strategy:
1. SEARCHING: Find first strong signal from drone
2. CONFIRMING_DIRECTION: Wait for drone to move, search along possible inclination lines
3. CALCULATING_PLANE: Collect 4+ points to determine inclination via linear regression
4. TRACKING: Predict position along orbital line, wait for drone arrival
5. LOST: Search along extended orbital line for recovery
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
                angular_dist = np.sqrt(az_diff**2 + el_diff**2)
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

        # Orbital parameters
        self.orbital_inclination = None  # Inclination angle in degrees (slope of az vs el line)
        self.orbital_el_intercept = None  # Y-intercept of the az-el line

        # Orbital refinement
        self.refinement_interval = 5  # Refine plane every N points
        self.points_since_refinement = 0
        self.orbital_confidence = 0.0  # 0-1 confidence in orbital model

        # Search parameters
        self.initial_heading = -1
        self.initial_inclination = -1
        self.heading_deviation = 30.0
        self.inclination_deviation = 10.0  # Added: uncertainty in inclination

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

    def wait_and_scan_along_line(self, center_az, center_el, wait_time=0.5, scan_radius=2.0, inclination_deg=None):
        """
        Move to position and wait for drone to arrive, scanning along expected orbital line.
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

        # Now wait and continuously scan along the orbital line
        start_wait = time.time()
        best_measurement = None
        best_strength = 0

        # If inclination is known, scan along that line
        # If unknown, scan small area around center
        if inclination_deg is not None:
            tan_incl = math.tan(math.radians(inclination_deg))
        else:
            tan_incl = 0  # Assume horizontal if unknown

        while (time.time() - start_wait) < wait_time:
            if not self.active or self.shared_data["shutdown"].value:
                break

            # Scan along the line back and forth
            elapsed = time.time() - start_wait
            scan_progress = (elapsed / wait_time)

            # Oscillate along the line
            offset = scan_radius * math.sin(scan_progress * math.pi * 4)  # 2 oscillations

            scan_az = (center_az + offset) % 360.0
            scan_el = center_el + offset * tan_incl if inclination_deg is not None else center_el
            scan_el = np.clip(scan_el, 0, 90)

            # Move to scan position (small movement)
            self.command_motors_to_target(scan_az, scan_el)
            time.sleep(0.01)  # Brief settling time

            # Take multiple readings
            for _ in range(5):  # More readings while waiting
                current_az, current_el, dist, strength, timestamp = self.read_current_state()

                # Check if valid with clutter filtering
                if self.is_valid_measurement(dist, strength, current_az, current_el):
                    measurement = (current_az, current_el, dist, strength, timestamp)

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

    def refine_orbital_inclination(self):
        """Refine inclination estimate using linear regression on az-el points"""
        if len(self.orbit_points) < 3:
            return

        # Extract azimuth and elevation from recent points
        azimuths = []
        elevations = []

        for pt in self.orbit_points:
            azimuths.append(pt[0])
            elevations.append(pt[1])

        # Handle azimuth wraparound (if points cross 0°/360° boundary)
        # Unwrap azimuths to avoid discontinuity
        unwrapped_az = []
        prev_az = azimuths[0]
        unwrapped_az.append(prev_az)

        for az in azimuths[1:]:
            diff = az - prev_az
            if diff > 180:
                az -= 360
            elif diff < -180:
                az += 360
            unwrapped_az.append(az)
            prev_az = az

        # Perform linear regression: el = intercept + slope * az
        # slope = tan(inclination)
        az_array = np.array(unwrapped_az)
        el_array = np.array(elevations)

        # Calculate linear fit
        n = len(az_array)
        sum_az = np.sum(az_array)
        sum_el = np.sum(el_array)
        sum_az_sq = np.sum(az_array**2)
        sum_az_el = np.sum(az_array * el_array)

        # Slope (tan of inclination)
        denominator = n * sum_az_sq - sum_az**2
        if abs(denominator) > 1e-10:
            slope = (n * sum_az_el - sum_az * sum_el) / denominator
            intercept = (sum_el - slope * sum_az) / n

            # Convert slope to inclination angle
            new_inclination = math.degrees(math.atan(slope))

            # Calculate R-squared for fit quality
            el_pred = intercept + slope * az_array
            ss_res = np.sum((el_array - el_pred)**2)
            ss_tot = np.sum((el_array - np.mean(el_array))**2)

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0

            # Update inclination with smoothing
            if self.orbital_inclination is not None:
                # Smooth update
                alpha = 0.3
                self.orbital_inclination = (1 - alpha) * self.orbital_inclination + alpha * new_inclination
            else:
                self.orbital_inclination = new_inclination

            self.orbital_el_intercept = intercept

            # Update confidence based on fit quality
            self.orbital_confidence = 0.7 * self.orbital_confidence + 0.3 * r_squared

            print("[Tracker] Inclination refined: {:.1f}° (R²={:.3f}, Conf={:.2f})".format(
                self.orbital_inclination, r_squared, self.orbital_confidence))
        else:
            print("[Tracker] Cannot refine inclination - points are vertically aligned")

    def _calculate_normal_from_cross_product(self, points_3d):
        """Calculate normal vector using cross product of well-separated points"""
        # This is kept for compatibility but simplified
        # The actual inclination is calculated from the linear fit
        if len(points_3d) < 2:
            return None

        # Simple cross product of two vectors from origin
        v1 = points_3d[0]
        v2 = points_3d[-1]

        normal = np.cross(v1, v2)
        return normal

    def perform_arc_scan(self, center_az, center_el, radius_deg, radius_el_deg=None, num_points=None):
        """
        Legacy arc scan for compatibility - scans in a circular pattern.
        Now primarily used for fallback when inclination is completely unknown.
        """
        best_measurement = None
        best_strength = 0

        if radius_el_deg is None:
            radius_el_deg = radius_deg * 0.5

        if num_points is None:
            num_points = self.arc_scan_points

        # Generate arc points in a circle
        for i in range(num_points):
            if not self.active or self.shared_data["shutdown"].value:
                break

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

    def perform_line_scan(self, center_az, center_el, scan_length=10.0, inclination_deg=None, num_inclinations=5):
        """
        Perform scan along expected orbital path (straight line in az-el space).
        If inclination is known, scan along that line.
        If unknown, try multiple possible inclinations.

        scan_length: total length of scan in degrees
        inclination_deg: known inclination angle (None if unknown)
        num_inclinations: number of different inclinations to try if unknown
        """
        best_measurement = None
        best_strength = 0

        if inclination_deg is not None:
            # Known inclination - scan along the line
            inclinations_to_try = [inclination_deg]
        else:
            # Unknown inclination - try multiple angles from -45° to +45°
            inclinations_to_try = []
            for i in range(num_inclinations):
                angle = -45 + (90 * i / (num_inclinations - 1))
                inclinations_to_try.append(angle)

        for incl in inclinations_to_try:
            if not self.active or self.shared_data["shutdown"].value:
                break

            # Calculate points along the line defined by this inclination
            # el = el_center + (az - az_center) * tan(inclination)
            tan_incl = math.tan(math.radians(incl))

            # Scan points along the line
            for i in range(self.arc_scan_points):
                # Distribute points along the line
                offset = -scan_length/2 + (scan_length * i / (self.arc_scan_points - 1))

                scan_az = (center_az + offset) % 360.0
                scan_el = center_el + offset * tan_incl
                scan_el = np.clip(scan_el, 0, 90)

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
                            if inclination_deg is None:
                                print("[Tracker]   Found signal at incl={:.1f}°: str={:.0f}".format(
                                    incl, strength))

                    time.sleep(0.002)  # 500Hz polling

            # If we found a strong signal and inclination is unknown, stop trying others
            if inclination_deg is None and best_strength > self.MIN_STRENGTH * 3:
                break

        return best_measurement

    def start_tracking(self, initial_heading=-1, heading_deviation=30.0,
                       initial_inclination=-1, inclination_deviation=10.0):
        """Initialize search parameters and start tracking"""
        self.initial_heading = initial_heading
        self.heading_deviation = heading_deviation
        self.initial_inclination = initial_inclination
        self.inclination_deviation = inclination_deviation
        self.state = TrackerState.SEARCHING
        self.active = True

        # Reset tracking state
        self.orbit_points.clear()
        self.orbital_inclination = None
        self.orbital_el_intercept = None
        self.orbital_normal = None
        self.orbital_confidence = 0.0
        self.arc_radius_current = self.arc_radius_initial
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.last_confirmed_point = None
        self.last_confirmed_3d = None

        print("[Tracker] ========== STARTING CIRCULAR DRONE TRACKER ==========")
        print("[Tracker] Target: 2m distance, 18°/s angular velocity")
        print("[Tracker] Initial heading: {}".format(
            "{:.1f}°".format(initial_heading) if initial_heading != -1 else "UNKNOWN (will search)"))
        print("[Tracker] Heading deviation: ±{:.1f}°".format(heading_deviation/2))
        print("[Tracker] Initial inclination: {}".format(
            "{:.1f}°".format(initial_inclination) if initial_inclination != -1 else "UNKNOWN (will determine)"))
        if initial_inclination != -1:
            print("[Tracker] Inclination deviation: ±{:.1f}°".format(inclination_deviation))
        print("[Tracker] Clutter filter: {}".format(
            "Active" if self.clutter_filter.background_data is not None else "Disabled"))
        print("[Tracker] =======================================================")
        print("[Tracker] Note: Inclination is the slope of the az-el line")
        print("[Tracker] (angle of elevation change per degree of azimuth)")
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

            print("[Tracker] Searching near heading {:.1f}° at elevation {:.1f}°".format(
                center_az, center_el))

            if self.initial_inclination != -1:
                # We have inclination - search along possible orbital lines
                measurement = self.perform_line_scan(center_az, center_el,
                                                     scan_length=self.heading_deviation,
                                                     inclination_deg=self.initial_inclination,
                                                     num_inclinations=3)  # Try ±inclination_deviation
            else:
                # No inclination - use wait and scan
                measurement = self.wait_and_scan_along_line(center_az, center_el,
                                                            wait_time=0.3,
                                                            scan_radius=self.heading_deviation / 2,
                                                            inclination_deg=None)
        else:
            # Sweep search - no information available
            sweep_time = time.time() % 10.0
            center_az = (sweep_time * 36.0) % 360.0
            center_el = 30.0

            if int(sweep_time) % 2 == 0:  # Print every 2 seconds
                print("[Tracker] Sweep searching at {:.1f}°".format(center_az))

            # Try multiple inclinations during sweep
            measurement = self.perform_line_scan(center_az, center_el,
                                                 scan_length=20.0,
                                                 inclination_deg=None,
                                                 num_inclinations=5)

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
        time_elapsed = time.time() - first_point[3]
        wait_time = 1.0  # Get good separation between points
        if time_elapsed < wait_time:
            print("[Tracker] Waiting {:.2f}s for drone to move...".format(wait_time - time_elapsed))
            time.sleep(wait_time - time_elapsed)

        # Predict next position - drone moves "right" at 18 deg/s
        predicted_az = (first_point[0] + self.ANGULAR_VELOCITY_DEG * wait_time) % 360.0

        if self.initial_inclination != -1:
            # Inclination known (with uncertainty) - search along possible inclination lines
            print("[Tracker] Searching with known inclination {:.1f}° ± {:.1f}°".format(
                self.initial_inclination, self.inclination_deviation))

            best_measurement = None
            best_strength = 0

            # Try different inclinations within the deviation range
            for incl_offset in [0, -self.inclination_deviation/2, self.inclination_deviation/2,
                                -self.inclination_deviation, self.inclination_deviation]:
                test_incl = self.initial_inclination + incl_offset
                tan_incl = math.tan(math.radians(test_incl))

                # Calculate expected elevation based on inclination
                az_change = self.ANGULAR_VELOCITY_DEG * wait_time
                predicted_el = first_point[1] + az_change * tan_incl
                predicted_el = np.clip(predicted_el, 0, 90)

                # Check this position
                self.command_motors_to_target(predicted_az, predicted_el)
                if self.wait_for_position(predicted_az, predicted_el, timeout=0.1):
                    for _ in range(5):
                        current_az, current_el, dist, strength, timestamp = self.read_current_state()

                        if self.is_valid_measurement(dist, strength, current_az, current_el):
                            if strength > best_strength:
                                best_measurement = (current_az, current_el, dist, strength, timestamp)
                                best_strength = strength
                                print("[Tracker]   Found at incl={:.1f}°: str={:.0f}".format(
                                    test_incl, strength))

                        time.sleep(0.002)

                if best_strength > self.MIN_STRENGTH * 3:
                    break

            measurement = best_measurement
        else:
            # Inclination unknown - search multiple possible inclination lines
            print("[Tracker] Unknown inclination - searching multiple orbital lines")

            # Use line scan to try different inclinations
            measurement = self.perform_line_scan(predicted_az, first_point[1],
                                                 scan_length=20.0,
                                                 inclination_deg=None,
                                                 num_inclinations=9)  # Try 9 different inclinations

        if measurement:
            az, el, dist, strength, timestamp = measurement
            self.orbit_points.append((az, el, dist, time.time()))

            # Calculate inclination from the two points
            # Inclination is the slope of the line in az-el space
            az_change = (az - first_point[0] + 180) % 360 - 180  # Handle wraparound
            el_change = el - first_point[1]

            if abs(az_change) > 1.0:  # Need some azimuth change
                # tan(inclination) = Δel / Δaz
                calculated_inclination = math.degrees(math.atan2(el_change, az_change))

                print("[Tracker] Second point found at ({:.1f}°, {:.1f}°) str={:.0f}".format(
                    az, el, strength))
                print("[Tracker] Motion: Δaz={:.1f}°, Δel={:.1f}°".format(az_change, el_change))
                print("[Tracker] Calculated inclination: {:.1f}°".format(calculated_inclination))

                # Store the calculated inclination
                self.orbital_inclination = calculated_inclination
                self.orbital_el_intercept = first_point[1] - first_point[0] * math.tan(math.radians(calculated_inclination))
            else:
                print("[Tracker] Points too close in azimuth, need more separation")
                self.orbit_points.pop()  # Remove second point
                return

            self.state = TrackerState.CALCULATING_PLANE
        else:
            print("[Tracker] Failed to find second point, returning to search")
            self.orbit_points.clear()
            self.state = TrackerState.SEARCHING

    def state_calculating_plane(self):
        """Collect enough points to reliably calculate orbital inclination"""
        if len(self.orbit_points) < 2:
            self.state = TrackerState.SEARCHING
            return

        # Need at least 4 points for reliable inclination calculation
        min_points_needed = 4

        if len(self.orbit_points) < min_points_needed:
            # Collect more points
            last_point = self.orbit_points[-1]

            # Wait for drone to move further
            time_since_last = time.time() - last_point[3]
            wait_time = 0.5  # Wait 0.5s between points (9° separation)
            if time_since_last < wait_time:
                time.sleep(wait_time - time_since_last)

            # Predict next position based on current inclination estimate
            predicted_az = (last_point[0] + self.ANGULAR_VELOCITY_DEG * wait_time) % 360.0

            # If we have an inclination estimate, use it
            if self.orbital_inclination is not None:
                tan_incl = math.tan(math.radians(self.orbital_inclination))
                predicted_el = last_point[1] + self.ANGULAR_VELOCITY_DEG * wait_time * tan_incl
                predicted_el = np.clip(predicted_el, 0, 90)
            else:
                # Use simple estimate from first two points
                if len(self.orbit_points) >= 2:
                    p1 = self.orbit_points[-2]
                    p2 = self.orbit_points[-1]
                    el_rate = (p2[1] - p1[1]) / ((p2[0] - p1[0]) if p2[0] != p1[0] else 1.0)
                    predicted_el = last_point[1] + self.ANGULAR_VELOCITY_DEG * wait_time * el_rate
                    predicted_el = np.clip(predicted_el, 0, 90)
                else:
                    predicted_el = last_point[1]

            print("[Tracker] Collecting point {} for inclination calculation".format(len(self.orbit_points) + 1))

            # Use wait-and-scan to find the drone
            measurement = self.wait_and_scan_along_line(predicted_az, predicted_el,
                                                        wait_time=0.3,
                                                        scan_radius=4.0,
                                                        inclination_deg=self.orbital_inclination)

            if measurement:
                az, el, dist, strength, timestamp = measurement
                self.orbit_points.append((az, el, dist, time.time()))
                print("[Tracker] Point {} collected at ({:.1f}°, {:.1f}°) str={:.0f}".format(
                    len(self.orbit_points), az, el, strength))
            else:
                print("[Tracker] Failed to find point {}, trying again".format(len(self.orbit_points) + 1))
            return

        # We have enough points, calculate inclination
        print("[Tracker] Calculating orbital inclination from {} points".format(len(self.orbit_points)))
        self.refine_orbital_inclination()

        if self.orbital_inclination is not None:
            # Initialize tracking state
            last_point = self.orbit_points[-1]
            self.last_confirmed_point = last_point

            # For 3D calculations (if needed for compatibility)
            self.last_confirmed_3d = np.array(self.spherical_to_cartesian(
                last_point[0], last_point[1], last_point[2])).flatten()

            # Calculate the 3D orbital normal for compatibility
            # For a tilted great circle, the normal depends on the inclination
            # This is simplified - the actual normal would depend on the specific orbit
            inclination_rad = math.radians(self.orbital_inclination)
            self.orbital_normal = np.array([
                -math.sin(inclination_rad),
                0,
                math.cos(inclination_rad)
            ])

            print("[Tracker] Orbital inclination established: {:.1f}°".format(self.orbital_inclination))
            print("[Tracker] Entering tracking mode")
            self.state = TrackerState.TRACKING
        else:
            print("[Tracker] Failed to calculate inclination, need better points")
            # Remove oldest point and try again
            if len(self.orbit_points) > 2:
                self.orbit_points.popleft()
            else:
                # Start over
                self.orbit_points.clear()
                self.state = TrackerState.SEARCHING

    def state_tracking(self):
        """Main tracking loop using predict-and-wait intercept strategy"""
        if self.orbital_inclination is None:
            print("[Tracker] ERROR: No orbital inclination calculated")
            self.state = TrackerState.SEARCHING
            return

        if self.last_confirmed_point is None:
            print("[Tracker] ERROR: No last confirmed point")
            self.state = TrackerState.SEARCHING
            return

        # PREDICT: Calculate intercept point using simple linear model
        # The drone moves along a straight line in az-el space:
        # el = el_start + (az - az_start) * tan(inclination)

        # Predict where drone will be after prediction_time
        predicted_az = (self.last_confirmed_point[0] + self.prediction_angle) % 360.0

        # Calculate expected elevation using the inclination
        tan_incl = math.tan(math.radians(self.orbital_inclination))
        az_change = self.prediction_angle
        predicted_el = self.last_confirmed_point[1] + az_change * tan_incl
        predicted_el = np.clip(predicted_el, 0, 90)

        print("[Tracker] Predicting intercept at ({:.1f}°, {:.1f}°) using incl={:.1f}°".format(
            predicted_az, predicted_el, self.orbital_inclination))

        # WAIT AND INTERCEPT: Move to predicted position and wait for drone
        # Use adaptive scan radius based on confidence
        scan_radius = self.arc_radius_current
        measurement = self.wait_and_scan_along_line(predicted_az, predicted_el,
                                                    wait_time=self.prediction_time,
                                                    scan_radius=scan_radius,
                                                    inclination_deg=self.orbital_inclination)

        if measurement:
            # Success - update tracking
            az, el, dist, strength, timestamp = measurement

            # Add to orbit points for continuous refinement
            self.orbit_points.append((az, el, dist, time.time()))
            self.last_confirmed_point = (az, el, dist, time.time())

            self.consecutive_hits += 1
            self.consecutive_misses = 0
            self.points_since_refinement += 1

            # Refine inclination periodically
            if self.points_since_refinement >= self.refinement_interval:
                self.refine_orbital_inclination()
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
        """Try to reacquire target with expanding search along orbital line"""
        if self.last_confirmed_point is None:
            self.state = TrackerState.SEARCHING
            return

        # Predict where drone should be now based on last known position
        time_elapsed = time.time() - self.last_confirmed_point[3]
        predicted_angle = self.ANGULAR_VELOCITY_DEG * time_elapsed

        # Cap prediction at one full orbit
        if predicted_angle > 360:
            predicted_angle = predicted_angle % 360

        # Predict position along the orbital line
        predicted_az = (self.last_confirmed_point[0] + predicted_angle) % 360.0

        if self.orbital_inclination is not None:
            # Use known inclination to predict elevation
            tan_incl = math.tan(math.radians(self.orbital_inclination))
            predicted_el = self.last_confirmed_point[1] + predicted_angle * tan_incl
            predicted_el = np.clip(predicted_el, 0, 90)
        else:
            predicted_el = self.last_confirmed_point[1]

        # Search radius expands with each miss
        search_radius = 10.0 + 5.0 * (self.consecutive_misses - self.max_misses)
        search_radius = min(search_radius, 45.0)  # Cap at 45 degrees

        print("[Tracker] Recovery search at ({:.1f}°, {:.1f}°) radius={:.1f}°".format(
            predicted_az, predicted_el, search_radius))

        # Use line scan if inclination is known, otherwise wait-and-scan
        if self.orbital_inclination is not None:
            # Search along the orbital line
            measurement = self.perform_line_scan(predicted_az, predicted_el,
                                                 scan_length=search_radius * 2,
                                                 inclination_deg=self.orbital_inclination)
        else:
            # Search in expanding area
            measurement = self.wait_and_scan_along_line(predicted_az, predicted_el,
                                                        wait_time=0.5,
                                                        scan_radius=search_radius,
                                                        inclination_deg=None)

        if measurement:
            print("[Tracker] TARGET REACQUIRED!")
            az, el, dist, strength, timestamp = measurement

            # Reset tracking
            self.orbit_points.append((az, el, dist, time.time()))
            self.last_confirmed_point = (az, el, dist, time.time())

            # Update 3D position if needed
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
                self.orbital_inclination = None
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
                        heading = -1

                    heading_dev = self.shared_data.get("heading_deviation")
                    if heading_dev is not None:
                        heading_dev = heading_dev.value if hasattr(heading_dev, 'value') else float(heading_dev)
                    else:
                        heading_dev = 30.0

                    inclination = self.shared_data.get("inclination")
                    if inclination is not None:
                        inclination = inclination.value if hasattr(inclination, 'value') else float(inclination)
                    else:
                        inclination = -1

                    inclination_dev = self.shared_data.get("inclination_deviation")
                    if inclination_dev is not None:
                        inclination_dev = inclination_dev.value if hasattr(inclination_dev, 'value') else float(inclination_dev)
                    else:
                        inclination_dev = 10.0

                    self.start_tracking(heading, heading_dev, inclination, inclination_dev)
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
                                       prediction_time_sec=0.5,
                                       background_file=background_file)
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