#!/usr/bin/env python3
"""
Robust LiDAR Target Tracker with Acquisition and Demo Modes
Key features:
- Initial acquisition scan triggered by acquire_points flag (independent of debug mode)
- Demo mode for tracking orbiting drone (2m away, 20s orbit)
- Waits for target_reached to ensure accurate positioning
- Verifies LiDAR data matches scan position
- Fixed angle wraparound at 0°/360° boundary
- Respects 1000Hz LiDAR polling limit
- Predictive tracking for smooth following
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
import math
from collections import deque

# ==============================================================================
# CONFIGURATION PARAMETERS - Adjust these for your system
# ==============================================================================

# LiDAR Parameters
LIDAR_MIN_INTERVAL = 0.001  # 1ms minimum between reads (1000Hz max)
MIN_STRENGTH_THRESHOLD = 30  # Much lower threshold to find any target
HIGH_CONFIDENCE_THRESHOLD = 100  # Strong signal threshold
ACQUISITION_STRENGTH_THRESHOLD = 20  # Even lower for acquisition

# Clutter Filter Parameters
ANGULAR_TOLERANCE = 1.0  # Degrees - for background matching
DISTANCE_MARGIN_CM = 50.0  # Reduced margin for better detection
CACHE_SIZE = 1500  # Number of cached clutter filter queries
DISABLE_CLUTTER_FOR_ACQUISITION = True  # Disable clutter filter during acquisition

# Movement Parameters
MOVEMENT_TIMEOUT = 1.0  # Seconds to wait for movement
POSITION_TOLERANCE = 2.0  # Degrees - acceptable position error
POSITION_VERIFY_DELAY = 0.015  # Slightly longer stabilization
MAX_POSITION_ERROR = 3.0  # Maximum acceptable position error in degrees

# Normal Tracking Parameters
SCAN_RADIUS_AZ = 10.0  # Degrees - normal scan radius azimuth
SCAN_RADIUS_EL = 10.0  # Degrees - normal scan radius elevation
SCAN_POINTS = 6  # Number of points in scan circle
MAX_SCAN_RADIUS_AZ = 30.0  # Maximum expanded search radius
MAX_SCAN_RADIUS_EL = 25.0  # Maximum expanded search radius

# Acquisition Parameters - Improved for better detection
ACQUISITION_AZ_RANGE = 60.0  # Total azimuth range to scan (±30°)
ACQUISITION_AZ_STEP = 10.0  # Step size for azimuth scanning
ACQUISITION_ELEVATIONS = [45, 40, 50, 35, 55, 30, 60, 25, 65, 20, 70]  # More levels
ACQUISITION_MIN_DISTANCE = 10.0  # Minimum valid distance (cm)
ACQUISITION_MAX_ATTEMPTS = 3  # Retry reading at each point

# Demo Mode Parameters (Orbiting Drone)
DEMO_ORBIT_TIME = 20.0  # Seconds for full orbit
DEMO_RADIUS_MIN = 150.0  # Minimum distance for drone (1.5m in cm)
DEMO_RADIUS_MAX = 250.0  # Maximum distance for drone (2.5m in cm)
DEMO_CENTER_ELEVATION = 45.0  # Default center elevation for orbit
DEMO_SCAN_RADIUS = 5.0  # Tight radius for predictive tracking

# Tracking State Parameters
MAX_LOST_COUNT = 5  # How many cycles before expanding search
HISTORY_SIZE = 4  # Position history for smoothing
TARGET_HISTORY_SIZE = 3  # Target history for smoothing

# Performance Parameters
MIN_CYCLE_TIME = 0.05  # Minimum 50ms per tracking cycle
STATS_PRINT_INTERVAL = 20  # Print statistics every N cycles


# ==============================================================================


class ClutterFilter:
    """
    Optimized clutter filter with caching for better performance.
    """

    def __init__(self, background_file="background_scan.npy"):
        self.angular_tolerance = ANGULAR_TOLERANCE
        self.distance_margin_cm = DISTANCE_MARGIN_CM
        self.background_tree = None
        self.background_data = None
        self._query_cache = {}
        self._cache_size = CACHE_SIZE

        try:
            self.background_data = np.load(background_file)
            print(f"[ClutterFilter] Loaded {len(self.background_data)} background points.")

            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords, leafsize=16)
            self.bg_distances = self.background_data[:, 2]
            print("[ClutterFilter] K-d tree built successfully.")

        except FileNotFoundError:
            print(f"[ClutterFilter] WARNING: Background file '{background_file}' not found.")
        except Exception as e:
            print(f"[ClutterFilter] ERROR: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """Check if target is valid with caching."""
        if self.background_tree is None:
            return True

        cache_key = (int(azimuth * 10), int(elevation * 10))

        if cache_key in self._query_cache:
            bg_distance = self._query_cache[cache_key]
        else:
            query_point = np.array([azimuth, elevation])
            try:
                angular_dist, idx = self.background_tree.query(query_point, k=1)

                if angular_dist < self.angular_tolerance:
                    bg_distance = self.bg_distances[idx]
                else:
                    bg_distance = float('inf')

                if len(self._query_cache) < self._cache_size:
                    self._query_cache[cache_key] = bg_distance

            except Exception as e:
                print(f"[ClutterFilter] Query error: {e}")
                return True

        return distance < (bg_distance - self.distance_margin_cm)


class AngleHandler:
    """Robust angle handling with 0/360 wraparound support."""

    @staticmethod
    def normalize(angle):
        """Normalize angle to [0, 360) range."""
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle

    @staticmethod
    def difference(angle1, angle2):
        """Calculate shortest angular difference between two angles."""
        diff = (angle2 - angle1 + 180) % 360 - 180
        return diff

    @staticmethod
    def shortest_path(current, target):
        """Calculate the target angle that requires minimum rotation."""
        diff = AngleHandler.difference(current, target)
        return AngleHandler.normalize(current + diff)

    @staticmethod
    def circular_mean(angles):
        """Compute mean of angles, properly handling wraparound."""
        if not angles:
            return None
        x_sum = sum(math.cos(math.radians(a)) for a in angles)
        y_sum = sum(math.sin(math.radians(a)) for a in angles)
        mean_angle = math.degrees(math.atan2(y_sum, x_sum))
        return AngleHandler.normalize(mean_angle)

    @staticmethod
    def is_near_boundary(angle, threshold=15):
        """Check if angle is near the 0/360 boundary."""
        norm_angle = AngleHandler.normalize(angle)
        return norm_angle < threshold or norm_angle > (360 - threshold)


class TargetTracker:
    """
    Robust tracker with acquisition mode, position verification, and demo mode.
    """

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file=background_file)
        self.angle_handler = AngleHandler()

        # Use configuration parameters
        self.lidar_min_interval = LIDAR_MIN_INTERVAL
        self.last_lidar_read = 0

        # Normal tracking parameters
        self.scan_radius_az = SCAN_RADIUS_AZ
        self.scan_radius_el = SCAN_RADIUS_EL
        self.scan_points = SCAN_POINTS
        self.min_strength_threshold = MIN_STRENGTH_THRESHOLD
        self.high_confidence_threshold = HIGH_CONFIDENCE_THRESHOLD

        # Movement parameters
        self.movement_timeout = MOVEMENT_TIMEOUT
        self.position_tolerance = POSITION_TOLERANCE
        self.position_verify_delay = POSITION_VERIFY_DELAY
        self.max_position_error = MAX_POSITION_ERROR

        # Demo mode parameters
        self.demo_mode = False
        self.demo_heading = 0.0
        self.demo_inclination = -1
        self.demo_orbit_time = DEMO_ORBIT_TIME
        self.demo_angular_velocity = 360.0 / DEMO_ORBIT_TIME
        self.demo_center_el = DEMO_CENTER_ELEVATION
        self.demo_last_update = None
        self.demo_orbit_points = []
        self.demo_orbit_determined = False

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.tracking_confidence = 0.0
        self.consecutive_good_tracks = 0

        # History tracking
        self.target_history = deque(maxlen=TARGET_HISTORY_SIZE)
        self.position_history = deque(maxlen=HISTORY_SIZE)
        self.lost_target_count = 0
        self.max_lost_count = MAX_LOST_COUNT

        # Performance monitoring
        self.cycle_count = 0
        self.successful_reads = 0
        self.failed_reads = 0

        print("[Tracker] Robust tracker initialized")
        print(f"[Tracker] Strength thresholds: acquisition={ACQUISITION_STRENGTH_THRESHOLD}, "
              f"min={MIN_STRENGTH_THRESHOLD}, high={HIGH_CONFIDENCE_THRESHOLD}")
        print(f"[Tracker] Normal scan: ±{self.scan_radius_az}° with {self.scan_points} points")
        print(f"[Tracker] Acquisition: ±{ACQUISITION_AZ_RANGE / 2}° range, {ACQUISITION_AZ_STEP}° steps")

    def wait_for_target_reached(self, timeout=None):
        """Wait for the system to reach the target position."""
        if timeout is None:
            timeout = self.movement_timeout

        start_time = time.time()

        # First wait for movement to start
        while self.shared_data["go_to_target"].value and time.time() - start_time < 0.1:
            time.sleep(0.001)

        # Now wait for target_reached flag
        while time.time() - start_time < timeout:
            if self.shared_data["shutdown"].value:
                return False

            if self.shared_data["target_reached"].value:
                self.shared_data["target_reached"].value = False
                time.sleep(self.position_verify_delay)
                return True

            time.sleep(0.002)

        return False

    def move_to_position_verified(self, azimuth, elevation):
        """Move to position and verify we actually reached it."""
        if self.shared_data["shutdown"].value:
            return False

        # Get current position
        current_az = self.shared_data["stepper_degrees"].value

        # Use shortest path for azimuth
        target_az = self.angle_handler.shortest_path(current_az, azimuth)

        # Ensure target_az is in valid range
        target_az = self.angle_handler.normalize(target_az)

        # Set targets
        self.shared_data["target_azimuth"].value = target_az
        self.shared_data["target_elevation"].value = elevation

        # Trigger movement
        self.shared_data["go_to_target"].value = True

        # Wait for target to be reached
        if not self.wait_for_target_reached():
            self.failed_reads += 1
            return False

        # Verify position
        actual_az = self.shared_data["stepper_degrees"].value
        actual_el = self.shared_data["servo_degrees"].value

        az_error = abs(self.angle_handler.difference(actual_az, target_az))
        el_error = abs(actual_el - elevation)

        if az_error > self.max_position_error or el_error > self.max_position_error:
            self.failed_reads += 1
            return False

        self.successful_reads += 1
        return True

    def read_lidar_at_position(self):
        """Read LiDAR data after confirming position."""
        # Ensure we don't exceed 1000Hz
        elapsed = time.time() - self.last_lidar_read
        if elapsed < self.lidar_min_interval:
            time.sleep(self.lidar_min_interval - elapsed)

        # Get current actual position
        actual_az = self.shared_data["stepper_degrees"].value
        actual_el = self.shared_data["servo_degrees"].value

        # Read LiDAR data
        with self.shared_data["lidar_data"].get_lock():
            distance = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]

        self.last_lidar_read = time.time()

        return actual_az, actual_el, distance, strength

    def acquisition_scan(self):
        """Perform wide initial scan to find any target - improved for better detection."""
        print("[Acquisition] Starting improved acquisition scan...")
        print(f"[Acquisition] Using threshold: {ACQUISITION_STRENGTH_THRESHOLD}")

        all_targets = []  # Collect all potential targets

        # Get starting position
        start_az = self.shared_data["stepper_degrees"].value
        print(f"[Acquisition] Starting from azimuth {start_az:.1f}°")

        # Calculate azimuth scan points without going over 360
        az_points = [start_az]
        offset = ACQUISITION_AZ_STEP

        # The total range is 60°, so we scan 30° to the left and 30° to the right.
        # We use ACQUISITION_AZ_RANGE / 2 as the limit for the offset.
        while offset <= (ACQUISITION_AZ_RANGE / 2):
            az_points.append(start_az + offset)  # Point to the right
            az_points.append(start_az - offset)  # Point to the left
            offset += ACQUISITION_AZ_STEP

        # Normalize all azimuth points to [0, 360)
        az_points = [self.angle_handler.normalize(az) for az in az_points]

        print(f"[Acquisition] Scanning {len(az_points)} azimuth points across {len(ACQUISITION_ELEVATIONS)} elevations")
        scan_count = 0
        for el_idx, elevation in enumerate(ACQUISITION_ELEVATIONS):
            if self.shared_data["shutdown"].value:
                break

            elevation = np.clip(elevation, 10, 80)

            # Reverse azimuth order for every other elevation (zigzag)
            if el_idx % 2 == 1:
                current_az_points = list(reversed(az_points))
            else:
                current_az_points = az_points

            for azimuth in current_az_points:
                if self.shared_data["shutdown"].value:
                    break

                scan_count += 1

                # Move to position and verify
                if not self.move_to_position_verified(azimuth, elevation):
                    print(f"[Acquisition] Failed to reach ({azimuth:.1f}°, {elevation:.1f}°)")
                    continue

                # Try multiple reads at this position for reliability
                for attempt in range(ACQUISITION_MAX_ATTEMPTS):
                    # Small delay between attempts
                    if attempt > 0:
                        time.sleep(0.002)

                    # Read LiDAR
                    actual_az, actual_el, distance, strength = self.read_lidar_at_position()

                    # Very relaxed criteria for acquisition
                    if distance > ACQUISITION_MIN_DISTANCE:
                        # During acquisition, optionally skip clutter filter or use relaxed version
                        is_valid = True
                        if not DISABLE_CLUTTER_FOR_ACQUISITION:
                            is_valid = self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength)

                        if is_valid and strength >= ACQUISITION_STRENGTH_THRESHOLD:
                            print(f"[Acquisition] Point {scan_count}: ({actual_az:.1f}°, {actual_el:.1f}°) "
                                  f"dist={distance:.0f}cm, str={strength:.0f}")
                            all_targets.append((actual_az, actual_el, distance, strength))

                            # If we find a decent target, we can stop
                            if strength >= MIN_STRENGTH_THRESHOLD:
                                print(f"[Acquisition] Good target found! Strength={strength:.0f}")
                                return (actual_az, actual_el, distance, strength)

        # If no good target found, return the best of what we found
        if all_targets:
            # Sort by strength
            all_targets.sort(key=lambda x: x[3], reverse=True)
            best = all_targets[0]
            print(f"[Acquisition] Best of {len(all_targets)} targets: "
                  f"({best[0]:.1f}°, {best[1]:.1f}°) str={best[3]:.0f}")
            return best

        print(f"[Acquisition] No targets found after scanning {scan_count} points")
        print("[Acquisition] Tips: Check if LiDAR is working, target is in range, or lower thresholds")
        return None

    def tracking_scan(self, center_az, center_el):
        """Perform tracking scan with position verification."""
        scan_results = []
        scan_start = time.time()

        # Adjust scan density based on confidence
        if self.tracking_confidence > 0.7:
            points_to_scan = max(3, self.scan_points - 2)
            radius_az = self.scan_radius_az * 0.7
            radius_el = self.scan_radius_el * 0.7
        else:
            points_to_scan = self.scan_points
            radius_az = self.scan_radius_az
            radius_el = self.scan_radius_el

        # Always scan center point first
        if self.move_to_position_verified(center_az, center_el):
            az, el, dist, strength = self.read_lidar_at_position()
            if dist > 0 and self.clutter_filter.is_valid_target(az, el, dist, strength):
                if strength >= self.min_strength_threshold:
                    scan_results.append((az, el, dist, strength))

        # Scan surrounding points
        for i in range(points_to_scan):
            if self.shared_data["shutdown"].value or time.time() - scan_start > 0.8:
                break

            angle = (2 * math.pi * i) / points_to_scan
            scan_az = center_az + radius_az * math.cos(angle)
            scan_el = center_el + radius_el * math.sin(angle)
            scan_el = np.clip(scan_el, 0, 90)
            scan_az = self.angle_handler.normalize(scan_az)

            if not self.move_to_position_verified(scan_az, scan_el):
                continue

            actual_az, actual_el, distance, strength = self.read_lidar_at_position()

            if distance > 0:
                if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                    if strength >= self.min_strength_threshold:
                        scan_results.append((actual_az, actual_el, distance, strength))

        return scan_results

    def find_best_target(self, scan_results):
        """Find the best target with consistency checking."""
        if not scan_results:
            return None

        # Sort by strength
        scan_results.sort(key=lambda x: x[3], reverse=True)

        # If we have history, prefer targets close to previous position
        if self.target_history and len(self.target_history) > 1:
            last_az, last_el = self.target_history[-1]

            def score_target(target):
                az, el, dist, strength = target
                az_diff = abs(self.angle_handler.difference(last_az, az))
                el_diff = abs(last_el - el)
                position_error = math.sqrt(az_diff ** 2 + el_diff ** 2)

                if position_error > 20:
                    return strength * 0.3
                elif position_error > 10:
                    return strength * 0.7
                else:
                    return strength

            best = max(scan_results, key=score_target)
        else:
            best = scan_results[0]

        return best

    def smooth_position(self, new_az, new_el):
        """Smooth position with proper angle wraparound handling."""
        self.position_history.append((new_az, new_el))

        if len(self.position_history) < 2:
            return new_az, new_el

        az_values = [p[0] for p in self.position_history]
        near_boundary = any(self.angle_handler.is_near_boundary(az) for az in az_values)

        if near_boundary:
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            weights = np.exp(np.linspace(-2, 0, len(self.position_history)))
            weights /= weights.sum()
            smooth_az = sum(az * w for (az, _), w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        el_values = [p[1] for p in self.position_history]
        smooth_el = np.average(el_values, weights=weights)

        return smooth_az, smooth_el

    def update_tracking_confidence(self, found_target, target_strength=0):
        """Update confidence based on tracking success."""
        if found_target:
            if target_strength > self.high_confidence_threshold:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
                self.consecutive_good_tracks += 1
            else:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.1)
                self.consecutive_good_tracks = 0
        else:
            self.tracking_confidence = max(0.0, self.tracking_confidence - 0.3)
            self.consecutive_good_tracks = 0

    def update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update satellite points in shared memory."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()

            print(f"[Tracker] Target: ({azimuth:.1f}°, {elevation:.1f}°) "
                  f"dist={distance:.0f}cm, str={strength:.0f}, conf={self.tracking_confidence:.2f}")
        except Exception as e:
            print(f"[Tracker] Error updating satellite_points: {e}")

    def clear_satellite_points(self):
        """Clear satellite points."""
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
        except Exception as e:
            print(f"[Tracker] Error clearing satellite_points: {e}")

    def demo_determine_orbit_plane(self):
        """Determine the orbit plane from collected points."""
        if len(self.demo_orbit_points) < 3:
            return False

        points = self.demo_orbit_points[-3:]

        el_changes = []
        az_changes = []
        for i in range(len(points) - 1):
            az_diff = self.angle_handler.difference(points[i][0], points[i + 1][0])
            el_diff = points[i + 1][1] - points[i][1]
            az_changes.append(az_diff)
            el_changes.append(el_diff)

        avg_az_change = np.mean(np.abs(az_changes))
        avg_el_change = np.mean(el_changes)

        if avg_az_change > 0:
            self.demo_inclination = avg_el_change / avg_az_change
        else:
            self.demo_inclination = 0.0

        if np.mean(az_changes) > 0:
            self.demo_angular_velocity = abs(self.demo_angular_velocity)
        else:
            self.demo_angular_velocity = -abs(self.demo_angular_velocity)

        self.demo_center_el = np.mean([p[1] for p in points])

        print(f"[Demo] Orbit determined: inclination={self.demo_inclination:.2f}°/°, "
              f"direction={'CW' if self.demo_angular_velocity > 0 else 'CCW'}, "
              f"center_el={self.demo_center_el:.1f}°")

        self.demo_orbit_determined = True
        return True

    def demo_predict_position(self, current_time):
        """Predict drone position based on orbital motion."""
        if self.demo_last_update is None:
            return self.demo_heading, self.demo_center_el

        dt = current_time - self.demo_last_update

        predicted_heading = self.demo_heading + self.demo_angular_velocity * dt
        predicted_heading = self.angle_handler.normalize(predicted_heading)

        if self.demo_inclination != -1 and self.demo_inclination != 0:
            heading_change = self.demo_angular_velocity * dt
            el_change = heading_change * self.demo_inclination
            predicted_el = self.demo_center_el + el_change
            predicted_el += 5 * math.sin(math.radians(predicted_heading))
        else:
            predicted_el = self.demo_center_el

        predicted_el = np.clip(predicted_el, 10, 80)

        return predicted_heading, predicted_el

    def demo_track_orbit(self):
        """Track drone in orbital motion with prediction."""
        current_time = time.time()

        predicted_az, predicted_el = self.demo_predict_position(current_time)

        print(f"[Demo] Predicted position: ({predicted_az:.1f}°, {predicted_el:.1f}°)")

        old_radius_az = self.scan_radius_az
        old_radius_el = self.scan_radius_el
        self.scan_radius_az = DEMO_SCAN_RADIUS
        self.scan_radius_el = DEMO_SCAN_RADIUS

        scan_results = self.tracking_scan(predicted_az, predicted_el)

        self.scan_radius_az = old_radius_az
        self.scan_radius_el = old_radius_el

        if scan_results:
            best = self.find_best_target(scan_results)

            if best:
                actual_az, actual_el, distance, strength = best

                dt = current_time - self.demo_last_update if self.demo_last_update else 0.05

                self.demo_heading = actual_az
                self.demo_last_update = current_time

                if not self.demo_orbit_determined:
                    self.demo_orbit_points.append((actual_az, actual_el))

                    if len(self.demo_orbit_points) >= 3:
                        self.demo_determine_orbit_plane()

                self.update_satellite_points(actual_az, actual_el, distance, strength)

                if len(self.demo_orbit_points) > 1 and dt > 0:
                    prev_az = self.demo_orbit_points[-2][0]
                    actual_velocity = self.angle_handler.difference(prev_az, actual_az) / dt
                    self.demo_angular_velocity = 0.7 * self.demo_angular_velocity + 0.3 * actual_velocity

                print(f"[Demo] Tracking: heading={actual_az:.1f}°, el={actual_el:.1f}°, "
                      f"dist={distance:.0f}cm, velocity={self.demo_angular_velocity:.1f}°/s")

                return True

        return False

    def demo_acquisition(self):
        """Special acquisition for demo mode - look for orbiting drone."""
        print("[Demo] Starting demo acquisition for orbiting drone...")

        start_heading = None
        try:
            if "heading" in self.shared_data and self.shared_data["heading"].value >= 0:
                start_heading = self.shared_data["heading"].value
                print(f"[Demo] Starting with provided heading: {start_heading:.1f}°")
        except:
            print("[Demo] No heading value available")

        try:
            if "inclination" in self.shared_data:
                provided_inclination = self.shared_data["inclination"].value
                if provided_inclination != -1:
                    self.demo_inclination = provided_inclination
                    print(f"[Demo] Using provided inclination: {provided_inclination:.2f}°")
        except:
            print("[Demo] No inclination value available")

        scan_elevations = [45, 35, 55, 25, 65]

        if start_heading is not None:
            scan_azimuths = [
                start_heading,
                start_heading + 10, start_heading - 10,
                start_heading + 20, start_heading - 20,
                start_heading + 30, start_heading - 30
            ]
        else:
            scan_azimuths = np.linspace(0, 350, 12)

        best_target = None
        best_strength = 0

        for el in scan_elevations:
            for az in scan_azimuths:
                if self.shared_data["shutdown"].value:
                    return None

                az = self.angle_handler.normalize(az)

                if not self.move_to_position_verified(az, el):
                    continue

                actual_az, actual_el, distance, strength = self.read_lidar_at_position()

                if DEMO_RADIUS_MIN < distance < DEMO_RADIUS_MAX:
                    if self.clutter_filter.is_valid_target(actual_az, actual_el, distance, strength):
                        if strength >= self.min_strength_threshold:
                            print(f"[Demo] Found potential drone: ({actual_az:.1f}°, {actual_el:.1f}°) "
                                  f"dist={distance:.0f}cm, str={strength:.0f}")

                            if strength > best_strength:
                                best_target = (actual_az, actual_el, distance, strength)
                                best_strength = strength

                            if strength > 150:
                                break

            if best_target and best_strength > 150:
                break

        if best_target:
            self.demo_heading = best_target[0]
            self.demo_center_el = best_target[1]
            self.demo_last_update = time.time()
            self.demo_orbit_points = [(best_target[0], best_target[1])]

            print(f"[Demo] Acquisition successful: drone at ({best_target[0]:.1f}°, {best_target[1]:.1f}°)")

            if self.demo_inclination == -1:
                print("[Demo] Inclination unknown, will determine from motion")
                self.demo_orbit_determined = False
            else:
                self.demo_orbit_determined = True

            return best_target

        print("[Demo] No drone found in expected range")
        return None

    def run(self):
        """Main tracking loop with independent acquisition, debug, and demo modes."""
        print("[Tracker] Starting with independent acquisition, debug, and demo modes")

        try:
            while not self.shared_data["shutdown"].value:
                # Check for demo mode (highest priority)
                if "demo" in self.shared_data and self.shared_data["demo"].value:
                    if not self.demo_mode:
                        print("[Demo] Demo mode activated - tracking orbiting drone")
                        self.demo_mode = True

                        target = self.demo_acquisition()

                        if target:
                            self.current_target_az = target[0]
                            self.current_target_el = target[1]
                            self.tracking_confidence = 0.8
                            self.update_satellite_points(target[0], target[1], target[2], target[3])
                        else:
                            print("[Demo] Failed to acquire drone")
                            self.demo_mode = False
                            self.shared_data["demo"].value = False
                            continue

                    if self.demo_mode:
                        cycle_start = time.time()

                        if self.demo_track_orbit():
                            self.lost_target_count = 0
                        else:
                            self.lost_target_count += 1
                            print(f"[Demo] Drone lost ({self.lost_target_count}/3)")

                            if self.lost_target_count >= 3:
                                print("[Demo] Drone lost, attempting re-acquisition")
                                target = self.demo_acquisition()

                                if target:
                                    self.lost_target_count = 0
                                    self.demo_heading = target[0]
                                    self.demo_last_update = time.time()
                                    print("[Demo] Re-acquired drone")
                                else:
                                    print("[Demo] Re-acquisition failed, exiting demo mode")
                                    self.demo_mode = False
                                    self.shared_data["demo"].value = False

                        cycle_time = time.time() - cycle_start
                        if cycle_time < MIN_CYCLE_TIME:
                            time.sleep(MIN_CYCLE_TIME - cycle_time)

                        continue

                else:
                    if self.demo_mode:
                        print("[Demo] Demo mode deactivated")
                        self.demo_mode = False
                        self.demo_orbit_points = []
                        self.demo_orbit_determined = False

                # Check for acquisition trigger (independent of debug mode)
                if self.shared_data["acquire_points"].value:
                    print("[Acquisition] Triggered")
                    self.shared_data["acquire_points"].value = False

                    target = self.acquisition_scan()

                    if target:
                        self.current_target_az = target[0]
                        self.current_target_el = target[1]
                        self.target_history.clear()
                        self.target_history.append((target[0], target[1]))
                        self.position_history.clear()
                        self.tracking_confidence = 0.5
                        self.update_satellite_points(target[0], target[1], target[2], target[3])

                        print(f"[Acquisition] Success! Target at ({target[0]:.1f}°, {target[1]:.1f}°)")

                        # Optionally enable debug mode after acquisition
                        # self.shared_data["debug_mode"].value = True
                    else:
                        print("[Acquisition] Failed - no target found")
                        self.clear_satellite_points()

                    continue

                # Normal tracking mode (debug mode)
                if not self.shared_data["debug_mode"].value:
                    if self.current_target_az is not None:
                        print("[Tracker] Debug mode disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.clear_satellite_points()
                        self.tracking_confidence = 0.0
                    time.sleep(0.1)
                    continue

                # Debug mode enabled - if no target, start with simple scan at current position
                if self.current_target_az is None:
                    print("[Debug] No target set, starting scan at current position")
                    # Initialize at current position
                    self.current_target_az = self.shared_data["stepper_degrees"].value
                    self.current_target_el = self.shared_data["servo_degrees"].value

                    # If at origin, start at a reasonable position
                    if self.current_target_az == 0 and self.current_target_el == 0:
                        self.current_target_az = 180.0
                        self.current_target_el = 45.0

                    print(
                        f"[Debug] Starting tracking at ({self.current_target_az:.1f}°, {self.current_target_el:.1f}°)")
                    self.tracking_confidence = 0.3  # Start with low confidence

                cycle_start = time.time()

                scan_results = self.tracking_scan(self.current_target_az, self.current_target_el)
                best_target = self.find_best_target(scan_results)

                if best_target:
                    self.lost_target_count = 0
                    self.update_tracking_confidence(True, best_target[3])

                    self.target_history.append((best_target[0], best_target[1]))
                    smooth_az, smooth_el = self.smooth_position(best_target[0], best_target[1])

                    self.current_target_az = smooth_az
                    self.current_target_el = smooth_el

                    self.update_satellite_points(smooth_az, smooth_el,
                                                 best_target[2], best_target[3])

                    if self.tracking_confidence > 0.5:
                        self.scan_radius_az = max(SCAN_RADIUS_AZ, self.scan_radius_az * 0.9)
                        self.scan_radius_el = max(SCAN_RADIUS_EL, self.scan_radius_el * 0.9)

                else:
                    self.lost_target_count += 1
                    self.update_tracking_confidence(False)

                    print(f"[Tracker] Target lost ({self.lost_target_count}/{self.max_lost_count}), "
                          f"confidence={self.tracking_confidence:.2f}")

                    if self.tracking_confidence < 0.2:
                        self.clear_satellite_points()

                    if self.lost_target_count >= self.max_lost_count:
                        self.scan_radius_az = min(self.scan_radius_az * 1.5, MAX_SCAN_RADIUS_AZ)
                        self.scan_radius_el = min(self.scan_radius_el * 1.5, MAX_SCAN_RADIUS_EL)
                        self.lost_target_count = 0
                        print(f"[Tracker] Expanding search to ±{self.scan_radius_az:.1f}°")

                        if self.tracking_confidence < 0.1:
                            print("[Tracker] Lost target, need re-acquisition")
                            self.current_target_az = None
                            self.current_target_el = None

                cycle_time = time.time() - cycle_start
                self.cycle_count += 1

                if self.cycle_count % STATS_PRINT_INTERVAL == 0:
                    success_rate = (self.successful_reads / max(1, self.successful_reads + self.failed_reads)) * 100
                    print(f"[Tracker] Stats: {cycle_time * 1000:.0f}ms cycle, "
                          f"{success_rate:.0f}% position success")

                if cycle_time < MIN_CYCLE_TIME:
                    time.sleep(MIN_CYCLE_TIME - cycle_time)

        except KeyboardInterrupt:
            print("[Tracker] Interrupted")
        except Exception as e:
            print(f"[Tracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Tracker] Shutting down")
            self.clear_satellite_points()


def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """
    Run the robust tracker process with independent acquisition, debug, and demo modes.

    Shared data flags expected:
    - acquire_points: Trigger acquisition scan (NOT required for debug mode)
    - debug_mode: Enable/disable tracking (works without acquisition)
    - demo: Enable demo mode for tracking orbiting drone
    - heading: Initial heading for demo mode (optional, -1 if not provided)
    - inclination: Orbit inclination (-1 if unknown)
    - target_reached: Set by motion system when position reached
    - shutdown: Stop the tracker

    Usage examples:

    1. Direct tracking without acquisition:
       shared_data["debug_mode"].value = True  # Start tracking immediately

    2. Find target then track:
       shared_data["acquire_points"].value = True  # Find target
       # After success...
       shared_data["debug_mode"].value = True  # Enable tracking

    3. Demo mode for orbiting drone:
       shared_data["demo"].value = True  # Track orbiting pattern
    """
    print("[Tracker] Initializing with independent acquisition, debug, and demo modes...")
    print("[Tracker] Debug mode does NOT require acquisition - can start tracking immediately")

    if "demo" not in shared_data:
        print("[Tracker] Warning: 'demo' flag not in shared_data")
    if "heading" not in shared_data:
        print("[Tracker] Warning: 'heading' value not in shared_data")
    if "inclination" not in shared_data:
        print("[Tracker] Warning: 'inclination' value not in shared_data")

    tracker = TargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
    finally:
        print("[Tracker] Process ended")