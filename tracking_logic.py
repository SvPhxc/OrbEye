#!/usr/bin/env python3
"""
Integrated Target Tracker for LiDAR Scanner System
Final version with proper synchronization with hardware controller
Handles acquisition, tracking, and demo modes seamlessly
"""

import time
import math
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Manager, Process
import threading
from collections import deque
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

# System states (must match hardware controller)
class SystemState(Enum):
    IDLE = 0
    MOVING = 1
    SCANNING = 2
    TRACKER_MOVE = 3
    ERROR = 4
    SHUTDOWN = 5
    PAUSED = 6


class Priority(Enum):
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2


# LiDAR parameters
LIDAR_FOV = 2.0  # degrees
LIDAR_MIN_INTERVAL = 0.001  # 1ms (1000Hz max)
MIN_STRENGTH_THRESHOLD = 30
HIGH_CONFIDENCE_THRESHOLD = 100
ACQUISITION_STRENGTH_THRESHOLD = 20

# Clutter filter
ANGULAR_TOLERANCE = 2.0
DISTANCE_MARGIN_CM = 50.0
CACHE_SIZE = 75000
DISABLE_CLUTTER_FOR_ACQUISITION = False

# Movement parameters
MOVEMENT_TIMEOUT = 2.0  # seconds
POSITION_TOLERANCE = 0.0  # degrees
POSITION_VERIFY_DELAY = 0.001
MAX_POSITION_ERROR = 0.0

# Tracking parameters
SCAN_RADIUS_AZ = 8.0
SCAN_RADIUS_EL = 8.0
SCAN_POINTS = 8
MAX_SCAN_RADIUS_AZ = 20.0
MAX_SCAN_RADIUS_EL = 20.0

# Acquisition parameters
ACQUISITION_AZ_RANGE = 60.0
ACQUISITION_AZ_STEP = 4.0
ACQUISITION_ELEVATIONS = [45, 40, 50, 35, 55, 30, 60, 25, 65, 20, 70]
ACQUISITION_MIN_DISTANCE = 10.0
ACQUISITION_MAX_ATTEMPTS = 3

# Demo mode
DEMO_ORBIT_TIME = 20.0
DEMO_RADIUS_MIN = 150.0  # cm
DEMO_RADIUS_MAX = 250.0  # cm
DEMO_CENTER_ELEVATION = 45.0
DEMO_SCAN_RADIUS = 4.0
DEMO_MIN_POINTS_FOR_PREDICTION = 5
DEMO_VELOCITY_SMOOTHING = 0.3
DEMO_PARABOLIC_CHECK_POINTS = 10

# Tracking state
MAX_LOST_COUNT = 5
HISTORY_SIZE = 4
TARGET_HISTORY_SIZE = 3

# Performance
MIN_CYCLE_TIME = 0.02
MIN_DEBUG_CYCLE_TIME = 0.03
STATS_PRINT_INTERVAL = 20


# ============================================================================
# CLUTTER FILTER
# ============================================================================

class ClutterFilter:
    """Efficient clutter filter with caching"""

    def __init__(self, background_file="background_scan.npy"):
        self.angular_tolerance = ANGULAR_TOLERANCE
        self.distance_margin_cm = DISTANCE_MARGIN_CM
        self.background_tree = None
        self.background_data = None
        self._query_cache = {}
        self._cache_size = CACHE_SIZE

        try:
            self.background_data = np.load(background_file)
            print(f"[Filter] Loaded {len(self.background_data)} background points")

            coords = self.background_data[:, [0, 1]]
            self.background_tree = cKDTree(coords, leafsize=16)
            self.bg_distances = self.background_data[:, 2]

        except FileNotFoundError:
            print(f"[Filter] No background file found - clutter filter disabled")
        except Exception as e:
            print(f"[Filter] Error: {e}")

    def is_valid_target(self, azimuth, elevation, distance, strength):
        """Check if measurement is a valid target"""
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

            except Exception:
                return True

        return distance < (bg_distance - self.distance_margin_cm)


# ============================================================================
# ANGLE UTILITIES
# ============================================================================

class AngleHandler:
    """Handle angle wraparound and calculations"""

    @staticmethod
    def normalize(angle):
        """Normalize to [0, 360)"""
        angle = angle % 360
        return angle if angle >= 0 else angle + 360

    @staticmethod
    def difference(angle1, angle2):
        """Shortest angular difference"""
        return ((angle2 - angle1 + 180) % 360) - 180

    @staticmethod
    def shortest_path(current, target):
        """Calculate target angle for minimum rotation"""
        diff = AngleHandler.difference(current, target)
        return AngleHandler.normalize(current + diff)

    @staticmethod
    def circular_mean(angles):
        """Mean of angles with wraparound"""
        if not angles:
            return None
        x_sum = sum(math.cos(math.radians(a)) for a in angles)
        y_sum = sum(math.sin(math.radians(a)) for a in angles)
        return AngleHandler.normalize(math.degrees(math.atan2(y_sum, x_sum)))

    @staticmethod
    def clamp_scan_range(center, radius):
        """Prevent scan from exceeding 360°"""
        return min(radius, 180.0)


# ============================================================================
# TARGET TRACKER
# ============================================================================

class TargetTracker:
    """Main tracker with integrated movement control"""

    def __init__(self, shared_data, background_file="background_scan.npy"):
        self.shared_data = shared_data
        self.clutter_filter = ClutterFilter(background_file)
        self.angle_handler = AngleHandler()

        # Tracking state
        self.current_target_az = None
        self.current_target_el = None
        self.tracking_confidence = 0.0
        self.consecutive_good_tracks = 0

        # Demo mode
        self.demo_mode = False
        self.demo_heading = 0.0
        self.demo_inclination = -1
        self.demo_angular_velocity = 360.0 / DEMO_ORBIT_TIME
        self.demo_center_el = DEMO_CENTER_ELEVATION
        self.demo_last_update = None
        self.demo_orbit_points = []
        self.demo_can_predict = False
        self.demo_motion_type = "unknown"
        self.demo_velocity_history = deque(maxlen=5)

        # History
        self.target_history = deque(maxlen=TARGET_HISTORY_SIZE)
        self.position_history = deque(maxlen=HISTORY_SIZE)
        self.lost_target_count = 0

        # Performance
        self.cycle_count = 0
        self.successful_reads = 0
        self.failed_reads = 0
        self.last_lidar_read = 0

        # Movement tracking
        self.last_request_id = 0

        print("[Tracker] Initialized with integrated movement control")
        print(f"[Tracker] FOV: {LIDAR_FOV}°, Tolerances: {POSITION_TOLERANCE}°")

    def request_movement(self, azimuth, elevation, priority=Priority.HIGH):
        """Request movement from hardware controller"""

        # Check system state
        max_wait = 3.0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            with self.shared_data["state_lock"]:
                current_state = SystemState(self.shared_data["system_state"].value)

                if current_state == SystemState.IDLE:
                    # Claim the system
                    self.shared_data["system_state"].value = SystemState.TRACKER_MOVE.value
                    break
                elif current_state == SystemState.SCANNING:
                    # Request scan pause
                    self.shared_data["background_scan_paused"].value = True
                    time.sleep(0.01)
                elif current_state in [SystemState.ERROR, SystemState.SHUTDOWN]:
                    return False
                else:
                    time.sleep(0.01)
        else:
            print(f"[Tracker] Timeout waiting for system availability")
            return False

        try:
            # Generate request
            self.last_request_id += 1
            request_id = self.last_request_id

            # Set movement parameters
            with self.shared_data["movement_lock"]:
                self.shared_data["movement_request_id"].value = request_id
                self.shared_data["target_azimuth"].value = azimuth
                self.shared_data["target_elevation"].value = elevation
                self.shared_data["movement_priority"].value = priority.value
                self.shared_data["go_to_target"].value = True

            # Wait for completion
            return self._wait_for_movement(request_id)

        finally:
            # Release system state if we still own it
            with self.shared_data["state_lock"]:
                if self.shared_data["system_state"].value == SystemState.TRACKER_MOVE.value:
                    self.shared_data["system_state"].value = SystemState.IDLE.value

    def _wait_for_movement(self, request_id):
        """Wait for movement completion"""
        start_time = time.time()

        while time.time() - start_time < MOVEMENT_TIMEOUT:
            if self.shared_data["shutdown"].value:
                return False

            # Check if our request was completed
            if self.shared_data["movement_complete_id"].value >= request_id:
                # Verify position
                time.sleep(POSITION_VERIFY_DELAY)

                actual_az = self.shared_data["stepper_degrees"].value
                actual_el = self.shared_data["servo_degrees"].value
                target_az = self.shared_data["target_azimuth"].value
                target_el = self.shared_data["target_elevation"].value

                az_error = abs(self.angle_handler.difference(actual_az, target_az))
                el_error = abs(actual_el - target_el)

                if az_error <= MAX_POSITION_ERROR and el_error <= MAX_POSITION_ERROR:
                    self.successful_reads += 1
                    return True
                else:
                    print(f"[Tracker] Position error: az={az_error:.1f}°, el={el_error:.1f}°")
                    self.failed_reads += 1
                    return False

            time.sleep(0.002)

        print(f"[Tracker] Movement timeout for request {request_id}")
        self.failed_reads += 1
        return False

    def read_lidar_verified(self):
        """Read LiDAR data with position verification"""

        # Rate limiting
        elapsed = time.time() - self.last_lidar_read
        if elapsed < LIDAR_MIN_INTERVAL:
            time.sleep(LIDAR_MIN_INTERVAL - elapsed)

        # Get expected position
        expected_az = self.shared_data["target_azimuth"].value
        expected_el = self.shared_data["target_elevation"].value

        # Wait for fresh data at this position
        start_time = time.time()
        last_timestamp = 0

        while time.time() - start_time < 0.1:  # 100ms timeout
            with self.shared_data["lidar_lock"]:
                timestamp = self.shared_data["lidar_data"][2]

                if timestamp > last_timestamp and self.shared_data["lidar_valid"].value:
                    # Check position match
                    lidar_az = self.shared_data["lidar_position"][0]
                    lidar_el = self.shared_data["lidar_position"][1]

                    az_error = abs(self.angle_handler.difference(lidar_az, expected_az))
                    el_error = abs(lidar_el - expected_el)

                    if az_error < POSITION_TOLERANCE and el_error < POSITION_TOLERANCE:
                        # Data is valid
                        distance = self.shared_data["lidar_data"][0]
                        strength = self.shared_data["lidar_data"][1]
                        self.last_lidar_read = time.time()
                        return lidar_az, lidar_el, distance, strength

                last_timestamp = timestamp

            time.sleep(0.001)

        return None, None, None, None

    def acquisition_scan(self):
        """Wide scan to find target"""
        print("[Acquisition] Starting with integrated movement...")

        all_targets = []
        start_az = self.shared_data["stepper_degrees"].value

        # Build scan pattern
        az_points = [start_az]
        offset = ACQUISITION_AZ_STEP
        max_offset = min(ACQUISITION_AZ_RANGE / 2, 180)

        while offset <= max_offset:
            if 2 * offset <= 360:
                az_points.append(start_az + offset)
                az_points.append(start_az - offset)
            offset += ACQUISITION_AZ_STEP

        az_points = [self.angle_handler.normalize(az) for az in az_points]

        # Remove duplicates
        seen = set()
        az_points = [x for x in az_points if not (x in seen or seen.add(x))]

        print(f"[Acquisition] Scanning {len(az_points)} x {len(ACQUISITION_ELEVATIONS)} points")

        for el_idx, elevation in enumerate(ACQUISITION_ELEVATIONS):
            if self.shared_data["shutdown"].value:
                break

            elevation = np.clip(elevation, 10, 80)

            # Zigzag pattern
            if el_idx % 2 == 1:
                current_az_points = list(reversed(az_points))
            else:
                current_az_points = az_points

            for azimuth in current_az_points:
                if self.shared_data["shutdown"].value:
                    break

                # Request movement
                if not self.request_movement(azimuth, elevation, Priority.HIGH):
                    continue

                # Multiple read attempts
                for attempt in range(ACQUISITION_MAX_ATTEMPTS):
                    if attempt > 0:
                        time.sleep(0.002)

                    actual_az, actual_el, distance, strength = self.read_lidar_verified()

                    if distance and distance > ACQUISITION_MIN_DISTANCE:
                        if DISABLE_CLUTTER_FOR_ACQUISITION:
                            is_valid = True
                        else:
                            is_valid = self.clutter_filter.is_valid_target(
                                actual_az, actual_el, distance, strength)

                        if is_valid and strength >= ACQUISITION_STRENGTH_THRESHOLD:
                            print(f"[Acquisition] Target: ({actual_az:.1f}°, {actual_el:.1f}°) "
                                  f"d={distance:.0f}cm, s={strength:.0f}")
                            all_targets.append((actual_az, actual_el, distance, strength))

                            if strength >= MIN_STRENGTH_THRESHOLD:
                                return (actual_az, actual_el, distance, strength)

        # Return best target
        if all_targets:
            all_targets.sort(key=lambda x: x[3], reverse=True)
            best = all_targets[0]
            print(f"[Acquisition] Best target: strength={best[3]:.0f}")
            return best

        print("[Acquisition] No targets found")
        return None

    def tracking_scan(self, center_az, center_el):
        """Scan around current target"""
        scan_results = []
        scan_start = time.time()

        # Adjust radius based on confidence
        if self.tracking_confidence > 0.7:
            radius_az = SCAN_RADIUS_AZ * 0.7
            radius_el = SCAN_RADIUS_EL * 0.7
        else:
            radius_az = SCAN_RADIUS_AZ
            radius_el = SCAN_RADIUS_EL

        # Clamp radius
        radius_az = self.angle_handler.clamp_scan_range(center_az, radius_az)
        radius_el = min(radius_el, 40)

        # Calculate points
        points_to_scan = max(3, min(8, int(2 * math.pi * radius_az / LIDAR_FOV)))

        # Scan center first
        if self.request_movement(center_az, center_el, Priority.HIGH):
            az, el, dist, strength = self.read_lidar_verified()
            if dist and dist > 0:
                if self.clutter_filter.is_valid_target(az, el, dist, strength):
                    if strength >= MIN_STRENGTH_THRESHOLD:
                        scan_results.append((az, el, dist, strength))
                        if strength > HIGH_CONFIDENCE_THRESHOLD * 1.5:
                            return scan_results

        # Scan surrounding points
        max_scan_time = 0.5 if self.shared_data["debug_mode"].value else 0.8

        for i in range(points_to_scan):
            if self.shared_data["shutdown"].value or time.time() - scan_start > max_scan_time:
                break

            angle = (2 * math.pi * i) / points_to_scan
            scan_az = center_az + radius_az * math.cos(angle)
            scan_el = center_el + radius_el * math.sin(angle)
            scan_el = np.clip(scan_el, 0, 90)
            scan_az = self.angle_handler.normalize(scan_az)

            if not self.request_movement(scan_az, scan_el, Priority.HIGH):
                continue

            az, el, dist, strength = self.read_lidar_verified()

            if dist and dist > 0:
                if self.clutter_filter.is_valid_target(az, el, dist, strength):
                    if strength >= MIN_STRENGTH_THRESHOLD:
                        scan_results.append((az, el, dist, strength))

        return scan_results

    def find_best_target(self, scan_results):
        """Select best target from scan"""
        if not scan_results:
            return None

        # Sort by strength
        scan_results.sort(key=lambda x: x[3], reverse=True)

        # If we have history, prefer consistent targets
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
        """Smooth target position"""
        self.position_history.append((new_az, new_el))

        if len(self.position_history) < 2:
            return new_az, new_el

        # Check for angle wraparound
        az_values = [p[0] for p in self.position_history]

        if any(abs(self.angle_handler.difference(az_values[i], az_values[i + 1])) > 180
               for i in range(len(az_values) - 1)):
            # Use circular mean
            smooth_az = self.angle_handler.circular_mean(az_values)
        else:
            # Weighted average
            weights = np.exp(np.linspace(-2, 0, len(self.position_history)))
            weights /= weights.sum()
            smooth_az = sum(az * w for (az, _), w in zip(self.position_history, weights))
            smooth_az = self.angle_handler.normalize(smooth_az)

        el_values = [p[1] for p in self.position_history]
        smooth_el = np.average(el_values, weights=weights[-len(el_values):])

        return smooth_az, smooth_el

    def update_tracking_confidence(self, found_target, strength=0):
        """Update tracking confidence"""
        if found_target:
            if strength > HIGH_CONFIDENCE_THRESHOLD:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
                self.consecutive_good_tracks += 1
            else:
                self.tracking_confidence = min(1.0, self.tracking_confidence + 0.1)
                self.consecutive_good_tracks = 0
        else:
            self.tracking_confidence = max(0.0, self.tracking_confidence - 0.3)
            self.consecutive_good_tracks = 0

    def update_satellite_points(self, azimuth, elevation, distance, strength):
        """Update tracking results in shared memory"""
        try:
            with self.shared_data["satellite_points"].get_lock():
                self.shared_data["satellite_points"][0] = azimuth
                self.shared_data["satellite_points"][1] = elevation
                self.shared_data["satellite_points"][2] = distance
                self.shared_data["satellite_points"][3] = strength
                self.shared_data["satellite_points"][4] = time.time()

            print(f"[Track] ({azimuth:.1f}°, {elevation:.1f}°) "
                  f"d={distance:.0f}cm s={strength:.0f} c={self.tracking_confidence:.2f}")
        except Exception as e:
            print(f"[Tracker] Error updating points: {e}")

    def clear_satellite_points(self):
        """Clear tracking results"""
        try:
            with self.shared_data["satellite_points"].get_lock():
                for i in range(5):
                    self.shared_data["satellite_points"][i] = 0.0
        except Exception as e:
            print(f"[Tracker] Error clearing points: {e}")

    def demo_track_orbit(self):
        """Track orbiting drone"""
        current_time = time.time()

        # Predict position if possible
        if self.demo_can_predict and self.demo_last_update:
            dt = current_time - self.demo_last_update
            dt = min(dt, 0.5)  # Limit prediction

            predicted_az = self.demo_heading + self.demo_angular_velocity * dt
            predicted_az = self.angle_handler.normalize(predicted_az)
            predicted_el = self.demo_center_el
        else:
            predicted_az = self.demo_heading
            predicted_el = self.demo_center_el

        # Tighter scan for demo
        old_radius = SCAN_RADIUS_AZ
        SCAN_RADIUS_AZ_LOCAL = DEMO_SCAN_RADIUS if self.demo_can_predict else DEMO_SCAN_RADIUS * 2

        scan_results = self.tracking_scan(predicted_az, predicted_el)

        if scan_results:
            best = self.find_best_target(scan_results)

            if best:
                actual_az, actual_el, distance, strength = best

                # Update orbit tracking
                self.demo_orbit_points.append((actual_az, actual_el, current_time))

                # Keep recent points only
                cutoff = current_time - 30.0
                self.demo_orbit_points = [(az, el, t) for az, el, t in self.demo_orbit_points
                                          if t > cutoff]

                # Update heading
                self.demo_heading = actual_az
                self.demo_last_update = current_time

                # Analyze motion
                if len(self.demo_orbit_points) >= DEMO_MIN_POINTS_FOR_PREDICTION:
                    self._analyze_demo_motion()

                self.update_satellite_points(actual_az, actual_el, distance, strength)
                return True

        return False

    def _analyze_demo_motion(self):
        """Analyze demo drone motion pattern"""
        if len(self.demo_orbit_points) < DEMO_MIN_POINTS_FOR_PREDICTION:
            return

        # Calculate velocities
        velocities = []
        for i in range(len(self.demo_orbit_points) - 1):
            p1 = self.demo_orbit_points[i]
            p2 = self.demo_orbit_points[i + 1]

            az_diff = self.angle_handler.difference(p1[0], p2[0])
            dt = p2[2] - p1[2]

            if dt > 0:
                velocities.append(az_diff / dt)

        if velocities:
            # Update angular velocity
            avg_velocity = np.mean(velocities)
            self.demo_angular_velocity = (1 - DEMO_VELOCITY_SMOOTHING) * self.demo_angular_velocity + \
                                         DEMO_VELOCITY_SMOOTHING * avg_velocity

            # Determine motion type
            velocity_std = np.std(velocities)
            if velocity_std < 2.0:
                self.demo_motion_type = "circular"
            else:
                self.demo_motion_type = "complex"

            self.demo_can_predict = True

            print(f"[Demo] Motion: {self.demo_motion_type}, vel={self.demo_angular_velocity:.1f}°/s")

    def run(self):
        """Main tracking loop"""
        print("[Tracker] Starting integrated tracking system")

        try:
            while not self.shared_data["shutdown"].value:
                cycle_start = time.time()

                # Check demo mode
                if self.shared_data["demo"].value:
                    if not self.demo_mode:
                        print("[Demo] Activated")
                        self.demo_mode = True

                        # Demo acquisition
                        target = self.acquisition_scan()
                        if target:
                            self.demo_heading = target[0]
                            self.demo_center_el = target[1]
                            self.demo_last_update = time.time()
                            self.demo_orbit_points = [(target[0], target[1], time.time())]
                            self.tracking_confidence = 0.8
                            self.update_satellite_points(*target)
                        else:
                            print("[Demo] Acquisition failed")
                            self.demo_mode = False
                            self.shared_data["demo"].value = False
                            continue

                    # Track demo
                    if self.demo_track_orbit():
                        self.lost_target_count = 0
                    else:
                        self.lost_target_count += 1
                        if self.lost_target_count >= 3:
                            print("[Demo] Lost drone, reacquiring...")
                            target = self.acquisition_scan()
                            if target:
                                self.demo_heading = target[0]
                                self.demo_center_el = target[1]
                                self.demo_last_update = time.time()
                                self.lost_target_count = 0
                            else:
                                print("[Demo] Reacquisition failed")
                                self.demo_mode = False
                                self.shared_data["demo"].value = False

                # Check acquisition trigger
                elif self.shared_data["acquire_points"].value:
                    print("[Acquisition] Triggered")
                    self.shared_data["acquire_points"].value = False

                    target = self.acquisition_scan()
                    if target:
                        self.current_target_az = target[0]
                        self.current_target_el = target[1]
                        self.target_history.clear()
                        self.target_history.append((target[0], target[1]))
                        self.tracking_confidence = 0.5
                        self.update_satellite_points(*target)
                        print(f"[Acquisition] Success at ({target[0]:.1f}°, {target[1]:.1f}°)")
                    else:
                        print("[Acquisition] Failed")
                        self.clear_satellite_points()

                # Debug mode tracking
                elif self.shared_data["debug_mode"].value:
                    if self.current_target_az is None:
                        self.current_target_az = self.shared_data["stepper_degrees"].value
                        self.current_target_el = self.shared_data["servo_degrees"].value

                        if self.current_target_az == 0 and self.current_target_el == 0:
                            self.current_target_az = 180.0
                            self.current_target_el = 45.0

                        self.tracking_confidence = 0.3

                    # Track target
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
                    else:
                        self.lost_target_count += 1
                        self.update_tracking_confidence(False)

                        if self.lost_target_count >= MAX_LOST_COUNT:
                            # Expand search
                            SCAN_RADIUS_AZ_NEW = min(SCAN_RADIUS_AZ * 1.5, MAX_SCAN_RADIUS_AZ)
                            SCAN_RADIUS_EL_NEW = min(SCAN_RADIUS_EL * 1.5, MAX_SCAN_RADIUS_EL)
                            self.lost_target_count = 0

                            if self.tracking_confidence < 0.1:
                                print("[Track] Lost, need reacquisition")
                                self.current_target_az = None
                                self.current_target_el = None
                                self.clear_satellite_points()
                else:
                    # Idle
                    if self.current_target_az is not None:
                        print("[Track] Debug mode disabled")
                        self.current_target_az = None
                        self.current_target_el = None
                        self.clear_satellite_points()
                    time.sleep(0.1)

                # Performance stats
                self.cycle_count += 1
                if self.cycle_count % STATS_PRINT_INTERVAL == 0:
                    cycle_time = time.time() - cycle_start
                    success_rate = (self.successful_reads /
                                    max(1, self.successful_reads + self.failed_reads)) * 100
                    print(f"[Stats] {cycle_time * 1000:.0f}ms cycle, "
                          f"{success_rate:.0f}% success, "
                          f"conf={self.tracking_confidence:.2f}")

                # Cycle timing
                cycle_time = time.time() - cycle_start
                target_time = MIN_DEBUG_CYCLE_TIME if self.shared_data["debug_mode"].value else MIN_CYCLE_TIME
                if cycle_time < target_time:
                    time.sleep(target_time - cycle_time)

        except KeyboardInterrupt:
            print("[Tracker] Interrupted")
        except Exception as e:
            print(f"[Tracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Tracker] Shutting down")
            self.clear_satellite_points()


# ============================================================================
# PROCESS ENTRY POINT
# ============================================================================

def run_tracker_process(shared_data, background_file="background_scan.npy"):
    """Run the tracker process"""
    print("=" * 60)
    print("Target Tracker - Integrated Version")
    print(f"FOV: {LIDAR_FOV}°, Movement timeout: {MOVEMENT_TIMEOUT}s")
    print(f"Acquisition: ±{ACQUISITION_AZ_RANGE / 2}° range")
    print(f"Demo mode: {DEMO_MIN_POINTS_FOR_PREDICTION} points before prediction")
    print("=" * 60)

    # Verify shared data has required keys
    required_keys = [
        "shutdown", "system_state", "state_lock", "movement_lock",
        "stepper_degrees", "servo_degrees", "target_azimuth", "target_elevation",
        "go_to_target", "movement_request_id", "movement_complete_id",
        "acquire_points", "debug_mode", "demo", "satellite_points",
        "lidar_data", "lidar_position", "lidar_valid", "lidar_lock"
    ]

    missing = [k for k in required_keys if k not in shared_data]
    if missing:
        print(f"[Tracker] ERROR: Missing shared data keys: {missing}")
        print("[Tracker] Cannot start - incompatible shared data structure")
        return

    tracker = TargetTracker(shared_data, background_file)

    try:
        tracker.run()
    except Exception as e:
        print(f"[Tracker] Process error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Tracker] Process ended")





