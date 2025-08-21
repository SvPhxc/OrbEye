import time
import math
import numpy as np
from collections import deque


class SpiralSearchCorrector:
    """
    Performs spiral search at waypoints and applies corrections to future points
    based on detected offsets from predictions.
    """

    def __init__(self, shared_data, max_spiral_radius=5.0, spiral_step=1.0):
        """
        Args:
            shared_data: Shared memory data structure
            max_spiral_radius: Maximum spiral search radius in degrees
            spiral_step: Step size between spiral rings in degrees
        """
        self.shared_data = shared_data
        self.max_spiral_radius = max_spiral_radius
        self.spiral_step = spiral_step

        # Correction tracking
        self.correction_history = deque(maxlen=5)  # Keep last 5 corrections
        self.current_az_correction = 0.0
        self.current_el_correction = 0.0
        self.correction_confidence = 0.0

        # Detection parameters
        self.min_strength_threshold = 100  # Minimum strength to consider valid
        self.high_strength_threshold = 500  # High confidence detection

    def spiral_search(self, center_az, center_el, detector_func, max_time=3.0):
        """
        Perform an efficient spiral search around a center point.
        Returns the offset from center where target was found.

        Args:
            center_az: Center azimuth in degrees
            center_el: Center elevation in degrees
            detector_func: Function(az, el, dist, strength) -> bool
            max_time: Maximum search time in seconds

        Returns:
            (found, az_offset, el_offset, strength) or (False, 0, 0, 0)
        """
        start_time = time.time()
        best_detection = None
        best_strength = 0

        # Start at center
        if self._check_position(center_az, center_el, detector_func):
            dist, strength, _ = self._get_lidar_reading()
            if strength > self.min_strength_threshold:
                return (True, 0.0, 0.0, strength)

        # Spiral outward in increasingly larger squares
        radius = 0.0
        while radius <= self.max_spiral_radius and (time.time() - start_time) < max_time:
            if self.shared_data["shutdown"].value:
                break

            radius += self.spiral_step
            points_in_ring = max(4, int(8 * radius / self.spiral_step))  # More points for larger radii

            for i in range(points_in_ring):
                if time.time() - start_time > max_time:
                    break

                # Calculate spiral point using parametric equations
                angle = (2 * math.pi * i) / points_in_ring
                az_offset = radius * math.cos(angle)
                el_offset = radius * math.sin(angle) * 0.7  # Compress elevation slightly

                test_az = center_az + az_offset
                test_el = np.clip(center_el + el_offset, 0, 90)

                # Move and check
                if self._move_to_position(test_az, test_el):
                    dist, strength, _ = self._get_lidar_reading()

                    # Check with detector function
                    if detector_func(test_az, test_el, dist, strength):
                        # Found a valid target
                        if strength > best_strength:
                            best_detection = (True, az_offset, el_offset, strength)
                            best_strength = strength

                            # If we found a high-confidence target, return immediately
                            if strength > self.high_strength_threshold:
                                print(f"[Spiral] High-confidence detection at offset "
                                      f"({az_offset:.2f}°, {el_offset:.2f}°), strength={strength:.0f}")
                                return best_detection

        # Return best detection found, or failure
        if best_detection:
            print(f"[Spiral] Best detection at offset ({best_detection[1]:.2f}°, {best_detection[2]:.2f}°), "
                  f"strength={best_detection[3]:.0f}")
            return best_detection

        return (False, 0.0, 0.0, 0.0)

    def update_correction(self, expected_az, expected_el, actual_az, actual_el, strength):
        """
        Update the correction factors based on detected vs expected position.
        Uses weighted average based on signal strength.
        """
        az_error = actual_az - expected_az
        el_error = actual_el - expected_el

        # Weight by signal strength
        weight = min(1.0, strength / self.high_strength_threshold)

        # Add to history with weight
        self.correction_history.append({
            'az_error': az_error,
            'el_error': el_error,
            'weight': weight,
            'time': time.time()
        })

        # Calculate weighted moving average
        if len(self.correction_history) > 0:
            total_weight = sum(c['weight'] for c in self.correction_history)
            if total_weight > 0:
                # Weighted average with time decay
                now = time.time()
                weighted_az = 0.0
                weighted_el = 0.0
                weight_sum = 0.0

                for correction in self.correction_history:
                    age = now - correction['time']
                    time_weight = math.exp(-age / 30.0)  # 30 second decay
                    combined_weight = correction['weight'] * time_weight

                    weighted_az += correction['az_error'] * combined_weight
                    weighted_el += correction['el_error'] * combined_weight
                    weight_sum += combined_weight

                if weight_sum > 0:
                    self.current_az_correction = weighted_az / weight_sum
                    self.current_el_correction = weighted_el / weight_sum
                    self.correction_confidence = min(1.0, weight_sum / len(self.correction_history))

                    print(f"[Correction] Updated: az={self.current_az_correction:.2f}°, "
                          f"el={self.current_el_correction:.2f}°, confidence={self.correction_confidence:.2f}")

    def apply_correction(self, target_az, target_el):
        """
        Apply the current correction to a target position.
        Correction strength depends on confidence.
        """
        # Apply correction scaled by confidence
        corrected_az = target_az + (self.current_az_correction * self.correction_confidence)
        corrected_el = target_el + (self.current_el_correction * self.correction_confidence)

        # Ensure within bounds
        corrected_el = np.clip(corrected_el, 0, 90)
        corrected_az = corrected_az % 360.0

        return corrected_az, corrected_el

    def _move_to_position(self, az, el):
        """Move to position and wait for arrival."""
        self.shared_data["target_azimuth"].value = float(az)
        self.shared_data["target_elevation"].value = float(el)
        self.shared_data["go_to_target"].value = True

        # Wait for movement to complete (simplified)
        timeout = 0.5  # Quick movements for spiral
        start = time.time()
        while time.time() - start < timeout:
            if self.shared_data["target_reached"].value:
                time.sleep(0.01)  # Brief settle time
                return True
            time.sleep(0.001)
        return False

    def _check_position(self, az, el, detector_func):
        """Move to position and check with detector."""
        if self._move_to_position(az, el):
            dist, strength, _ = self._get_lidar_reading()
            return detector_func(az, el, dist, strength)
        return False

    def _get_lidar_reading(self):
        """Get current LiDAR reading."""
        # Brief wait to ensure fresh data
        time.sleep(0.002)  # 2ms for fresh reading at 1000Hz

        with self.shared_data["lidar_data"].get_lock():
            dist = float(self.shared_data["lidar_data"][0])
            strength = float(self.shared_data["lidar_data"][1])
            ts = float(self.shared_data["lidar_data"][2])

        return dist, strength, ts


def enhanced_patrol_waypoints(shared_data, waypoints, clutter_filter=None,
                              spiral_corrector=None, **kwargs):
    """
    Enhanced patrol with spiral search and correction at each waypoint.

    Args:
        shared_data: Shared memory structure
        waypoints: List of (az, el) tuples
        clutter_filter: Optional ClutterFilter instance
        spiral_corrector: SpiralSearchCorrector instance
        **kwargs: Additional arguments for patrol
    """
    if spiral_corrector is None:
        spiral_corrector = SpiralSearchCorrector(shared_data)

    # Create detector function
    def detector(az, el, dist, strength):
        # Basic detection logic
        if dist < 10 or dist > 16000:
            return False
        if strength < spiral_corrector.min_strength_threshold:
            return False

        # Use clutter filter if available
        if clutter_filter:
            try:
                # Assuming clutter_filter has is_foreground or similar method
                if hasattr(clutter_filter, 'is_foreground'):
                    return clutter_filter.is_foreground(az, el, dist, strength)
                elif hasattr(clutter_filter, 'is_target'):
                    return clutter_filter.is_target(az, el, dist, strength)
            except:
                pass

        return True  # Accept if passes basic checks

    print(f"[EnhancedPatrol] Starting patrol of {len(waypoints)} waypoints with spiral search")

    # Track successful detections for orbit fitting
    detection_history = []

    for i, (original_az, original_el) in enumerate(waypoints):
        if shared_data["shutdown"].value:
            break

        # Apply correction to this waypoint
        target_az, target_el = spiral_corrector.apply_correction(original_az, original_el)

        print(f"[EnhancedPatrol] Waypoint {i + 1}/{len(waypoints)}: "
              f"Original({original_az:.1f}°, {original_el:.1f}°) -> "
              f"Corrected({target_az:.1f}°, {target_el:.1f}°)")

        # Perform spiral search
        found, az_offset, el_offset, strength = spiral_corrector.spiral_search(
            target_az, target_el, detector, max_time=2.0
        )

        if found:
            # Calculate actual detection position
            actual_az = target_az + az_offset
            actual_el = target_el + el_offset

            print(f"[EnhancedPatrol] Detection at ({actual_az:.1f}°, {actual_el:.1f}°), "
                  f"strength={strength:.0f}")

            # Update correction for future waypoints
            spiral_corrector.update_correction(
                original_az, original_el,
                actual_az, actual_el,
                strength
            )

            # Store detection
            detection_history.append({
                'az': actual_az,
                'el': actual_el,
                'strength': strength,
                'time': time.time(),
                'waypoint_idx': i
            })

            # Update satellite points for visualization
            try:
                with shared_data["satellite_points"].get_lock():
                    shared_data["satellite_points"][:] = [
                        actual_az, actual_el,
                        float(shared_data["lidar_data"][0]),  # distance
                        strength, time.time()
                    ]

                # Also append to tracking history if recording
                if shared_data.get("record_tle_points") and shared_data["record_tle_points"].value:
                    shared_data["tracking_history"].append([
                        actual_az, actual_el,
                        float(shared_data["lidar_data"][0]),
                        strength, time.time()
                    ])
            except Exception as e:
                print(f"[EnhancedPatrol] Error updating points: {e}")

            # Set satellite detected flag
            if shared_data.get("satellite_detected"):
                shared_data["satellite_detected"].value = True
        else:
            print(f"[EnhancedPatrol] No detection at waypoint {i + 1}")

            # Reduce correction confidence if we're missing detections
            spiral_corrector.correction_confidence *= 0.8

    # Return to first waypoint with corrections
    if waypoints and not shared_data["shutdown"].value:
        first_az, first_el = spiral_corrector.apply_correction(waypoints[0][0], waypoints[0][1])
        spiral_corrector._move_to_position(first_az, first_el)

    print(f"[EnhancedPatrol] Patrol complete. {len(detection_history)} detections made.")
    return detection_history


def run_enhanced_orbit_patrol_from_xyz(shared_data, pts_xyz,
                                       num_points=9,
                                       spiral_radius=3.0,
                                       **kwargs):
    """
    Run orbit patrol with spiral search enhancement.

    Args:
        shared_data: Shared memory structure
        pts_xyz: Nx3 array of orbit points in km
        num_points: Number of waypoints to use
        spiral_radius: Maximum spiral search radius in degrees
        **kwargs: Additional arguments
    """
    from orbit_patrol_from_xyz import (
        xyz_to_azel_center, filter_by_elevation,
        select_evenly_spaced, _az_short_diff
    )

    # Convert to az/el
    azel_path = xyz_to_azel_center(pts_xyz)

    # Filter by elevation
    min_el = kwargs.get('min_el_deg', 0.0)
    max_el = kwargs.get('max_el_deg', 60.0)
    visible = filter_by_elevation(azel_path, min_el=min_el, max_el=max_el)

    if not visible:
        print("[EnhancedPatrol] No points in elevation window.")
        return

    # Select waypoints
    waypoints = select_evenly_spaced(visible, num_points)

    # Start near current position if requested
    if kwargs.get('start_near_current', True):
        try:
            cur_az = float(shared_data["stepper_degrees"].value) % 360.0
            cur_el = float(shared_data["servo_degrees"].value)
            start = min(range(len(waypoints)),
                        key=lambda i: abs(_az_short_diff(cur_az, waypoints[i][0])) + abs(cur_el - waypoints[i][1]))
            waypoints = waypoints[start:] + waypoints[:start]
        except:
            pass

    print(f"[EnhancedPatrol] Starting enhanced patrol with {len(waypoints)} waypoints")

    # Create spiral corrector
    corrector = SpiralSearchCorrector(shared_data, max_spiral_radius=spiral_radius)

    # Create clutter filter if available
    clutter_filter = None
    if kwargs.get('clutter_filter'):
        clutter_filter = kwargs['clutter_filter']

    # Run enhanced patrol
    detections = enhanced_patrol_waypoints(
        shared_data, waypoints,
        clutter_filter=clutter_filter,
        spiral_corrector=corrector,
        **kwargs
    )

    return detections