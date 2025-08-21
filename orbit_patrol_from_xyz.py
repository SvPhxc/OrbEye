import time
import math
import numpy as np
from collections import deque

# Import the original functions we need
from orbit_patrol_from_xyz import (
    xyz_to_azel_center, filter_by_elevation, select_evenly_spaced,
    _az_short_diff, approx_arc_deg, _goto_and_wait, _should_abort,
    _bind_cf_detector, make_detection_proceed_condition
)


class SpiralSearchEnhancer:
    """
    Adds spiral search and correction capabilities to orbit patrol.
    Designed to work with existing proceed_condition mechanism.
    """

    def __init__(self, shared_data, enable_spiral=True, spiral_radius=3.0, spiral_step=0.5):
        self.shared_data = shared_data
        self.enable_spiral = enable_spiral
        self.spiral_radius = spiral_radius
        self.spiral_step = spiral_step

        # Correction tracking
        self.correction_history = deque(maxlen=5)
        self.az_correction = 0.0
        self.el_correction = 0.0
        self.confidence = 0.0

    def spiral_search_at_waypoint(self, az, el, proceed_condition, deadline_s):
        """
        Performs spiral search using the existing proceed_condition.
        Returns (found, actual_az, actual_el, offset_az, offset_el).
        """
        if not self.enable_spiral:
            # Just check at the given position
            found = proceed_condition(az, el, deadline_s)
            return (found, az, el, 0.0, 0.0)

        # First check center point
        center_deadline = time.monotonic() + 0.5  # Quick check at center
        if proceed_condition(az, el, center_deadline):
            return (True, az, el, 0.0, 0.0)

        # Spiral outward
        radius = 0.0
        best_position = None

        while radius <= self.spiral_radius and time.monotonic() < deadline_s:
            if _should_abort(self.shared_data):
                break

            radius += self.spiral_step
            points_in_ring = max(4, int(8 * radius / self.spiral_step))

            for i in range(points_in_ring):
                if time.monotonic() >= deadline_s:
                    break

                angle = (2 * math.pi * i) / points_in_ring
                az_offset = radius * math.cos(angle)
                el_offset = radius * math.sin(angle) * 0.7  # Compress elevation

                test_az = (az + az_offset) % 360.0
                test_el = np.clip(el + el_offset, 0, 90)

                # Move to test position
                if _goto_and_wait(self.shared_data, test_az, test_el, settle_timeout_s=0.3):
                    # Check with proceed condition (quick timeout)
                    check_deadline = time.monotonic() + 0.3
                    if proceed_condition(test_az, test_el, check_deadline):
                        print(f"[Spiral] Found at offset ({az_offset:.1f}°, {el_offset:.1f}°)")
                        return (True, test_az, test_el, az_offset, el_offset)

        return (False, az, el, 0.0, 0.0)

    def update_correction(self, expected_az, expected_el, actual_az, actual_el, confidence_weight=1.0):
        """Update correction based on detection offset."""
        az_error = _az_short_diff(expected_az, actual_az)
        el_error = actual_el - expected_el

        self.correction_history.append({
            'az_error': az_error,
            'el_error': el_error,
            'weight': confidence_weight,
            'time': time.time()
        })

        # Calculate weighted average with time decay
        if self.correction_history:
            now = time.time()
            total_weight = 0.0
            weighted_az = 0.0
            weighted_el = 0.0

            for correction in self.correction_history:
                age = now - correction['time']
                time_weight = math.exp(-age / 60.0)  # 60 second decay
                weight = correction['weight'] * time_weight

                weighted_az += correction['az_error'] * weight
                weighted_el += correction['el_error'] * weight
                total_weight += weight

            if total_weight > 0:
                self.az_correction = weighted_az / total_weight
                self.el_correction = weighted_el / total_weight
                self.confidence = min(1.0, total_weight / len(self.correction_history))

    def apply_correction(self, az, el):
        """Apply correction to a waypoint."""
        if self.confidence > 0.1:
            corrected_az = (az + self.az_correction * self.confidence) % 360.0
            corrected_el = np.clip(el + self.el_correction * self.confidence, 0, 90)
            return corrected_az, corrected_el
        return az, el


def patrol_waypoints_enhanced(shared_data,
                              waypoints,
                              proceed_condition=None,
                              dwell_seconds: float = 2.0,
                              settle_timeout_s: float = 3.0,
                              max_wait_s: float | None = None,
                              next_wp_speed_deg_per_s: float | None = None,
                              enable_spiral: bool = True,
                              spiral_radius: float = 3.0):
    """
    Enhanced version of patrol_waypoints with spiral search and correction.
    Maintains exact same interface as original.
    """
    n = len(waypoints)
    if n == 0:
        return

    # Initialize spiral enhancer
    enhancer = SpiralSearchEnhancer(
        shared_data,
        enable_spiral=enable_spiral,
        spiral_radius=spiral_radius
    )

    # Go to first waypoint
    first_az, first_el = waypoints[0]
    if _should_abort(shared_data): return
    _goto_and_wait(shared_data, first_az, first_el, settle_timeout_s=settle_timeout_s)

    for i in range(n):
        if _should_abort(shared_data): break

        # Get original waypoint
        original_az, original_el = waypoints[i]

        # Apply correction if we have confidence
        az, el = enhancer.apply_correction(original_az, original_el)

        if enhancer.confidence > 0.1 and (az != original_az or el != original_el):
            print(f"[Patrol] WP{i + 1}: ({original_az:.1f}°,{original_el:.1f}°) -> "
                  f"({az:.1f}°,{el:.1f}°) [correction applied]")

        # Move to waypoint
        _goto_and_wait(shared_data, az, el, settle_timeout_s=settle_timeout_s)

        # Calculate timeout
        derived_timeout = None
        if next_wp_speed_deg_per_s and next_wp_speed_deg_per_s > 1e-6 and n > 1:
            nxt = waypoints[(i + 1) % n]
            arc_deg = approx_arc_deg(az, el, nxt[0], nxt[1])
            derived_timeout = max(0.5, arc_deg / float(next_wp_speed_deg_per_s))

        if max_wait_s is not None and derived_timeout is not None:
            deadline = time.monotonic() + min(max_wait_s, derived_timeout)
        elif max_wait_s is not None:
            deadline = time.monotonic() + max_wait_s
        elif derived_timeout is not None:
            deadline = time.monotonic() + derived_timeout
        else:
            deadline = time.monotonic() + dwell_seconds

        # Wait for detection or timeout
        if proceed_condition is None:
            # No detection logic, just wait
            while time.monotonic() < deadline and not _should_abort(shared_data):
                time.sleep(0.01)
        else:
            # Use spiral search with proceed condition
            if enable_spiral:
                found, actual_az, actual_el, offset_az, offset_el = enhancer.spiral_search_at_waypoint(
                    az, el, proceed_condition, deadline
                )

                if found:
                    # Update correction for future waypoints
                    enhancer.update_correction(original_az, original_el, actual_az, actual_el)

                    # Record detection if enabled
                    if shared_data.get("record_tle_points") and shared_data["record_tle_points"].value:
                        dist_cm = float(shared_data["lidar_data"][0])
                        strength = float(shared_data["lidar_data"][1])
                        ts = float(shared_data["lidar_data"][2])

                        # Update satellite_points
                        try:
                            with shared_data["satellite_points"].get_lock():
                                shared_data["satellite_points"][:] = [actual_az, actual_el, dist_cm, strength, ts]
                        except:
                            shared_data["satellite_points"][:] = [actual_az, actual_el, dist_cm, strength, ts]

                        # Append to history
                        try:
                            shared_data["tracking_history"].append([actual_az, actual_el, dist_cm, strength, ts])
                            print(f"[Patrol] Detection saved at ({actual_az:.1f}°, {actual_el:.1f}°)")
                        except:
                            pass

                    if shared_data.get("satellite_detected"):
                        shared_data["satellite_detected"].value = True
                else:
                    # Reduce confidence if we miss
                    enhancer.confidence *= 0.8
            else:
                # Original behavior without spiral
                detected = bool(proceed_condition(az, el, deadline_s=deadline))

                if detected and shared_data.get("record_tle_points") and shared_data["record_tle_points"].value:
                    dist_cm = float(shared_data["lidar_data"][0])
                    strength = float(shared_data["lidar_data"][1])
                    ts = float(shared_data["lidar_data"][2])

                    try:
                        with shared_data["satellite_points"].get_lock():
                            shared_data["satellite_points"][:] = [az, el, dist_cm, strength, ts]
                    except:
                        shared_data["satellite_points"][:] = [az, el, dist_cm, strength, ts]

                    try:
                        shared_data["tracking_history"].append([az, el, dist_cm, strength, ts])
                    except:
                        pass

                    if shared_data.get("satellite_detected"):
                        shared_data["satellite_detected"].value = True

    # Return to first visible point
    if not _should_abort(shared_data):
        first_az, first_el = enhancer.apply_correction(first_az, first_el)
        _goto_and_wait(shared_data, first_az, first_el, settle_timeout_s=settle_timeout_s)

    shared_data["go_to_target"].value = False


def run_orbit_patrol_from_xyz(shared_data,
                              pts_xyz,
                              num_points: int = 9,
                              dwell_seconds: float = 2.0,
                              min_el_deg: float = 0.0,
                              max_el_deg: float = 60.0,
                              start_near_current: bool = True,
                              proceed_condition=None,
                              max_wait_s: float | None = None,
                              next_wp_speed_deg_per_s: float | None = None,
                              enable_spiral: bool = True,
                              spiral_radius: float = 3.0):
    """
    Enhanced drop-in replacement for run_orbit_patrol_from_xyz.
    Same interface with added spiral search parameters.
    """
    azel_path = xyz_to_azel_center(pts_xyz)
    visible = filter_by_elevation(azel_path, min_el=min_el_deg, max_el=max_el_deg)

    if not visible:
        print("[OrbitPatrolXYZ] No points in elevation window.")
        return

    wps = select_evenly_spaced(visible, num_points)

    if start_near_current:
        try:
            cur_az = float(shared_data["stepper_degrees"].value) % 360.0
            cur_el = float(shared_data["servo_degrees"].value)
            start = min(range(len(wps)),
                        key=lambda i: abs(_az_short_diff(cur_az, wps[i][0])) + abs(cur_el - wps[i][1]))
            wps = wps[start:] + wps[:start]
        except Exception:
            pass

    print(f"[OrbitPatrolXYZ] Waypoints ({len(wps)}): " +
          ", ".join([f"({az:.1f}°, {el:.1f}°)" for az, el in wps]))

    if enable_spiral:
        print(f"[OrbitPatrolXYZ] Spiral search enabled with radius {spiral_radius}°")

    # Use enhanced patrol
    patrol_waypoints_enhanced(shared_data, wps,
                              proceed_condition=proceed_condition,
                              dwell_seconds=dwell_seconds,
                              settle_timeout_s=3.0,
                              max_wait_s=max_wait_s,
                              next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                              enable_spiral=enable_spiral,
                              spiral_radius=spiral_radius)

    print("[OrbitPatrolXYZ] Patrol complete.")


def run_orbit_patrol_from_query(shared_data,
                                query: str,
                                num_points: int = 9,
                                dwell_seconds: float = 2.0,
                                min_el_deg: float = 0.0,
                                max_el_deg: float = 60.0,
                                duration_minutes: int = 90,
                                step_seconds: int = 60,
                                start_near_current: bool = True,
                                proceed_condition=None,
                                next_wp_speed_deg_per_s: float | None = None,
                                max_wait_s: float | None = None,
                                full_circle: bool = False,
                                full_circle_samples: int = 720,
                                enable_spiral: bool = True,
                                spiral_radius: float = 3.0):
    """
    Enhanced drop-in replacement for run_orbit_patrol_from_query.
    Same interface with added spiral search parameters.
    """
    if full_circle:
        from orbit_patrol_from_xyz import generate_full_circle_xyz_from_raan_incl
        raan = float(shared_data["rann"].value)
        incl = float(shared_data["inclination"].value)
        pts_unit = generate_full_circle_xyz_from_raan_incl(
            raan, incl, n_samples=full_circle_samples
        )
        print(f"[OrbitPatrolXYZ] Full-circle mode: RAAN={raan:.2f}°, inc={incl:.2f}°, samples={len(pts_unit)}")
        run_orbit_patrol_from_xyz(shared_data,
                                  pts_unit,
                                  num_points=num_points,
                                  dwell_seconds=dwell_seconds,
                                  min_el_deg=min_el_deg,
                                  max_el_deg=max_el_deg,
                                  start_near_current=start_near_current,
                                  proceed_condition=proceed_condition,
                                  next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                                  max_wait_s=max_wait_s,
                                  enable_spiral=enable_spiral,
                                  spiral_radius=spiral_radius)
        return

    # Time-sampled mode
    from datahandler import get_orbit_xyz_for_query
    name, pts_km = get_orbit_xyz_for_query(query,
                                           duration_minutes=duration_minutes,
                                           step_seconds=step_seconds)
    print(f"[OrbitPatrolXYZ] Using orbit '{name}', {len(pts_km)} samples.")
    run_orbit_patrol_from_xyz(shared_data,
                              pts_km,
                              num_points=num_points,
                              dwell_seconds=dwell_seconds,
                              min_el_deg=min_el_deg,
                              max_el_deg=max_el_deg,
                              start_near_current=start_near_current,
                              proceed_condition=proceed_condition,
                              next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                              max_wait_s=max_wait_s,
                              enable_spiral=enable_spiral,
                              spiral_radius=spiral_radius)


# For backward compatibility, keep original function names available
patrol_waypoints = patrol_waypoints_enhanced

# Example usage showing it's a drop-in replacement
if __name__ == "__main__":
    # This would work exactly like the original, but with spiral search
    # run_orbit_patrol_from_query(
    #     shared_data,
    #     "ISS (ZARYA)",
    #     num_points=30,
    #     enable_spiral=True,  # New parameter (default True)
    #     spiral_radius=3.0    # New parameter (default 3.0)
    # )
    pass