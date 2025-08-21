import time
import math
import numpy as np
from collections import deque
import traceback

# --- Assumed Project Imports ---
# These functions are expected to exist in your project structure based on the context provided.
# If they are in different files, you may need to adjust the import paths.
try:
    from datahandler import get_orbit_xyz_for_query
    from orbit_patrol_from_xyz import (
        generate_full_circle_xyz_from_raan_incl,
        xyz_to_azel_center,
        filter_by_elevation,
        select_evenly_spaced,
        _az_short_diff,
    )
except ImportError as e:
    print(f"[spiral_orbit_patrol] Warning: Could not import dependencies: {e}. "
          "Ensure datahandler.py and orbit_patrol_from_xyz.py are in the correct path.")


    # Define dummy functions to allow the script to be parsed
    def get_orbit_xyz_for_query(*args, **kwargs):
        return None, []


    def generate_full_circle_xyz_from_raan_incl(*args, **kwargs):
        return []


    def xyz_to_azel_center(pts):
        return np.array([])


    def filter_by_elevation(path, min_el, max_el):
        return []


    def select_evenly_spaced(path, num_points):
        return []


    def _az_short_diff(a, b):
        return 0


# --------------------------------------------------------------------------
# --- Core Spiral Search and Correction Logic (from your provided code) ----
# --------------------------------------------------------------------------

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

        # Spiral outward
        radius = 0.0
        while radius <= self.max_spiral_radius and (time.time() - start_time) < max_time:
            if self.shared_data["shutdown"].value:
                break

            radius += self.spiral_step
            points_in_ring = max(4, int(8 * radius / self.spiral_step))

            for i in range(points_in_ring):
                if time.time() - start_time > max_time or self.shared_data["shutdown"].value:
                    break

                angle = (2 * math.pi * i) / points_in_ring
                az_offset = radius * math.cos(angle)
                el_offset = radius * math.sin(angle) * 0.7

                test_az = center_az + az_offset
                test_el = np.clip(center_el + el_offset, 0, 90)

                if self._move_to_position(test_az, test_el):
                    dist, strength, _ = self._get_lidar_reading()

                    if detector_func(test_az, test_el, dist, strength):
                        if strength > best_strength:
                            best_detection = (True, az_offset, el_offset, strength)
                            best_strength = strength
                            if strength > self.high_strength_threshold:
                                print(f"[Spiral] High-confidence detection at offset "
                                      f"({az_offset:.2f}°, {el_offset:.2f}°), strength={strength:.0f}")
                                return best_detection

        if best_detection:
            print(f"[Spiral] Best detection at offset ({best_detection[1]:.2f}°, {best_detection[2]:.2f}°), "
                  f"strength={best_detection[3]:.0f}")
            return best_detection

        return (False, 0.0, 0.0, 0.0)

    def update_correction(self, expected_az, expected_el, actual_az, actual_el, strength):
        """Update the correction factors based on detected vs expected position."""
        az_error = actual_az - expected_az
        el_error = actual_el - expected_el
        weight = min(1.0, strength / self.high_strength_threshold)

        self.correction_history.append(
            {'az_error': az_error, 'el_error': el_error, 'weight': weight, 'time': time.time()})

        if len(self.correction_history) > 0:
            now = time.time()
            weighted_az, weighted_el, weight_sum = 0.0, 0.0, 0.0

            for correction in self.correction_history:
                age = now - correction['time']
                time_weight = math.exp(-age / 30.0)
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
        """Apply the current correction to a target position."""
        corrected_az = target_az + (self.current_az_correction * self.correction_confidence)
        corrected_el = target_el + (self.current_el_correction * self.correction_confidence)
        corrected_el = np.clip(corrected_el, 0, 90)
        corrected_az %= 360.0
        return corrected_az, corrected_el

    def _move_to_position(self, az, el):
        """Move to position and wait for arrival."""
        self.shared_data["target_azimuth"].value = float(az)
        self.shared_data["target_elevation"].value = float(el)
        self.shared_data["go_to_target"].value = True

        timeout = 0.5
        start = time.time()
        while time.time() - start < timeout:
            if self.shared_data["target_reached"].value:
                time.sleep(0.01)
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
        time.sleep(0.002)
        with self.shared_data["lidar_data"].get_lock():
            dist = float(self.shared_data["lidar_data"][0])
            strength = float(self.shared_data["lidar_data"][1])
            ts = float(self.shared_data["lidar_data"][2])
        return dist, strength, ts


def enhanced_patrol_waypoints(shared_data, waypoints, clutter_filter=None,
                              spiral_corrector=None, **kwargs):
    """Enhanced patrol with spiral search and correction at each waypoint."""
    if spiral_corrector is None:
        spiral_corrector = SpiralSearchCorrector(shared_data)

    def detector(az, el, dist, strength):
        if dist < 10 or dist > 16000 or strength < spiral_corrector.min_strength_threshold:
            return False
        if clutter_filter:
            try:
                if hasattr(clutter_filter, 'is_foreground'):
                    return clutter_filter.is_foreground(az, el, dist, strength)
            except Exception:
                pass
        return True

    print(f"[EnhancedPatrol] Starting patrol of {len(waypoints)} waypoints with spiral search")
    detection_history = []

    for i, (original_az, original_el) in enumerate(waypoints):
        if shared_data["shutdown"].value or shared_data["orbit_patrol_cancel"].value:
            print("[EnhancedPatrol] Patrol cancelled.")
            break

        target_az, target_el = spiral_corrector.apply_correction(original_az, original_el)
        print(f"[EnhancedPatrol] Waypoint {i + 1}/{len(waypoints)}: "
              f"Original({original_az:.1f}°, {original_el:.1f}°) -> Corrected({target_az:.1f}°, {target_el:.1f}°)")

        found, az_offset, el_offset, strength = spiral_corrector.spiral_search(
            target_az, target_el, detector, max_time=2.0
        )

        if found:
            actual_az = target_az + az_offset
            actual_el = target_el + el_offset
            print(f"[EnhancedPatrol] Detection at ({actual_az:.1f}°, {actual_el:.1f}°), strength={strength:.0f}")

            spiral_corrector.update_correction(original_az, original_el, actual_az, actual_el, strength)

            detection_info = {'az': actual_az, 'el': actual_el, 'strength': strength, 'time': time.time(),
                              'waypoint_idx': i}
            detection_history.append(detection_info)

            try:
                with shared_data["satellite_points"].get_lock():
                    shared_data["satellite_points"][:] = [actual_az, actual_el, float(shared_data["lidar_data"][0]),
                                                          strength, time.time()]
                if shared_data.get("record_tle_points") and shared_data["record_tle_points"].value:
                    with shared_data["tracking_history"].get_lock():
                        shared_data["tracking_history"].append(
                            [actual_az, actual_el, float(shared_data["lidar_data"][0]), strength, time.time()])
                if shared_data.get("satellite_detected"):
                    shared_data["satellite_detected"].value = True
            except Exception as e:
                print(f"[EnhancedPatrol] Error updating shared points: {e}")
        else:
            print(f"[EnhancedPatrol] No detection at waypoint {i + 1}")
            spiral_corrector.correction_confidence *= 0.8

    if waypoints and not shared_data["shutdown"].value:
        first_az, first_el = spiral_corrector.apply_correction(waypoints[0][0], waypoints[0][1])
        spiral_corrector._move_to_position(first_az, first_el)

    print(f"[EnhancedPatrol] Patrol complete. {len(detection_history)} detections made.")
    return detection_history


def run_enhanced_orbit_patrol_from_xyz(shared_data, pts_xyz, num_points=9, spiral_radius=3.0, **kwargs):
    """Run orbit patrol with spiral search enhancement from XYZ coordinates."""
    azel_path = xyz_to_azel_center(pts_xyz)
    min_el = kwargs.get('min_el_deg', 0.0)
    max_el = kwargs.get('max_el_deg', 60.0)
    visible = filter_by_elevation(azel_path, min_el=min_el, max_el=max_el)

    if not visible:
        print("[EnhancedPatrol] No points in elevation window.")
        return []

    waypoints = select_evenly_spaced(visible, num_points)

    if kwargs.get('start_near_current', True):
        try:
            cur_az = float(shared_data["stepper_degrees"].value) % 360.0
            cur_el = float(shared_data["servo_degrees"].value)
            start = min(range(len(waypoints)),
                        key=lambda i: abs(_az_short_diff(cur_az, waypoints[i][0])) + abs(cur_el - waypoints[i][1]))
            waypoints = waypoints[start:] + waypoints[:start]
        except Exception:
            pass

    print(f"[EnhancedPatrol] Starting enhanced patrol with {len(waypoints)} waypoints")
    corrector = SpiralSearchCorrector(shared_data, max_spiral_radius=spiral_radius)
    clutter_filter = kwargs.get('clutter_filter')

    return enhanced_patrol_waypoints(shared_data, waypoints, clutter_filter=clutter_filter, spiral_corrector=corrector,
                                     **kwargs)


# --------------------------------------------------------------------------
# --- Main Function to Run Patrol from a Query (as requested) ---
# --------------------------------------------------------------------------

def run_orbit_patrol_from_query(shared_data, query, **kwargs):
    """
    Fetches orbit data for a query and runs an enhanced orbit patrol with spiral search.

    This function is the primary entry point for initiating a patrol from a satellite name.
    It orchestrates fetching orbital data, configuring the patrol, and executing the search pattern.

    Args:
        shared_data: The shared memory data structure.
        query (str): The name of the satellite to track (e.g., "ISS (ZARYA)").
        **kwargs: Additional configuration options to override defaults, such as:
                  - num_points (int): Number of waypoints.
                  - spiral_radius (float): Search radius in degrees.
                  - min_el_deg (float): Minimum elevation for waypoints.
                  - max_el_deg (float): Maximum elevation.
    """
    print(f"[OrbitPatrol] Starting enhanced orbit patrol for query: '{query}'")
    shared_data["orbit_patrol_active"].value = True
    shared_data["orbit_patrol_cancel"].value = False

    try:
        # Get orbit data from TLE using the query
        name, pts_km = get_orbit_xyz_for_query(
            query,
            duration_minutes=kwargs.get('duration_minutes', 90),
            step_seconds=kwargs.get('step_seconds', 60)
        )

        if pts_km is None or len(pts_km) == 0:
            print(f"[OrbitPatrol] Could not get orbit data for '{query}'. Checking for fallback.")
            if shared_data.get("rann") and shared_data.get("inclination"):
                print("[OrbitPatrol] Falling back to full circle generation from RAAN/Inclination.")
                raan = float(shared_data["rann"].value)
                incl = float(shared_data["inclination"].value)
                pts_km = generate_full_circle_xyz_from_raan_incl(raan, incl, n_samples=360)
            else:
                print("[OrbitPatrol] No valid orbit data or fallback available. Aborting.")
                return None

        # Configure patrol parameters from shared_data and kwargs
        patrol_config = {
            'num_points': shared_data["orbit_patrol_points"].value,
            'spiral_radius': kwargs.get('spiral_radius', 4.0),
            'min_el_deg': kwargs.get('min_el_deg', 10.0),
            'max_el_deg': kwargs.get('max_el_deg', 80.0),
            'start_near_current': kwargs.get('start_near_current', True),
        }
        patrol_config.update(kwargs)  # Allow any kwarg to be passed through

        # Attempt to create and add a clutter filter
        try:
            from tracking_logic import ClutterFilter
            background_path = shared_data["background_path"].value.decode('utf-8')
            clutter_filter = ClutterFilter(background_path)
            patrol_config['clutter_filter'] = clutter_filter
            print("[OrbitPatrol] ClutterFilter initialized.")
        except Exception as e:
            print(f"[OrbitPatrol] ClutterFilter not available: {e}. Proceeding without it.")

        # Run the core patrol logic with the generated XYZ points
        detections = run_enhanced_orbit_patrol_from_xyz(
            shared_data,
            pts_km,
            **patrol_config
        )

        # Process and print results
        if detections:
            print(f"[OrbitPatrol] Patrol for '{query}' completed with {len(detections)} detections.")
            if len(detections) > 1:
                avg_strength = sum(d['strength'] for d in detections) / len(detections)
                print(f"[OrbitPatrol] Average signal strength: {avg_strength:.0f}")
        else:
            print(f"[OrbitPatrol] Patrol for '{query}' completed with no detections.")

        return detections

    except Exception as e:
        print(f"[OrbitPatrol] A critical error occurred during the patrol: {e}")
        traceback.print_exc()
        return None
    finally:
        print(f"[OrbitPatrol] Patrol sequence for '{query}' finished.")
        shared_data["orbit_patrol_active"].value = False
        shared_data["satellite_detected"].value = False