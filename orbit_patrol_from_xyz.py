import time
import math
import numpy as np
from collections import deque


# Placeholder for a function that would typically exist in a data handling module.
# This fetches satellite TLEs and computes their position over time.
def get_orbit_xyz_for_query(query, duration_minutes, step_seconds):
    """
    Placeholder function to simulate fetching orbital data.
    In a real system, this would use a library like Skyfield to get TLEs
    and calculate the satellite's position.

    Returns a simulated orbit for "ISS (ZARYA)".
    """
    print(f"[DataHandler] Simulating orbit data fetch for '{query}'...")
    if "iss" in query.lower():
        # Simulate a typical LEO orbit (e.g., ISS)
        earth_radius_km = 6371
        iss_altitude_km = 400
        orbit_radius = earth_radius_km + iss_altitude_km

        num_steps = int((duration_minutes * 60) / step_seconds)
        pts_xyz = []
        for i in range(num_steps):
            angle = (i / num_steps) * 2 * np.pi
            # Simple circular orbit in the XY plane for demonstration
            x = orbit_radius * np.cos(angle)
            y = orbit_radius * np.sin(angle)
            z = 0  # Simplified equatorial orbit
            pts_xyz.append([x, y, z])

        print(f"[DataHandler] Generated {len(pts_xyz)} simulated points for ISS.")
        return "ISS (ZARYA)", np.array(pts_xyz)
    else:
        print(f"[DataHandler] No simulation data for '{query}'. Returning empty path.")
        return query, np.array([])


# Placeholder for a function to generate a full orbit from orbital elements
def generate_full_circle_xyz_from_raan_incl(raan_deg, incl_deg, n_samples=360):
    """
    Generates a full circular orbit path as unit vectors from RAAN and inclination.
    """
    print(f"[OrbitGenerator] Generating full circle with RAAN={raan_deg}°, Incl={incl_deg}°")
    raan_rad = np.radians(raan_deg)
    incl_rad = np.radians(incl_deg)

    # Rotation matrix for inclination
    R_incl = np.array([
        [1, 0, 0],
        [0, np.cos(incl_rad), -np.sin(incl_rad)],
        [0, np.sin(incl_rad), np.cos(incl_rad)]
    ])

    # Rotation matrix for RAAN
    R_raan = np.array([
        [np.cos(raan_rad), -np.sin(raan_rad), 0],
        [np.sin(raan_rad), np.cos(raan_rad), 0],
        [0, 0, 1]
    ])

    path = []
    for i in range(n_samples):
        angle = 2 * np.pi * i / n_samples
        # Point on a simple equatorial unit circle
        p = np.array([np.cos(angle), np.sin(angle), 0])
        # Rotate for inclination, then for RAAN
        p_rot = R_raan @ R_incl @ p
        path.append(p_rot)

    return np.array(path)


# --- Helper functions for coordinate conversion and waypoint selection ---

def _az_short_diff(a, b):
    """Calculate the shortest angle difference between two azimuths."""
    diff = (b - a) % 360
    return diff - 360 if diff > 180 else diff


def xyz_to_azel_center(pts_xyz, center_xyz=np.array([0, 0, 6371])):
    """Converts XYZ coordinates to Azimuth/Elevation relative to a center."""
    relative_pts = pts_xyz - center_xyz
    x, y, z = relative_pts[:, 0], relative_pts[:, 1], relative_pts[:, 2]

    dist_2d = np.sqrt(x ** 2 + y ** 2)
    elevation = np.degrees(np.arctan2(z, dist_2d))
    azimuth = np.degrees(np.arctan2(y, x)) % 360

    return np.column_stack((azimuth, elevation))


def filter_by_elevation(azel_path, min_el, max_el):
    """Filters waypoints to be within a specific elevation range."""
    return azel_path[(azel_path[:, 1] >= min_el) & (azel_path[:, 1] <= max_el)]


def select_evenly_spaced(points, num_points):
    """Selects a number of evenly spaced points from a path."""
    if len(points) <= num_points:
        return points
    indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
    return points[indices]


# --- Core Classes and Functions for Enhanced Patrol ---

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

        self.correction_history = deque(maxlen=5)
        self.current_az_correction = 0.0
        self.current_el_correction = 0.0
        self.correction_confidence = 0.0

        self.min_strength_threshold = 100
        self.high_strength_threshold = 500

    def spiral_search(self, center_az, center_el, detector_func, max_time=3.0):
        start_time = time.time()
        best_detection = None
        best_strength = 0

        if self._check_position(center_az, center_el, detector_func):
            dist, strength, _ = self._get_lidar_reading()
            if strength > self.min_strength_threshold:
                return (True, 0.0, 0.0, strength)

        radius = 0.0
        while radius <= self.max_spiral_radius and (time.time() - start_time) < max_time:
            if self.shared_data["shutdown"].value:
                break
            radius += self.spiral_step
            points_in_ring = max(4, int(8 * radius / self.spiral_step))

            for i in range(points_in_ring):
                if time.time() - start_time > max_time: break
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
                                print(
                                    f"[Spiral] High-confidence detection at offset ({az_offset:.2f}°, {el_offset:.2f}°), strength={strength:.0f}")
                                return best_detection

        if best_detection:
            print(
                f"[Spiral] Best detection at offset ({best_detection[1]:.2f}°, {best_detection[2]:.2f}°), strength={best_detection[3]:.0f}")
            return best_detection
        return (False, 0.0, 0.0, 0.0)

    def update_correction(self, expected_az, expected_el, actual_az, actual_el, strength):
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
                print(
                    f"[Correction] Updated: az={self.current_az_correction:.2f}°, el={self.current_el_correction:.2f}°, conf={self.correction_confidence:.2f}")

    def apply_correction(self, target_az, target_el):
        corrected_az = target_az + (self.current_az_correction * self.correction_confidence)
        corrected_el = target_el + (self.current_el_correction * self.correction_confidence)
        return corrected_az % 360.0, np.clip(corrected_el, 0, 90)

    def _move_to_position(self, az, el):
        self.shared_data["target_azimuth"].value = float(az)
        self.shared_data["target_elevation"].value = float(el)
        self.shared_data["go_to_target"].value = True
        start = time.time()
        while time.time() - start < 0.5:
            if self.shared_data["target_reached"].value:
                time.sleep(0.01)
                return True
            time.sleep(0.001)
        return False

    def _check_position(self, az, el, detector_func):
        if self._move_to_position(az, el):
            dist, strength, _ = self._get_lidar_reading()
            return detector_func(az, el, dist, strength)
        return False

    def _get_lidar_reading(self):
        time.sleep(0.002)
        with self.shared_data["lidar_data"].get_lock():
            return float(self.shared_data["lidar_data"][0]), float(self.shared_data["lidar_data"][1]), float(
                self.shared_data["lidar_data"][2])


def enhanced_patrol_waypoints(shared_data, waypoints, clutter_filter=None, spiral_corrector=None, **kwargs):
    if spiral_corrector is None:
        spiral_corrector = SpiralSearchCorrector(shared_data)

    def detector(az, el, dist, strength):
        if not (10 < dist < 16000 and strength > spiral_corrector.min_strength_threshold):
            return False
        if clutter_filter and hasattr(clutter_filter, 'is_valid_target'):
            return clutter_filter.is_valid_target(az, el, dist, strength)
        return True

    print(f"[EnhancedPatrol] Starting patrol of {len(waypoints)} waypoints with spiral search")
    detection_history = []

    for i, (original_az, original_el) in enumerate(waypoints):
        if shared_data["shutdown"].value: break

        target_az, target_el = spiral_corrector.apply_correction(original_az, original_el)
        print(
            f"[EnhancedPatrol] WP {i + 1}/{len(waypoints)}: Orig({original_az:.1f}°, {original_el:.1f}°) -> Corr({target_az:.1f}°, {target_el:.1f}°)")

        found, az_offset, el_offset, strength = spiral_corrector.spiral_search(target_az, target_el, detector,
                                                                               max_time=2.0)

        if found:
            actual_az, actual_el = target_az + az_offset, target_el + el_offset
            print(f"[EnhancedPatrol] Detection at ({actual_az:.1f}°, {actual_el:.1f}°), strength={strength:.0f}")

            spiral_corrector.update_correction(original_az, original_el, actual_az, actual_el, strength)

            det_info = {'az': actual_az, 'el': actual_el, 'strength': strength, 'time': time.time(), 'waypoint_idx': i}
            detection_history.append(det_info)

            try:
                # Update shared data for visualization and tracking history
                dist, _, _ = spiral_corrector._get_lidar_reading()
                with shared_data["satellite_points"].get_lock():
                    shared_data["satellite_points"][:] = [actual_az, actual_el, dist, strength, time.time()]
                if shared_data.get("record_tle_points") and shared_data["record_tle_points"].value:
                    shared_data["tracking_history"].append([actual_az, actual_el, dist, strength, time.time()])
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
    azel_path = xyz_to_azel_center(pts_xyz)
    visible = filter_by_elevation(azel_path, min_el=kwargs.get('min_el_deg', 0.0),
                                  max_el=kwargs.get('max_el_deg', 60.0))

    if len(visible) == 0:
        print("[EnhancedPatrol] No points in elevation window.")
        return []

    waypoints = select_evenly_spaced(visible, num_points)

    if kwargs.get('start_near_current', True):
        cur_az = float(shared_data["stepper_degrees"].value) % 360.0
        cur_el = float(shared_data["servo_degrees"].value)
        start = min(range(len(waypoints)),
                    key=lambda i: abs(_az_short_diff(cur_az, waypoints[i][0])) + abs(cur_el - waypoints[i][1]))
        waypoints = np.roll(waypoints, -start, axis=0)

    print(f"[EnhancedPatrol] Starting enhanced patrol with {len(waypoints)} waypoints")
    corrector = SpiralSearchCorrector(shared_data, max_spiral_radius=spiral_radius)
    clutter_filter = kwargs.get('clutter_filter', None)

    return enhanced_patrol_waypoints(shared_data, waypoints, clutter_filter=clutter_filter, spiral_corrector=corrector,
                                     **kwargs)


# ==============================================================================
# === NEW MAIN FUNCTION TO BE CALLED FROM TRACKING_LOGIC =======================
# ==============================================================================

def run_orbit_patrol_from_query(shared_data, query="ISS (ZARYA)", **kwargs):
    """
    Initiates an enhanced orbit patrol by fetching data for a satellite query.

    This function serves as the primary entry point. It retrieves orbital path
    data (XYZ coordinates), generates visible waypoints (Az/El), and then
    executes the patrol using spiral search and correction logic.

    Args:
        shared_data: The main shared memory data structure.
        query (str): The name of the satellite to track (e.g., "ISS (ZARYA)").
        **kwargs: A dictionary of configuration options, including:
            - num_points (int): Number of waypoints for the patrol.
            - min_el_deg, max_el_deg (float): Elevation filter for waypoints.
            - spiral_radius (float): The radius for the spiral search.
            - clutter_filter (ClutterFilter): An instance of the clutter filter.
            - full_circle (bool): If True, generate a full orbit from RAAN/Incl.
            - and other parameters for get_orbit_xyz_for_query.

    Returns:
        list: A list of detection dictionaries from the patrol.
    """
    print(f"[OrbitPatrol] Received request to start enhanced patrol for query: '{query}'")

    # --- 1. Get Orbital Path Data (XYZ coordinates) ---
    pts_km = np.array([])
    # Check if full circle generation mode is requested and possible
    if kwargs.get('full_circle', False) and shared_data.get("rann") and shared_data.get("inclination"):
        try:
            raan = float(shared_data["rann"].value)
            incl = float(shared_data["inclination"].value)
            # Use a reasonable altitude for LEO if generating from elements
            altitude_km = 400
            earth_radius_km = 6371
            orbit_radius_km = earth_radius_km + altitude_km

            n_samples = kwargs.get('full_circle_samples', 720)
            unit_vectors = generate_full_circle_xyz_from_raan_incl(raan, incl, n_samples)
            pts_km = unit_vectors * orbit_radius_km
            print(f"[OrbitPatrol] Generated full-circle orbit at ~{altitude_km} km altitude.")
        except Exception as e:
            print(f"[OrbitPatrol] Could not generate full circle orbit: {e}")

    # Fallback to TLE-based query if full circle fails or is not requested
    if len(pts_km) == 0:
        duration = kwargs.get('duration_minutes', 90)
        step = kwargs.get('step_seconds', 60)
        _, pts_km = get_orbit_xyz_for_query(query, duration_minutes=duration, step_seconds=step)

    if pts_km is None or len(pts_km) == 0:
        print(f"[OrbitPatrol] ERROR: Could not get any orbit data for '{query}'. Aborting patrol.")
        return []

    # --- 2. Run the Enhanced Patrol ---
    # The run_enhanced_orbit_patrol_from_xyz function handles waypoint generation,
    # correction, and the actual patrol execution. We pass all kwargs to it.

    # Extract specific parameters for run_enhanced_orbit_patrol_from_xyz
    patrol_config = {
        'num_points': kwargs.get('num_points', 30),
        'spiral_radius': kwargs.get('spiral_radius', 4.0),
        'min_el_deg': kwargs.get('min_el_deg', 10.0),
        'max_el_deg': kwargs.get('max_el_deg', 80.0),
        'start_near_current': kwargs.get('start_near_current', True),
        'clutter_filter': kwargs.get('clutter_filter', None)
    }

    detections = run_enhanced_orbit_patrol_from_xyz(
        shared_data,
        pts_km,
        **patrol_config
    )

    # --- 3. Process and Return Results ---
    if detections:
        print(f"[OrbitPatrol] Patrol for '{query}' completed with {len(detections)} detections.")
    else:
        print(f"[OrbitPatrol] Patrol for '{query}' completed with no detections.")

    return detections