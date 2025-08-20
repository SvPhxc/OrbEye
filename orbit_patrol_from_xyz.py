import time, math
import numpy as np

# ---------------- conversion & helpers ----------------
def xyz_to_azel_center(xyz: np.ndarray):
    """
    xyz: (..., 3) ECI (or TEME) points from Earth's center (units don't matter).
    Returns list[(az_deg, el_deg)] with:
      - az from +X toward +Y, in [0, 360)
      - el from XY-plane, + toward +Z
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    # Normalize to direction
    norms = np.linalg.norm(xyz, axis=1)
    norms[norms == 0.0] = 1.0
    u = xyz / norms[:, None]
    x, y, z = u[:, 0], u[:, 1], u[:, 2]

    az = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return list(zip(az.tolist(), el.tolist()))

def filter_by_elevation(azels, min_el=0.0, max_el=60.0):
    return [(az, el) for az, el in azels if (min_el <= el <= max_el)]

def select_evenly_spaced(seq, k: int):
    if not seq:
        return []
    k = max(1, int(k))
    if k >= len(seq):
        return seq
    idx = np.linspace(0, len(seq) - 1, k).astype(int)
    return [seq[i] for i in idx]

def rotate_to_nearest_current(shared_data, waypoints):
    """
    Rotate waypoint list so we start from the point closest to current mount az/el.
    Minimizes the first big slew.
    """
    if not waypoints:
        return waypoints
    try:
        cur_az = float(shared_data["stepper_degrees"].value) % 360.0
        cur_el = float(shared_data["servo_degrees"].value)
    except Exception:
        return waypoints

    # angular distance in az (short way around) + elevation difference
    def ang_dist(wp):
        az, el = wp
        d_az = abs(((az - cur_az + 180.0) % 360.0) - 180.0)
        d_el = abs(el - cur_el)
        # weight az and el roughly equally
        return d_az + d_el

    start = min(range(len(waypoints)), key=lambda i: ang_dist(waypoints[i]))
    return waypoints[start:] + waypoints[:start]

# ---------------- motor driving ----------------
def patrol_waypoints(shared_data,
                     waypoints,
                     dwell_seconds: float = 2.0,
                     settle_timeout_s: float = 3.0):
    """
    Uses your existing GOTO loop:
      shared_data["target_azimuth"], ["target_elevation"],
      ["go_to_target"], ["target_reached"].
    """
    for (az, el) in waypoints:
        if shared_data["shutdown"].value:
            break

        # Command move
        shared_data["target_azimuth"].value = float(az)
        shared_data["target_elevation"].value = float(el)
        shared_data["go_to_target"].value = True

        # Wait for reach or timeout
        t0 = time.monotonic()
        while (time.monotonic() - t0) < settle_timeout_s and not shared_data["shutdown"].value:
            if shared_data["target_reached"].value:
                break
            time.sleep(0.01)

        # Dwell
        t1 = time.monotonic()
        while (time.monotonic() - t1) < dwell_seconds and not shared_data["shutdown"].value:
            time.sleep(0.01)

    # Let controller fall back to other states if needed
    shared_data["go_to_target"].value = False

# ---------------- public entry points ----------------
def run_orbit_patrol_from_xyz(shared_data,
                              pts_xyz,                  # Nx3 (km, m, whatever)
                              num_points: int = 9,
                              dwell_seconds: float = 2.0,
                              min_el_deg: float = 0.0,
                              max_el_deg: float = 60.0,
                              start_near_current: bool = True):
    """
    Convert XYZ track to az/el (center-of-Earth observer), clip by elevation,
    pick K waypoints, optionally rotate to start near current pointing, then patrol.
    """
    azel_path = xyz_to_azel_center(pts_xyz)
    visible = filter_by_elevation(azel_path, min_el=min_el_deg, max_el=max_el_deg)
    if not visible:
        print("[OrbitPatrolXYZ] No points in elevation window.")
        return

    wps = select_evenly_spaced(visible, num_points)
    if start_near_current:
        wps = rotate_to_nearest_current(shared_data, wps)

    print(f"[OrbitPatrolXYZ] Waypoints ({len(wps)}): " +
          ", ".join([f"({az:.1f}°, {el:.1f}°)" for az, el in wps]))

    patrol_waypoints(shared_data, wps, dwell_seconds=dwell_seconds, settle_timeout_s=3.0)
    print("[OrbitPatrolXYZ] Patrol complete.")

def run_orbit_patrol_from_query(shared_data,
                                query: str,
                                num_points: int = 9,
                                dwell_seconds: float = 2.0,
                                min_el_deg: float = 0.0,
                                max_el_deg: float = 60.0,
                                duration_minutes: int = 90,
                                step_seconds: int = 60,
                                start_near_current: bool = True):
    """
    Convenience wrapper: calls your datahandler to get orbit points,
    then runs the patrol.
    """
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
                              start_near_current=start_near_current)
