import time, math
import numpy as np

# ---------- helpers: convert & choose ----------
def xyz_to_azel_center(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
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
    if not seq: return []
    k = max(1, int(k))
    if k >= len(seq): return seq
    idx = np.linspace(0, len(seq) - 1, k).astype(int)
    return [seq[i] for i in idx]

def _az_short_diff(a, b):
    d = (b - a + 180.0) % 360.0 - 180.0
    return d

def approx_arc_deg(a0, e0, a1, e1):
    d_el = (e1 - e0)
    d_az = _az_short_diff(a0, a1)
    c = math.cos(math.radians(0.5 * (e0 + e1)))
    return math.hypot(d_el, d_az * max(1e-6, c))

# ---------- mount primitives ----------
def _goto_and_wait(shared_data, az, el, settle_timeout_s=3.0):
    shared_data["target_azimuth"].value = float(az)
    shared_data["target_elevation"].value = float(el)
    shared_data["go_to_target"].value = True
    t0 = time.monotonic()
    while (time.monotonic() - t0) < settle_timeout_s and not shared_data["shutdown"].value:
        if shared_data["target_reached"].value:
            return True
        time.sleep(0.01)
    return shared_data["target_reached"].value

def _should_abort(shared_data):
    return (
        shared_data["shutdown"].value
        or (shared_data.get("orbit_patrol_cancel") and shared_data["orbit_patrol_cancel"].value)
        or shared_data["background_scan_active"].value
        or shared_data["lidar_track_mode_active"].value
    )

# ---------- proceed condition (ClutterFilter) ----------
def make_detection_proceed_condition(clutter_filter, shared_data,
                                     confirm_hits=3, max_age_s=0.5):
    """
    Returns: proceed_condition(az, el, deadline_s) -> bool
    Calls clutter_filter with (az, el, dist_cm, strength) until it returns True confirm_hits times,
    or until deadline is reached.
    """
    def proceed_condition(az, el, deadline_s):
        ok_streak = 0
        last_seen_ts = -1.0
        while time.monotonic() < deadline_s:
            if _should_abort(shared_data):
                return False

            dist_cm = float(shared_data["lidar_data"][0])
            strength = float(shared_data["lidar_data"][1])
            ts = float(shared_data["lidar_data"][2])

            # fresh LiDAR only
            if ts > 0 and (time.time() - ts) <= max_age_s:
                try:
                    is_fg = clutter_filter.is_foreground(az, el, dist_cm, strength)
                except AttributeError:
                    try:
                        is_fg = clutter_filter.classify(az, el, dist_cm, strength)
                    except AttributeError:
                        is_fg = bool(clutter_filter.update(az, el, dist_cm, strength))

                if is_fg:
                    if ts != last_seen_ts:
                        ok_streak += 1
                        last_seen_ts = ts
                    if ok_streak >= confirm_hits:
                        return True
                else:
                    ok_streak = 0

            time.sleep(0.01)
        return False
    return proceed_condition

# ---------- main patrol ----------
def patrol_waypoints(shared_data,
                     waypoints,
                     proceed_condition=None,
                     dwell_seconds: float = 2.0,
                     settle_timeout_s: float = 3.0,
                     max_wait_s: float | None = None,
                     next_wp_speed_deg_per_s: float | None = None):
    """
    For each waypoint: slew+settle -> wait until detection or timeout -> next.
    After last waypoint, returns to the FIRST visible one and holds.
    """
    n = len(waypoints)
    if n == 0:
        return

    first_az, first_el = waypoints[0]
    if _should_abort(shared_data): return
    _goto_and_wait(shared_data, first_az, first_el, settle_timeout_s=settle_timeout_s)

    for i in range(n):
        if _should_abort(shared_data): break
        az, el = waypoints[i]

        # ensure at wp
        _goto_and_wait(shared_data, az, el, settle_timeout_s=settle_timeout_s)

        # derive timeout from speed & spacing
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

        # wait for detection or time
        if proceed_condition is None:
            while time.monotonic() < deadline and not _should_abort(shared_data):
                time.sleep(0.01)
        else:
            proceed_condition(az, el, deadline_s=deadline)

    # go back to first visible point
    if not _should_abort(shared_data):
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
                              next_wp_speed_deg_per_s: float | None = None):
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

    patrol_waypoints(shared_data, wps,
                     proceed_condition=proceed_condition,
                     dwell_seconds=dwell_seconds,
                     settle_timeout_s=3.0,
                     max_wait_s=max_wait_s,
                     next_wp_speed_deg_per_s=next_wp_speed_deg_per_s)

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
                                max_wait_s: float | None = None):
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
                              max_wait_s=max_wait_s)
