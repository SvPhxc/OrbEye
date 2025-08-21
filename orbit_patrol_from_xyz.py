

import time, math
import numpy as np


# ---------- helpers: convert & choose ----------
def _eci_plane_basis_from_raan_incl(raan_deg: float, incl_deg: float):
    Ω = math.radians(raan_deg)
    i = math.radians(incl_deg)
    # Ascending node (unit) in ECI XY
    p = np.array([math.cos(Ω), math.sin(Ω), 0.0], dtype=float)
    # 90° ahead in plane
    q = np.array([-math.sin(Ω) * math.cos(i), math.cos(Ω) * math.cos(i), math.sin(i)], dtype=float)
    return p, q


def generate_full_circle_xyz_from_raan_incl(raan_deg: float,
                                            incl_deg: float,
                                            n_samples: int = 720):
    """
    Returns Nx3 unit vectors covering the entire orbital plane (2π).
    """
    p, q = _eci_plane_basis_from_raan_incl(raan_deg, incl_deg)
    nus = np.linspace(0.0, 2 * math.pi, int(n_samples), endpoint=False)
    pts = np.cos(nus)[:, None] * p[None, :] + np.sin(nus)[:, None] * q[None, :]
    # Already unit vectors; small numerical drift might exist—normalize to be safe
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return pts / norms


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


# ---------- spiral search helpers (NEW) ----------
def generate_spiral_waypoints(center_az, center_el, max_radius_deg=1.0, points_per_rotation=16, num_rotations=2):
    """Generates Archimedean spiral waypoints around a center point."""
    waypoints = []
    center_el_rad = math.radians(center_el)
    cos_el = math.cos(center_el_rad)
    if abs(cos_el) < 1e-6: cos_el = 1e-6  # Avoid division by zero near zenith

    total_points = int(points_per_rotation * num_rotations)
    for i in range(total_points + 1):
        if i == 0:
            waypoints.append((center_az, center_el))
            continue

        angle = i * (2 * math.pi / points_per_rotation)
        radius = (i / total_points) * max_radius_deg

        # Convert polar (radius, angle) to Cartesian offsets in degrees,
        # adjusting for spherical distortion in azimuth.
        d_el = radius * math.sin(angle)
        d_az = radius * math.cos(angle) / cos_el

        az = (center_az + d_az + 360.0) % 360.0
        el = max(-90.0, min(90.0, center_el + d_el))
        waypoints.append((az, el))
    return waypoints


def _perform_spiral_search(shared_data, center_az, center_el, spiral_radius_deg, points_per_rotation, num_rotations):
    """
    Performs a spiral scan, returning the position of the highest LiDAR strength.
    """
    spiral_wps = generate_spiral_waypoints(center_az, center_el, spiral_radius_deg, points_per_rotation, num_rotations)

    best_pos = (center_az, center_el)
    max_strength = -1.0

    print(f"[Patrol.Spiral] Searching {len(spiral_wps)} points, radius {spiral_radius_deg}°")

    for az, el in spiral_wps:
        if _should_abort(shared_data):
            break

        # Use a short settle time for quick scanning
        if not _goto_and_wait(shared_data, az, el, settle_timeout_s=0.5):
            continue  # Move to next point if settle fails

        time.sleep(0.05)  # Brief pause for LiDAR to update

        strength = float(shared_data["lidar_data"][1])
        ts = float(shared_data["lidar_data"][2])

        # Consider only fresh samples
        if (time.time() - ts) <= 0.5:
            if strength > max_strength:
                max_strength = strength
                best_pos = (az, el)
                print(f"[Patrol.Spiral] New best strength: {max_strength:.1f} @ ({az:.2f}°, {el:.2f}°)")

    return best_pos, max_strength


# ---------- proceed condition (ClutterFilter) ----------
def _bind_cf_detector(clutter_filter):
    """
    Inspect ClutterFilter to find a suitable method and return a callable:
        detector(az, el, dist_cm, strength) -> bool
    Accepted method names (first one found wins): is_foreground, is_target,
    classify, detect, predict, evaluate, check, process, feed.
    The method's return can be bool, numeric (score>0 => True), or a tuple/list
    where the first bool/numeric will be used.
    """
    candidates = [
        "is_foreground", "is_target", "classify", "detect",
        "predict", "evaluate", "check", "process", "feed"
    ]
    for name in candidates:
        meth = getattr(clutter_filter, name, None)
        if callable(meth):
            print(f"[Patrol] Using ClutterFilter.{name}() for detection.")

            def detector(az, el, dist_cm, strength, _m=meth):
                res = _m(az, el, dist_cm, strength)
                # Normalize result to bool
                if isinstance(res, (bool, np.bool_)):
                    return bool(res)
                if isinstance(res, (tuple, list)) and len(res):
                    # Prefer first boolean in the sequence
                    for v in res:
                        if isinstance(v, (bool, np.bool_)):
                            return bool(v)
                    # Otherwise, use first numeric as score
                    v = res[0]
                    if isinstance(v, (int, float, np.floating)):
                        return v > 0
                    return bool(v)
                if isinstance(res, (int, float, np.floating)):
                    return res > 0
                return bool(res)

            return detector
    raise AttributeError(
        "ClutterFilter has no usable detector method. "
        "Expected one of: is_foreground, is_target, classify, detect, "
        "predict, evaluate, check, process, or feed."
    )


def make_detection_proceed_condition(clutter_filter, shared_data,
                                     confirm_hits=3, max_age_s=0.5):
    """
    Returns proceed_condition(az, el, deadline_s) -> bool
    Waits until the ClutterFilter reports foreground at the current waypoint
    (confirming 'confirm_hits' fresh LiDAR samples), or until deadline.
    """
    detector = _bind_cf_detector(clutter_filter)

    def proceed_condition(az, el, deadline_s):
        ok_streak = 0
        last_seen_ts = -1.0
        while time.monotonic() < deadline_s:
            if _should_abort(shared_data):
                return False

            # latest LiDAR sample
            dist_cm = float(shared_data["lidar_data"][0])
            strength = float(shared_data["lidar_data"][1])
            ts = float(shared_data["lidar_data"][2])

            # only consider fresh samples
            if ts > 0 and (time.time() - ts) <= max_age_s:
                if detector(az, el, dist_cm, strength):
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
                     next_wp_speed_deg_per_s: float | None = None,
                     enable_spiral_search: bool = False,
                     spiral_radius_deg: float = 0.5,
                     spiral_points_per_rotation: int = 16,
                     spiral_rotations: int = 2):
    """
    For each waypoint: slew+settle -> [spiral search] -> wait -> next.
    If spiral search finds a better spot, it corrects subsequent waypoints.
    """
    n = len(waypoints)
    if n == 0:
        return

    # Make a mutable copy of waypoints to allow for in-flight correction
    wps = [list(wp) for wp in waypoints]

    first_az, first_el = wps[0]
    if _should_abort(shared_data): return
    _goto_and_wait(shared_data, first_az, first_el, settle_timeout_s=settle_timeout_s)

    for i in range(n):
        if _should_abort(shared_data): break

        nominal_az, nominal_el = wps[i]
        current_az, current_el = nominal_az, nominal_el

        # ensure at wp
        _goto_and_wait(shared_data, current_az, current_el, settle_timeout_s=settle_timeout_s)

        # --- Optional: Spiral search and path correction ---
        if enable_spiral_search:
            (found_az, found_el), max_str = _perform_spiral_search(
                shared_data, current_az, current_el,
                spiral_radius_deg, spiral_points_per_rotation, spiral_rotations
            )

            # If a detection was made and it's offset from the nominal point
            if max_str > 0:
                d_az = _az_short_diff(current_az, found_az)
                d_el = found_el - current_el

                # If the correction is significant, apply it to future waypoints
                if abs(d_az) > 0.01 or abs(d_el) > 0.01:
                    print(f"[Patrol] Path corrected by (d_az={d_az:.2f}°, d_el={d_el:.2f}°)")
                    for j in range(i + 1, n):
                        wps[j][0] = (wps[j][0] + d_az + 360.0) % 360.0
                        wps[j][1] = max(-90.0, min(90.0, wps[j][1] + d_el))

                # Update current target to the best found position for detection dwell
                current_az, current_el = found_az, found_el
                _goto_and_wait(shared_data, current_az, current_el, settle_timeout_s=1.0)

        # derive timeout from speed & spacing
        derived_timeout = None
        if next_wp_speed_deg_per_s and next_wp_speed_deg_per_s > 1e-6 and n > 1:
            nxt = wps[(i + 1) % n]
            arc_deg = approx_arc_deg(current_az, current_el, nxt[0], nxt[1])
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
            detected = bool(proceed_condition(current_az, current_el, deadline_s=deadline))

            # If we confirmed a detection at this waypoint, save it
            if detected and (shared_data.get("record_tle_points") and shared_data["record_tle_points"].value):
                # Read the freshest LiDAR sample
                dist_cm = float(shared_data["lidar_data"][0])
                strength = float(shared_data["lidar_data"][1])
                ts = float(shared_data["lidar_data"][2])

                # 1) Update the single-slot 'satellite_points' (used by GUI heatmap)
                try:
                    with shared_data["satellite_points"].get_lock():
                        shared_data["satellite_points"][:] = [current_az, current_el, dist_cm, strength, ts]
                        print("[Patrol] saved -> satellite_points")
                except Exception:
                    shared_data["satellite_points"][:] = [current_az, current_el, dist_cm, strength, ts]
                    print("[Patrol] saved -> satellite_points (no lock)")

                # 2) Append to the growing history for TLE fitting
                try:
                    shared_data["tracking_history"].append([current_az, current_el, dist_cm, strength, ts])
                    print(f"[Patrol] appended to tracking_history (N={len(shared_data['tracking_history'])})")
                except Exception as e:
                    print(f"[Patrol] WARNING: couldn't append to tracking_history: {e}")

                if shared_data.get("satellite_detected"):
                    shared_data["satellite_detected"].value = True

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
                              next_wp_speed_deg_per_s: float | None = None,
                              enable_spiral_search: bool = False,
                              spiral_radius_deg: float = 0.5,
                              spiral_points_per_rotation: int = 16,
                              spiral_rotations: int = 2):
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
                     next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                     enable_spiral_search=enable_spiral_search,
                     spiral_radius_deg=spiral_radius_deg,
                     spiral_points_per_rotation=spiral_points_per_rotation,
                     spiral_rotations=spiral_rotations)

    print("[OrbitPatrolXYZ] Patrol complete.")


def make_detection_proceed_condition_from_callable(detector, shared_data,
                                                   confirm_hits=3, max_age_s=0.5):
    """
    detector(az, el, dist_cm, strength) -> bool
    Returns proceed_condition(az, el, deadline_s) that waits until detector is True
    for 'confirm_hits' fresh LiDAR samples (or until deadline).
    """

    def proceed_condition(az, el, deadline_s):
        ok_streak = 0
        last_seen_ts = -1.0
        import time as _t
        print(f"[Patrol] Waiting for detection @ ({az:.2f}°, {el:.2f}°) "
              f"confirm_hits={confirm_hits}, max_age={max_age_s}s")
        while _t.monotonic() < deadline_s:
            if (shared_data["shutdown"].value
                    or (shared_data.get("orbit_patrol_cancel") and shared_data["orbit_patrol_cancel"].value)
                    or shared_data["background_scan_active"].value
                    or shared_data["lidar_track_mode_active"].value):
                print("[Patrol] Aborted proceed-condition wait.")
                return False

            dist_cm = float(shared_data["lidar_data"][0])
            strength = float(shared_data["lidar_data"][1])
            ts = float(shared_data["lidar_data"][2])

            if ts > 0 and (_t.time() - ts) <= max_age_s:
                hit = bool(detector(az, el, dist_cm, strength))
                if hit:
                    if ts != last_seen_ts:
                        ok_streak += 1
                        last_seen_ts = ts
                        print(f"[Patrol]   hit {ok_streak}/{confirm_hits} "
                              f"(dist={dist_cm:.1f}cm, str={strength:.0f}, ts={ts:.3f})")
                    if ok_streak >= confirm_hits:
                        print("[Patrol] Detection confirmed.")
                        return True
                else:
                    # Only log resets if we had started a streak
                    if ok_streak > 0:
                        print("[Patrol]   streak reset.")
                    ok_streak = 0

            _t.sleep(0.01)

        print("[Patrol] Proceed deadline reached with no detection.")
        return False

    return proceed_condition


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
                                enable_spiral_search: bool = False,
                                spiral_radius_deg: float = 0.5,
                                spiral_points_per_rotation: int = 16,
                                spiral_rotations: int = 2):
    """
    If full_circle=True, ignore time sampling and generate the entire orbital plane
    from RAAN & inclination (read from shared_data["rann"], ["inclination"]).
    Otherwise, sample the orbit over a time window via datahandler.
    """
    if full_circle:
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
                                  enable_spiral_search=enable_spiral_search,
                                  spiral_radius_deg=spiral_radius_deg,
                                  spiral_points_per_rotation=spiral_points_per_rotation,
                                  spiral_rotations=spiral_rotations)
        return

    # --- time-sampled fallback (existing behavior) ---
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
                              enable_spiral_search=enable_spiral_search,
                              spiral_radius_deg=spiral_radius_deg,
                              spiral_points_per_rotation=spiral_points_per_rotation,
                              spiral_rotations=spiral_rotations)