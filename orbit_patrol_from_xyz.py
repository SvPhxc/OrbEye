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


# ---------- spiral search helpers ----------
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

        d_el = radius * math.sin(angle)
        d_az = radius * math.cos(angle) / cos_el

        az = (center_az + d_az + 360.0) % 360.0
        el = max(-90.0, min(90.0, center_el + d_el))
        waypoints.append((az, el))
    return waypoints


def _perform_spiral_search(shared_data, center_az, center_el, spiral_radius_deg, points_per_rotation, num_rotations,
                           spiral_settle_s):
    """
    Performs a single spiral scan, returning the position of the highest LiDAR strength.
    """
    spiral_wps = generate_spiral_waypoints(center_az, center_el, spiral_radius_deg, points_per_rotation, num_rotations)

    best_pos = (center_az, center_el)
    max_strength = -1.0

    print(f"[Patrol.Spiral] Searching {len(spiral_wps)} points, radius {spiral_radius_deg}°")

    for az, el in spiral_wps:
        if _should_abort(shared_data):
            break

        # MODIFIED: Use the new, faster settle time for spirals
        if not _goto_and_wait(shared_data, az, el, settle_timeout_s=spiral_settle_s):
            continue

        time.sleep(0.005)

        strength = float(shared_data["lidar_data"][1])
        ts = float(shared_data["lidar_data"][2])

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
                if isinstance(res, (bool, np.bool_)):
                    return bool(res)
                if isinstance(res, (tuple, list)) and len(res):
                    for v in res:
                        if isinstance(v, (bool, np.bool_)):
                            return bool(v)
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
        "Expected one of: " + ", ".join(candidates)
    )


def make_detection_proceed_condition(clutter_filter, shared_data,
                                     confirm_hits=3, max_age_s=0.5):
    """
    Returns proceed_condition(az, el, deadline_s) -> bool
    Waits until the ClutterFilter reports foreground at the current waypoint.
    """
    detector = _bind_cf_detector(clutter_filter)

    return make_detection_proceed_condition_from_callable(
        detector, shared_data, confirm_hits, max_age_s
    )


# ---------- main patrol ----------
def _search_with_continuous_spiral(shared_data, center_az, center_el, deadline_s,
                                   detector, confirm_hits, max_age_s,
                                   spiral_radius_deg, spiral_points_per_rotation, spiral_rotations, spiral_settle_s):
    """
    Continuously spirals around a point, checking for detection, until deadline.
    Returns (detected, (found_az, found_el)).
    """
    ok_streak = 0
    last_seen_ts = -1.0

    spiral_wps = generate_spiral_waypoints(
        center_az, center_el, spiral_radius_deg, spiral_points_per_rotation, spiral_rotations
    )
    wp_idx = 0

    print(f"[Patrol.Search] Beginning continuous spiral until {deadline_s:.1f}s")

    while time.monotonic() < deadline_s:
        if _should_abort(shared_data):
            return False, (center_az, center_el)

        az, el = spiral_wps[wp_idx]
        # MODIFIED: Use the new, faster settle time for spirals
        _goto_and_wait(shared_data, az, el, settle_timeout_s=spiral_settle_s)
        wp_idx = (wp_idx + 1) % len(spiral_wps)

        dwell_deadline = time.monotonic() + 0.2
        while time.monotonic() < dwell_deadline:
            if _should_abort(shared_data):
                return False, (center_az, center_el)

            dist_cm, strength, ts = [float(v) for v in shared_data["lidar_data"]]
            if ts > 0 and (time.time() - ts) <= max_age_s:
                if detector(az, el, dist_cm, strength):
                    if ts != last_seen_ts:
                        ok_streak += 1
                        last_seen_ts = ts
                    if ok_streak >= confirm_hits:
                        print(f"[Patrol.Search] Detection confirmed at ({az:.2f}°, {el:.2f}°)")
                        return True, (az, el)
                else:
                    ok_streak = 0
            time.sleep(0.001)

    print("[Patrol.Search] Deadline reached with no confirmed detection.")
    return False, (center_az, center_el)


def patrol_waypoints(shared_data,
                     waypoints,
                     proceed_condition=None,
                     dwell_seconds: float = 2.0,
                     settle_timeout_s: float = 3.0,
                     max_wait_s: float | None = None,
                     next_wp_speed_deg_per_s: float | None = None,
                     enable_spiral_search: bool = True,
                     continuous_spiral_search: bool = True,
                     spiral_radius_deg: float = 4,
                     spiral_points_per_rotation: int = 16,
                     spiral_rotations: int = 2,
                     spiral_settle_s: float = 0.001):  # MODIFIED: Added parameter
    """
    For each waypoint: slew+settle -> [search] -> wait until detection or timeout -> next.
    """
    n = len(waypoints)
    if n == 0:
        return

    wps = [list(wp) for wp in waypoints]
    first_az, first_el = wps[0]
    if _should_abort(shared_data): return
    _goto_and_wait(shared_data, first_az, first_el, settle_timeout_s=settle_timeout_s)

    for i in range(n):
        if _should_abort(shared_data): break

        nominal_az, nominal_el = wps[i]
        detected = False

        derived_timeout = None
        if next_wp_speed_deg_per_s and next_wp_speed_deg_per_s > 1e-6 and n > 1:
            nxt = wps[(i + 1) % n]
            arc_deg = approx_arc_deg(nominal_az, nominal_el, nxt[0], nxt[1])
            derived_timeout = max(0.5, arc_deg / float(next_wp_speed_deg_per_s))

        if max_wait_s is not None and derived_timeout is not None:
            deadline = time.monotonic() + min(max_wait_s, derived_timeout)
        elif max_wait_s is not None:
            deadline = time.monotonic() + max_wait_s
        elif derived_timeout is not None:
            deadline = time.monotonic() + derived_timeout
        else:
            deadline = time.monotonic() + dwell_seconds

        can_continuous_search = (
                continuous_spiral_search and
                proceed_condition is not None and
                hasattr(proceed_condition, 'detector') and
                hasattr(proceed_condition, 'confirm_hits')
        )

        if can_continuous_search:
            detected, (current_az, current_el) = _search_with_continuous_spiral(
                shared_data, nominal_az, nominal_el, deadline,
                detector=proceed_condition.detector,
                confirm_hits=proceed_condition.confirm_hits,
                max_age_s=proceed_condition.max_age_s,
                spiral_radius_deg=spiral_radius_deg,
                spiral_points_per_rotation=spiral_points_per_rotation,
                spiral_rotations=spiral_rotations,
                spiral_settle_s=spiral_settle_s  # MODIFIED: Pass parameter down
            )
        else:
            current_az, current_el = nominal_az, nominal_el
            _goto_and_wait(shared_data, current_az, current_el, settle_timeout_s=settle_timeout_s)

            if enable_spiral_search:
                (found_az, found_el), max_str = _perform_spiral_search(
                    shared_data, current_az, current_el,
                    spiral_radius_deg, spiral_points_per_rotation, spiral_rotations,
                    spiral_settle_s=spiral_settle_s  # MODIFIED: Pass parameter down
                )
                if max_str > 0:
                    d_az = _az_short_diff(current_az, found_az)
                    d_el = found_el - current_el
                    if abs(d_az) > 0.01 or abs(d_el) > 0.01:
                        print(f"[Patrol] Path corrected by (d_az={d_az:.2f}°, d_el={d_el:.2f}°)")
                        for j in range(i + 1, n):
                            wps[j][0] = (wps[j][0] + d_az + 360.0) % 360.0
                            wps[j][1] = max(-90.0, min(90.0, wps[j][1] + d_el))
                    current_az, current_el = found_az, found_el
                    _goto_and_wait(shared_data, current_az, current_el, settle_timeout_s=1.0)

            if proceed_condition is None:
                while time.monotonic() < deadline and not _should_abort(shared_data):
                    time.sleep(0.01)
            else:
                detected = bool(proceed_condition(current_az, current_el, deadline_s=deadline))

        # Decide if recording is enabled
        record_ok = True if not shared_data.get("record_tle_points") else shared_data["record_tle_points"].value

        if detected and record_ok:
            # Use ACTUAL encoder angles at detection time
            cur_az = float(shared_data["stepper_degrees"].value) % 360.0
            cur_el = float(shared_data["servo_degrees"].value)

            # Fresh LiDAR
            dist_cm, strength, ts = [float(v) for v in shared_data["lidar_data"]]

            # 1) Single-slot for heatmap
            try:
                with shared_data["satellite_points"].get_lock():
                    shared_data["satellite_points"][:] = [cur_az, cur_el, dist_cm, strength, ts]
                    print(f"[Patrol] saved -> satellite_points ({cur_az:.2f}°, {cur_el:.2f}°)")
            except Exception:
                shared_data["satellite_points"][:] = [cur_az, cur_el, dist_cm, strength, ts]
                print(f"[Patrol] saved -> satellite_points (no lock) ({cur_az:.2f}°, {cur_el:.2f}°)")

            # 2) Append to history (for later TLE fit)
            try:
                shared_data["tracking_history"].append([cur_az, cur_el, dist_cm, strength, ts])
                print(f"[Patrol] appended to tracking_history (N={len(shared_data['tracking_history'])})")
            except Exception as e:
                print(f"[Patrol] WARNING: couldn't append to tracking_history: {e}")

            if shared_data.get("satellite_detected"):
                shared_data["satellite_detected"].value = True

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
                              continuous_spiral_search: bool = False,
                              spiral_radius_deg: float = 0.5,
                              spiral_points_per_rotation: int = 16,
                              spiral_rotations: int = 2,
                              spiral_settle_s: float = 0.001):  # MODIFIED: Added parameter
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
                     continuous_spiral_search=continuous_spiral_search,
                     spiral_radius_deg=spiral_radius_deg,
                     spiral_points_per_rotation=spiral_points_per_rotation,
                     spiral_rotations=spiral_rotations,
                     spiral_settle_s=spiral_settle_s)  # MODIFIED: Pass parameter down

    print("[OrbitPatrolXYZ] Patrol complete.")


def make_detection_proceed_condition_from_callable(detector, shared_data,
                                                   confirm_hits=3, max_age_s=0.5):
    """
    Returns proceed_condition(az, el, deadline_s) that waits until detector is True.
    It also attaches the detector and its parameters for use in other modes.
    """

    def proceed_condition(az, el, deadline_s):
        ok_streak = 0
        last_seen_ts = -1.0
        import time as _t
        print(f"[Patrol] Waiting for detection @ ({az:.2f}°, {el:.2f}°) "
              f"confirm_hits={confirm_hits}, max_age={max_age_s}s")
        while _t.monotonic() < deadline_s:
            if _should_abort(shared_data):
                print("[Patrol] Aborted proceed-condition wait.")
                return False

            dist_cm, strength, ts = [float(v) for v in shared_data["lidar_data"]]
            if ts > 0 and (_t.time() - ts) <= max_age_s:
                if bool(detector(az, el, dist_cm, strength)):
                    if ts != last_seen_ts:
                        ok_streak += 1
                        last_seen_ts = ts
                        print(f"[Patrol]   hit {ok_streak}/{confirm_hits} "
                              f"(dist={dist_cm:.1f}cm, str={strength:.0f}, ts={ts:.3f})")
                    if ok_streak >= confirm_hits:
                        print("[Patrol] Detection confirmed.")
                        return True
                else:
                    if ok_streak > 0:
                        print("[Patrol]   streak reset.")
                    ok_streak = 0
            _t.sleep(0.01)
        print("[Patrol] Proceed deadline reached with no detection.")
        return False

    proceed_condition.detector = detector
    proceed_condition.confirm_hits = confirm_hits
    proceed_condition.max_age_s = max_age_s
    return proceed_condition


def run_orbit_patrol_from_query(shared_data,
                                query: str,
                                num_points: int = 9,
                                dwell_seconds: float = 0.0,
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
                                enable_spiral_search: bool = True,
                                continuous_spiral_search: bool = True,
                                spiral_radius_deg: float = 4,
                                spiral_points_per_rotation: int = 8,
                                spiral_rotations: int = 0,
                                spiral_settle_s: float = 0.001):  # MODIFIED: Added parameter
    """
    If full_circle=True, generate the entire orbital plane.
    Otherwise, sample the orbit over a time window via datahandler.
    """
    if full_circle:
        raan = float(shared_data["rann"].value)
        incl = float(shared_data["inclination"].value)
        pts_unit = generate_full_circle_xyz_from_raan_incl(
            raan, incl, n_samples=full_circle_samples
        )
        print(f"[OrbitPatrolXYZ] Full-circle mode: RAAN={raan:.2f}°, inc={incl:.2f}°,"
              f" samples={len(pts_unit)}")
        run_orbit_patrol_from_xyz(shared_data, pts_unit,
                                  num_points=num_points, dwell_seconds=dwell_seconds,
                                  min_el_deg=min_el_deg, max_el_deg=max_el_deg,
                                  start_near_current=start_near_current,
                                  proceed_condition=proceed_condition,
                                  next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                                  max_wait_s=max_wait_s,
                                  enable_spiral_search=enable_spiral_search,
                                  continuous_spiral_search=continuous_spiral_search,
                                  spiral_radius_deg=spiral_radius_deg,
                                  spiral_points_per_rotation=spiral_points_per_rotation,
                                  spiral_rotations=spiral_rotations,
                                  spiral_settle_s=spiral_settle_s)  # MODIFIED: Pass parameter down
        return

    from datahandler import get_orbit_xyz_for_query
    name, pts_km = get_orbit_xyz_for_query(query,
                                           duration_minutes=duration_minutes,
                                           step_seconds=step_seconds)
    print(f"[OrbitPatrolXYZ] Using orbit '{name}', {len(pts_km)} samples.")
    run_orbit_patrol_from_xyz(shared_data, pts_km,
                              num_points=num_points, dwell_seconds=dwell_seconds,
                              min_el_deg=min_el_deg, max_el_deg=max_el_deg,
                              start_near_current=start_near_current,
                              proceed_condition=proceed_condition,
                              next_wp_speed_deg_per_s=next_wp_speed_deg_per_s,
                              max_wait_s=max_wait_s,
                              enable_spiral_search=enable_spiral_search,
                              continuous_spiral_search=continuous_spiral_search,
                              spiral_radius_deg=spiral_radius_deg,
                              spiral_points_per_rotation=spiral_points_per_rotation,
                              spiral_rotations=spiral_rotations,
                              spiral_settle_s=spiral_settle_s)  # MODIFIED: Pass parameter down