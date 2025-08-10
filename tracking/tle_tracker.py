# tracking/tle_tracker.py
"""
TLE-guided acquisition & EKF-assisted tracking module.

Public API:
- acquire_target_from_tle(pi, shared_data, movement_queue, tle_data) -> bool
- track_target_with_ekf(pi, shared_data, movement_queue) -> bool

Design notes:
- Relies on run_lidar process to perform background-based detection when
  shared_data['acquire_points'] or shared_data['ekf_running'] is True.
- Uses motor_utils.track_target and movement_queue for stepper motion.
- Carefully uses locks when reading/writing shared multiprocessing Arrays/Values.
"""

import time
import math
import numpy as np
from collections import deque

# Import motor utilities from your codebase
from motors.motor_utils import track_target, move, MICROSTEP_ANGLE

# Tunable parameters (sane defaults)
COARSE_HALF_SPAN_DEG = 15.0        # 30x30 deg window
COARSE_STEP_DEG = 3.0
REFINE_RADIUS_DEG = 2.0
REFINE_STEP_DEG = 0.4
HILLCLIMB_RADIUS_DEG = 3.8
HILLCLIMB_STEP_DEG = 0.4
DWELL_TIME_S = 0.04                # time to wait for LiDAR to stabilise after motion
LIDAR_STALE_TOL_S = 0.12
MOVE_PUT_RATE_LIMIT_S = 0.02       # minimum interval between movement_queue.put() calls
MAX_ACQUISITION_TIME_S = 45.0
MIN_POINT_SEPARATION_S = 0.18      # minimum time between point1 and point2
MAX_REFINEMENT_ATTEMPTS = 3
HILLCLIMB_MAX_ITERS = 25
PAN_TOLERANCE_DEG = 0.25
TILT_TOLERANCE_DEG = 0.2

def _shortest_angular_delta(target, current):
    """Return signed shortest delta in degrees (-180, 180]."""
    return ((target - current + 540.0) % 360.0) - 180.0

def _normalize_az(az):
    return az % 360.0

def _clamp_tilt(el):
    return max(0.0, min(90.0, el))

def _read_lidar(shared):
    """Return (distance_cm, strength, ts) with lidar_data lock held for consistency."""
    arr = shared["lidar_data"]
    with arr.get_lock():
        d, s, ts = float(arr[0]), float(arr[1]), float(arr[2])
    return d, s, ts

def _write_satellite_point(shared, az, el, strength, distance_cm):
    sp = shared["satellite_points"]
    with sp.get_lock():
        sp[0], sp[1], sp[2], sp[3] = float(az), float(el), float(strength), float(distance_cm)
    shared["satellite_detected"].value = True

def _populate_points_buffer(shared, points):
    """
    points: sequence of dicts with keys ['az','el','distance_m','strength','timestamp']
    writes them into shared['points_buffer'] and sets points_count atomically.
    """
    buf = shared["points_buffer"]
    count = shared["points_count"]
    with count.get_lock():
        # write values (5 floats per point)
        for i, p in enumerate(points):
            base = i * 5
            # Ensure we write floats into the Array
            buf[base + 0] = float(p['az'])
            buf[base + 1] = float(p['el'])
            buf[base + 2] = float(p['distance_m'])
            buf[base + 3] = float(p['strength'])
            buf[base + 4] = float(p['timestamp'])
        count.value = len(points)

def _wait_for_fresh_lidar(shared, old_ts, timeout_s):
    """Wait until lidar_data timestamp changes (fresh sample) or timeout. Return new sample dict or None."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        d, s, ts = _read_lidar(shared)
        if ts != old_ts:
            az = float(shared["stepper_degrees"].value)
            el = float(shared["servo_degrees"].value)
            return {'az': az, 'el': el, 'distance_m': d / 100.0, 'strength': s, 'timestamp': ts}
        time.sleep(0.002)
    return None

def _wait_until_pose(shared, target_az, target_el, timeout_s):
    """Block until turret reaches pose within tolerance or timeout. Returns True if reached."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shared["shutdown"].value:
        cur_az = float(shared["stepper_degrees"].value)
        cur_el = float(shared["servo_degrees"].value)
        if abs(_shortest_angular_delta(target_az, cur_az)) <= PAN_TOLERANCE_DEG and abs(target_el - cur_el) <= TILT_TOLERANCE_DEG:
            return True
        time.sleep(0.005)
    return False

def _generate_spiral_grid(center_az, center_el, half_span_deg, step_deg):
    """
    Yield (az, el) positions covering a square region centered at (center_az, center_el).
    Spiral ordering: center first, then rings outward.
    """
    center_az = _normalize_az(center_az)
    center_el = float(center_el)
    max_offset_steps = int(math.ceil(half_span_deg / step_deg))
    # start with (0,0)
    yield center_az, _clamp_tilt(center_el)
    for layer in range(1, max_offset_steps + 1):
        # traverse square ring: top edge (left->right), right edge (top->bottom), bottom (right->left), left (bottom->top)
        # offsets from -layer..layer
        az0 = center_az
        for dx in range(-layer, layer + 1):
            for dy in [ -layer, layer ]:
                az = _normalize_az(az0 + dx * step_deg)
                el = _clamp_tilt(center_el + dy * step_deg)
                yield az, el
        for dy in range(-layer + 1, layer):
            for dx in [ -layer, layer ]:
                az = _normalize_az(az0 + dx * step_deg)
                el = _clamp_tilt(center_el + dy * step_deg)
                yield az, el

class TLETracker:
    def __init__(self, pi, shared_data, movement_queue, config=None):
        self.pi = pi
        self.shared = shared_data
        self.q = movement_queue
        self.last_put_time = 0.0
        self.config = config or {}
        # read thresholds from shared state where appropriate
        # nothing to precompute here

    def _rate_limited_move(self, az, el, block=True, timeout_s=1.0):
        """
        Move the turret to (az,el). For pan we use movement_queue with rate-limiting.
        For tilt call track_target which will also call smooth_servo_move as needed.
        Returns True if queued and (if block) reached pose within time.
        """
        if self.shared["shutdown"].value:
            return False

        # compute pan delta and decide whether to issue a stepper movement
        cur_az = float(self.shared["stepper_degrees"].value)
        cur_el = float(self.shared["servo_degrees"].value)
        delta_pan = _shortest_angular_delta(az, cur_az)
        need_pan = abs(delta_pan) > PAN_TOLERANCE_DEG
        need_tilt = abs(el - cur_el) > TILT_TOLERANCE_DEG

        # If pan movement is large, break into chunks to avoid waves too big for the stepper worker
        max_chunk_deg = 30.0  # conservative chunking (keeps waves moderate)
        if need_pan:
            remaining = delta_pan
            sign = 1.0 if remaining > 0 else -1.0
            while abs(remaining) > 0.0 and not self.shared["shutdown"].value:
                chunk = sign * min(abs(remaining), max_chunk_deg)
                # queue the chunk
                now = time.monotonic()
                wait_for_rate = max(0.0, MOVE_PUT_RATE_LIMIT_S - (now - self.last_put_time))
                if wait_for_rate > 0:
                    time.sleep(wait_for_rate)
                # movement_queue expects ('left'/'right', degrees, delay)
                direction = 'right' if chunk > 0 else 'left'
                degrees = abs(chunk)
                try:
                    self.q.put((direction, float(degrees), 0.0001))
                except Exception:
                    # Queue might be closed; fallback to direct track_target call
                    track_target(self.pi, az, el, 0.0001, self.q, self.shared)
                    break
                self.last_put_time = time.monotonic()
                remaining -= chunk
                # small pause between chains to avoid flooding wave_chain
                time.sleep(0.005)

        # tilt: use track_target's servo control (it will do tilt via smooth_servo_move)
        if need_tilt and not need_pan:
            # If no pan was needed (or we already queued pan), just tilt now
            track_target(self.pi, az, el, 0.0001, self.q, self.shared)
        elif need_tilt and need_pan:
            # call track_target once to ensure tilt is set to the final desired tilt
            track_target(self.pi, az, el, 0.0001, self.q, self.shared)

        if block:
            return _wait_until_pose(self.shared, az, el, timeout_s)
        return True

    def _coarse_search_and_detect(self, seed_az, seed_el, max_time_s=MAX_ACQUISITION_TIME_S):
        """
        Coarse grid search using a spiral generator. As we move, rely on run_lidar's detection
        (shared['satellite_detected']) to notify us. Return first detection as a sample dict or None.
        """
        deadline = time.monotonic() + max_time_s
        # ensure detection logic in lidar runs: caller should have set shared['acquire_points'] = True
        grid = _generate_spiral_grid(seed_az, seed_el, COARSE_HALF_SPAN_DEG, COARSE_STEP_DEG)
        for az, el in grid:
            if time.monotonic() > deadline or self.shared["shutdown"].value:
                return None
            # move (block briefly)
            ok = self._rate_limited_move(az, el, block=True, timeout_s=0.8)
            if not ok:
                continue
            # wait for fresh lidar sample
            _, _, old_ts = _read_lidar(self.shared)
            sample = _wait_for_fresh_lidar(self.shared, old_ts, LIDAR_STALE_TOL_S)
            if sample is None:
                continue
            # check if run_lidar's detection flag has been set
            if self.shared["satellite_detected"].value:
                # read satellite_points (atomic) to get detection az/el/strength/distance
                sp = self.shared["satellite_points"]
                with sp.get_lock():
                    det_az, det_el, det_strength, det_range_cm = float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])
                # sanity check: strength & range acceptance
                min_m, max_m = self.shared["lidar_acceptance_range"]
                distance_m = det_range_cm / 100.0
                min_ok = (min_m <= distance_m <= max_m)
                min_strength = 1000 if self.shared["debug_mode"].value else 5000
                if det_strength >= min_strength and min_ok:
                    return {'az': det_az, 'el': det_el, 'strength': det_strength, 'distance_m': distance_m, 'timestamp': time.time()}
                else:
                    # reset detection flag and continue searching
                    self.shared["satellite_detected"].value = False
                    continue
            else:
                # fallback heuristic: large strength spike vs threshold
                _, strength, _ = _read_lidar(self.shared)
                min_strength = 1000 if self.shared["debug_mode"].value else 5000
                if strength >= min_strength:
                    # read current turret az/el for the sample
                    az_cur = float(self.shared['stepper_degrees'].value)
                    el_cur = float(self.shared['servo_degrees'].value)
                    return {'az': az_cur, 'el': el_cur, 'strength': strength, 'distance_m': _read_lidar(self.shared)[0] / 100.0, 'timestamp': time.time()}

        return None

    def _refine_target_local(self, rough_az, rough_el, radius_deg=REFINE_RADIUS_DEG, step_deg=REFINE_STEP_DEG, timeout_s=4.0):
        """
        Local hill-climb / refine candidate: search a dense grid around rough_az/rough_el
        and return the best-scoring measurement (based on LiDAR strength and range).
        """
        best = None
        start = time.monotonic()
        # iterate candidate offsets
        offsets_az = np.arange(-radius_deg, radius_deg + 1e-6, step_deg)
        offsets_el = np.arange(-radius_deg, radius_deg + 1e-6, step_deg)
        for d_az in offsets_az:
            for d_el in offsets_el:
                if self.shared["shutdown"].value: return None
                if time.monotonic() - start > timeout_s:
                    break
                cand_az = _normalize_az(rough_az + d_az)
                cand_el = _clamp_tilt(rough_el + d_el)
                # Move to candidate
                self._rate_limited_move(cand_az, cand_el, block=True, timeout_s=0.6)
                # get fresh lidar
                _, _, old_ts = _read_lidar(self.shared)
                sample = _wait_for_fresh_lidar(self.shared, old_ts, LIDAR_STALE_TOL_S)
                if sample is None:
                    continue
                # Evaluate candidate
                strength = sample['strength']
                distance_m = sample['distance_m']
                min_m, max_m = self.shared["lidar_acceptance_range"]
                if not (min_m <= distance_m <= max_m):
                    continue
                # Apply threshold
                min_strength = 1000 if self.shared["debug_mode"].value else 5000
                if strength < min_strength:
                    continue
                # If better than best, keep it
                if (best is None) or (strength > best['strength']):
                    best = {'az': sample['az'], 'el': sample['el'], 'distance_m': distance_m, 'strength': strength, 'timestamp': sample['timestamp']}
        # Move to best final pose
        if best:
            self._rate_limited_move(best['az'], best['el'], block=True, timeout_s=0.6)
        return best

    def acquire_target_from_tle(self, tle_data):
        """
        High-level acquisition invoked when shared_data['acquire_points'] is True.
        Returns True on success and populates shared['points_buffer'] with 3 points.
        """
        print("[TLE-TRACKER] Starting acquisition from TLE...")
        start_time = time.monotonic()
        # Ensure lidar process knows it's in acquisition mode (so it can run background detection)
        self.shared["acquire_points"].value = True

        # Seed: try TLE-based seed. If tle_data missing, fallback to (current pan, current tilt) or (180,45)
        try:
            if tle_data:
                # integrate with your existing tracking.tle_utils.get_tle_prediction if present
                from tracking.tle_utils import get_tle_prediction
                seed_az, seed_el = get_tle_prediction(tle_data, None)
            else:
                # if debug mode, use current pose as seed; otherwise fallback to midpoint
                if self.shared["debug_mode"].value:
                    seed_az = float(self.shared["stepper_degrees"].value)
                    seed_el = float(self.shared["servo_degrees"].value)
                else:
                    seed_az, seed_el = 180.0, 45.0
        except Exception:
            seed_az, seed_el = 180.0, 45.0

        # Coarse search + detection
        detection = self._coarse_search_and_detect(seed_az, seed_el, max_time_s=min(MAX_ACQUISITION_TIME_S, 20.0))
        if not detection:
            print("[TLE-TRACKER] Coarse search failed to find a candidate.")
            self.shared["acquire_points"].value = False
            return False

        # Point 1: refine locally around detection
        for attempt in range(MAX_REFINEMENT_ATTEMPTS):
            p1 = self._refine_target_local(detection['az'], detection['el'], REFINE_RADIUS_DEG, REFINE_STEP_DEG, timeout_s=3.0)
            if p1:
                print(f"[TLE-TRACKER] Point1 acquired: Str {p1['strength']:.0f} @ ({p1['az']:.2f},{p1['el']:.2f})")
                break
            else:
                print("[TLE-TRACKER] Refinement attempt failed, retrying coarse detection fallback...")
                time.sleep(0.1)
                detection = self._coarse_search_and_detect(seed_az, seed_el, max_time_s=6.0)
                if not detection:
                    break
        if not p1:
            print("[TLE-TRACKER] Could not refine point1.")
            self.shared["acquire_points"].value = False
            return False

        # Wait briefly to get velocity estimate (point2)
        time.sleep(max(0.2, MIN_POINT_SEPARATION_S))
        # Attempt reacquire: search tightly around p1
        p2 = self._refine_target_local(p1['az'], p1['el'], REFINE_RADIUS_DEG, REFINE_STEP_DEG, timeout_s=3.0)
        if (not p2) or (p2['timestamp'] - p1['timestamp'] < MIN_POINT_SEPARATION_S):
            # try one more time
            p2 = self._refine_target_local(p1['az'], p1['el'], REFINE_RADIUS_DEG, REFINE_STEP_DEG, timeout_s=3.0)
        if not p2:
            print("[TLE-TRACKER] Could not acquire point2.")
            self.shared["acquire_points"].value = False
            return False
        print(f"[TLE-TRACKER] Point2 acquired: Str {p2['strength']:.0f} @ ({p2['az']:.2f},{p2['el']:.2f})")

        # Predict point3 based on simple linear extrapolation of angular velocities
        dt = p2['timestamp'] - p1['timestamp'] if (p2['timestamp'] - p1['timestamp']) > 1e-3 else 0.5
        delta_az = _shortest_angular_delta(p2['az'], p1['az'])
        vel_az = delta_az / dt
        vel_el = (p2['el'] - p1['el']) / dt
        pred_az_p3 = _normalize_az(p2['az'] + vel_az * 0.7)  # 0.7s ahead as used in your acquisition
        pred_el_p3 = _clamp_tilt(p2['el'] + vel_el * 0.7)

        # Try refining at predicted position; fallback to local refine around p2 if needed
        p3 = self._refine_target_local(pred_az_p3, pred_el_p3, REFINE_RADIUS_DEG, REFINE_STEP_DEG, timeout_s=3.0)
        if not p3:
            # fallback
            p3 = self._refine_target_local(p2['az'], p2['el'], REFINE_RADIUS_DEG, REFINE_STEP_DEG, timeout_s=3.0)
        if not p3:
            print("[TLE-TRACKER] Could not acquire point3.")
            self.shared["acquire_points"].value = False
            return False
        print(f"[TLE-TRACKER] Point3 acquired: Str {p3['strength']:.0f} @ ({p3['az']:.2f},{p3['el']:.2f})")

        # Populate points_buffer in shared memory for EKF initialization
        pts = [p1, p2, p3]
        _populate_points_buffer(self.shared, [{'az': p['az'], 'el': p['el'], 'distance_m': p['distance_m'], 'strength': p['strength'], 'timestamp': p['timestamp']} for p in pts])
        # finished acquisition: leave acquisition flag False (motor_controller will handle handoff)
        self.shared["acquire_points"].value = False
        print("[TLE-TRACKER] Acquisition successful; points_buffer populated.")
        return True

    def track_target_with_ekf(self):
        """
        Called repeatedly while EKF is running. Uses predicted_azimuth/predicted_elevation
        from shared memory, does a fast movement to predicted pose, performs a tight
        hill-climb to maximize LiDAR strength, and writes a refined measurement into
        shared['satellite_points'] so that run_lidar/run_ekf_tracker can consume it.
        Returns True when a valid refined measurement was produced; False otherwise.
        """
        if not self.shared["ekf_running"].value or self.shared["ekf_initialized"].value is False:
            return False

        pred_az = float(self.shared["predicted_azimuth"].value)
        pred_el = float(self.shared["predicted_elevation"].value)
        # Move quickly to predicted pose (block briefly)
        moved = self._rate_limited_move(pred_az, pred_el, block=True, timeout_s=0.4)
        if not moved:
            # Too slow or shutdown: abort this cycle
            return False

        # Hill-climb / micro-scan around predicted pose to find local max strength
        best = None
        center_az = pred_az
        center_el = pred_el
        iters = 0
        radius = min(HILLCLIMB_RADIUS_DEG, 4.0)
        step = HILLCLIMB_STEP_DEG
        # small grid search first, keep it fast (limit iterations)
        offsets = np.arange(-radius, radius + 1e-6, step)
        for d_az in offsets:
            for d_el in offsets:
                if self.shared["shutdown"].value or iters >= HILLCLIMB_MAX_ITERS:
                    break
                cand_az = _normalize_az(center_az + d_az)
                cand_el = _clamp_tilt(center_el + d_el)
                # Skip unnecessary tiny moves if we're already extremely close
                cur_az = float(self.shared['stepper_degrees'].value)
                cur_el = float(self.shared['servo_degrees'].value)
                if abs(_shortest_angular_delta(cand_az, cur_az)) < 0.15 and abs(cand_el - cur_el) < 0.15:
                    # Already at this approximate pose; don't issue move
                    pass
                else:
                    # Move (non-blocking small moves are OK but we want to wait a tiny bit)
                    self._rate_limited_move(cand_az, cand_el, block=True, timeout_s=0.18)
                # collect a fresh sample
                _, _, old_ts = _read_lidar(self.shared)
                sample = _wait_for_fresh_lidar(self.shared, old_ts, LIDAR_STALE_TOL_S)
                if sample is None:
                    iters += 1
                    continue
                # qualify sample
                min_m, max_m = self.shared['lidar_acceptance_range']
                if not (min_m <= sample['distance_m'] <= max_m):
                    iters += 1
                    continue
                min_strength = 1000 if self.shared["debug_mode"].value else 4000
                if sample['strength'] < min_strength:
                    iters += 1
                    continue
                # Accept sample as candidate if strength higher
                if (best is None) or (sample['strength'] > best['strength']):
                    best = {'az': sample['az'], 'el': sample['el'], 'distance_m': sample['distance_m'], 'strength': sample['strength'], 'timestamp': sample['timestamp']}
                iters += 1
            if self.shared["shutdown"].value or iters >= HILLCLIMB_MAX_ITERS:
                break

        if best:
            # Move turret to best pose (ensure pose + fresh measurement)
            self._rate_limited_move(best['az'], best['el'], block=True, timeout_s=0.25)
            # Wait for fresh lidar reading at that pose
            _, _, old_ts = _read_lidar(self.shared)
            sample = _wait_for_fresh_lidar(self.shared, old_ts, LIDAR_STALE_TOL_S)
            if sample is None:
                # No fresh sample to report; still publish best based on last read
                _write_satellite_point(self.shared, best['az'], best['el'], best['strength'], best['distance_m'] * 100.0)
            else:
                _write_satellite_point(self.shared, sample['az'], sample['el'], sample['strength'], sample['distance_m'] * 100.0)
            # signal that a measurement is available (run_ekf_tracker will pick up lidar_data + current pan/tilt)
            return True
        else:
            # No good candidate found
            return False

# Module-level convenience functions to match requested API
def acquire_target_from_tle(pi, shared_data, movement_queue, tle_data):
    tt = TLETracker(pi, shared_data, movement_queue)
    return tt.acquire_target_from_tle(tle_data)

def track_target_with_ekf(pi, shared_data, movement_queue):
    tt = TLETracker(pi, shared_data, movement_queue)
    return tt.track_target_with_ekf()
