import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic, time
import math
import signal


# --- Helper functions for coordinate conversion ---
def spherical_to_cartesian(az_rad, el_rad, dist):
    """Converts spherical coordinates (azimuth, elevation, distance) to Cartesian (x, y, z)."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to spherical (azimuth, elevation, distance)."""
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return az, el, dist


# --- Constants for GPIO pins ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625


# (stepper_worker is unchanged)
def stepper_worker(movement_queue, shared_data):
    print("[WORKER] Stepper worker started.")
    pi = pigpio.pi()
    if not pi.connected:
        print("[WORKER] pigpio not connected.")
        return

    pulse_wave_id = -1
    try:
        us_delay = 500
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)

        pi.wave_add_generic([
            pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay),
            pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)
        ])
        pulse_wave_id = pi.wave_create()

        while not shared_data['shutdown'].value:
            try:
                command = movement_queue.get(timeout=0.1)
            except Exception:
                continue
            if command is None:
                break

            direction, degrees_to_move, _ = command
            ideal_microsteps = degrees_to_move / MICROSTEP_ANGLE
            total = ideal_microsteps + shared_data['cumulative_error'].value
            actual_steps = round(total)
            shared_data['cumulative_error'].value = total - actual_steps
            if actual_steps == 0:
                continue

            pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
            repeats_lsb = actual_steps % 256
            repeats_msb = actual_steps // 256
            chain = [255, 0, pulse_wave_id, 255, 1, repeats_lsb, repeats_msb]
            pi.wave_chain(chain)

            while pi.wave_tx_busy():
                if shared_data['shutdown'].value:
                    break
                sleep(0.01)

            # update shared az
            current_pos = shared_data['stepper_degrees'].value
            actual_deg = actual_steps * MICROSTEP_ANGLE
            new_pos = (current_pos - actual_deg) if direction == 'left' else (current_pos + actual_deg)
            shared_data['stepper_degrees'].value = new_pos % 360

    finally:
        try:
            pi.wave_tx_stop()
        except:
            pass
        try:
            if pulse_wave_id != -1:
                pi.wave_delete(pulse_wave_id)
        except:
            pass
        if pi.connected:
            pi.stop()
        print("[WORKER] Stepper worker shutting down.")


# --- Servo and Movement Functions (unchanged) ---
def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    """Moves the servo smoothly using software-timed PWM pulses."""
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if step == 0: return

    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step)
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09) + (28 / 0.09)
        pi.set_PWM_dutycycle(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

    final_pulse_width = 500 + (target_degrees / 0.09) + (28 / 0.09)
    pi.set_PWM_dutycycle(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees


def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)


def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value;
    current_tilt = shared_data["servo_degrees"].value;
    adjusted_azimuth = target_azimuth % 360;
    adjusted_elevation = max(0, min(180, target_elevation))
    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1: move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue,
                                  shared_data)
    if abs(adjusted_elevation - current_tilt) > 1: smooth_servo_move(pi, adjusted_elevation, shared_data)


def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING HIGH-FIDELITY CONCENTRIC RING SEARCH ---")
    pan_direction = 1;
    initial_pan_angle = shared_data['stepper_degrees'].value
    for radius in range(int(90.0), -1, -int(1.5)):
        if shared_data['shutdown'].value: break
        smooth_servo_move(pi, 90.0 - radius, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        scan_frequency_hz = 1778
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000)
        degrees_per_second = scan_frequency_hz * MICROSTEP_ANGLE;
        duration = 360.0 / degrees_per_second
        start_time = monotonic()
        while monotonic() - start_time < duration:
            if shared_data['shutdown'].value: break
            elapsed_time = monotonic() - start_time;
            degrees_turned = elapsed_time * degrees_per_second
            current_pan = (initial_pan_angle + degrees_turned * pan_direction) % 360
            with shared_data['stepper_degrees'].get_lock():
                shared_data['stepper_degrees'].value = current_pan
            sleep(0.01)
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        initial_pan_angle = (initial_pan_angle + 360 * pan_direction) % 360
        shared_data['stepper_degrees'].value = initial_pan_angle
        if shared_data['shutdown'].value: break
        pan_direction *= -1
    print("\n--- HIGH-FIDELITY SEARCH FINISHED ---");
    smooth_servo_move(pi, 90.0, shared_data);
    return True


# --- NEW: Intelligent Point Acquisition Logic ---

def wait_for_detection(shared_data, timeout=2.0):
    """
    Checks for a satellite detection flag, captures the point, and resets the flag.
    Returns the point as a dictionary or None on timeout.
    """
    start_time = monotonic()
    while monotonic() - start_time < timeout:
        if shared_data['shutdown'].value: return None
        if shared_data['satellite_detected'].value:
            # Atomically read the point and reset the flag
            with shared_data["satellite_points"].get_lock():
                point = {
                    "az": shared_data["satellite_points"][0],
                    "el": shared_data["satellite_points"][1],
                    "str": shared_data["satellite_points"][2],
                    "dist": shared_data["satellite_points"][3] / 100.0,  # cm -> m
                    "time": time()
                }
                shared_data['satellite_detected'].value = False
            return point
        sleep(0.01)  # Be a good citizen
    return None


def spiral_for_one_point(pi, shared_data, movement_queue):
    """
    Performs a tight outward spiral, returning the first valid point detected.
    """
    center_az = shared_data['stepper_degrees'].value
    center_el = shared_data['servo_degrees'].value

    radius_start = 1.0  # degrees
    radius_max = 15.0  # Max search radius before giving up
    radius_step = 0.02  # How much to increase radius per angle step
    angle_step = 10.0  # How many degrees to turn per step

    current_angle = 0.0
    while (radius_start + radius_step * current_angle) < radius_max:
        if shared_data['shutdown'].value: return None

        # Check for a detection from the LiDAR process
        detection = wait_for_detection(shared_data, timeout=0.01)
        if detection:
            return detection

        # Archimedean spiral formula
        r = radius_start + radius_step * current_angle
        az = center_az + r * math.cos(math.radians(current_angle))
        el = max(0, min(90, center_el + r * math.sin(math.radians(current_angle))))

        track_target(pi, az, el, 0.0001, movement_queue, shared_data)
        sleep(0.03)  # Dwell time for the LiDAR to get a reading

        current_angle += angle_step

    return None  # Failed to find a point within the spiral


def intelligent_acquire_three(pi, shared_data, movement_queue):
    """
    Implements the 3-phase intelligent acquisition strategy for EKF initialization.
    """
    print("[Acquisition] Starting 3-point intelligent acquisition.")
    shared_data["points_count"].value = 0
    acquired_points = []

    # --- PHASE 1: Find the best first point with the highest strength ---
    print("[Acquisition] Phase 1: Finding initial best point...")
    center_az = shared_data['stepper_degrees'].value
    center_el = shared_data['servo_degrees'].value
    search_radius = 4.0  # Search 4 degrees around the current position
    candidate_points = []

    # Perform a small 3x3 grid search around the current point
    for el_offset in np.linspace(-search_radius, search_radius, 3):
        for az_offset in np.linspace(-search_radius, search_radius, 3):
            if shared_data['shutdown'].value: return
            track_target(pi, center_az + az_offset, center_el + el_offset, 0.0001, movement_queue, shared_data)
            sleep(0.3)  # Dwell time to get a stable reading

            detection = wait_for_detection(shared_data, timeout=0.2)
            if detection:
                candidate_points.append(detection)
                print(
                    f"  > Found candidate: az={detection['az']:.1f}, el={detection['el']:.1f}, str={detection['str']:.0f}")

    if not candidate_points:
        print("[Acquisition] Phase 1 FAILED: No points detected in initial search.")
        return

    # Select the point with the highest signal strength
    first_point = max(candidate_points, key=lambda p: p['str'])
    acquired_points.append(first_point)
    print(f"[Acquisition] Phase 1 SUCCESS. Best point: str={first_point['str']:.0f}")

    # --- PHASE 2: Determine motion vector with a spiral search ---
    print("\n[Acquisition] Phase 2: Spiraling to find second point...")
    track_target(pi, first_point['az'], first_point['el'], 0.0001, movement_queue, shared_data)

    second_point = spiral_for_one_point(pi, shared_data, movement_queue)

    if not second_point:
        print("[Acquisition] Phase 2 FAILED: No second point found during spiral search.")
        return

    acquired_points.append(second_point)
    print(f"[Acquisition] Phase 2 SUCCESS. Found second point at az={second_point['az']:.1f}")

    # --- PHASE 3: Predict and capture the third point ---
    print("\n[Acquisition] Phase 3: Predicting third point...")
    p1 = acquired_points[0]
    p2 = acquired_points[1]

    try:
        p1_xyz = np.array(spherical_to_cartesian(np.deg2rad(p1['az']), np.deg2rad(p1['el']), p1['dist']))
        p2_xyz = np.array(spherical_to_cartesian(np.deg2rad(p2['az']), np.deg2rad(p2['el']), p2['dist']))

        dt = p2['time'] - p1['time']
        if dt < 0.02: dt = 0.1  # Avoid division by zero, assume minimum time delta

        velocity_xyz = (p2_xyz - p1_xyz) / dt

        # Predict 0.5 seconds into the future
        prediction_dt = 0.5
        p3_predicted_xyz = p2_xyz + velocity_xyz * prediction_dt

        az_pred_rad, el_pred_rad, _ = cartesian_to_spherical(*p3_predicted_xyz)
        az_pred_deg = np.rad2deg(az_pred_rad)
        el_pred_deg = np.rad2deg(el_pred_rad)

        print(f"  > Pointing to predicted location: az={az_pred_deg:.1f}, el={el_pred_deg:.1f}")
        track_target(pi, az_pred_deg, el_pred_deg, 0.0001, movement_queue, shared_data)

        # Dwell and wait for detection at the predicted spot
        third_point = wait_for_detection(shared_data, timeout=2.0)

        if not third_point:
            print("[Acquisition] Phase 3 FAILED: Did not detect drone at predicted location.")
            return

        acquired_points.append(third_point)
        print(f"[Acquisition] Phase 3 SUCCESS. Captured third point at az={third_point['az']:.1f}")

    except Exception as e:
        print(f"[Acquisition] Phase 3 ERROR: {e}")
        return

    # --- Finalize: Load points into shared buffer for EKF ---
    if len(acquired_points) == 3:
        print("\n[Acquisition] ACQUISITION COMPLETE. Loading buffer for EKF initialization.")
        with shared_data["points_count"].get_lock(), shared_data["points_buffer"].get_lock():
            for i, point in enumerate(acquired_points):
                base = 4 * i
                shared_data["points_buffer"][base + 0] = point['az']
                shared_data["points_buffer"][base + 1] = point['el']
                shared_data["points_buffer"][base + 2] = point['dist']
                shared_data["points_buffer"][base + 3] = point['str']
            shared_data["points_count"].value = 3
            shared_data["ekf_start"].value = True  # Signal EKF to initialize
    else:
        print("[Acquisition] FAILED to acquire 3 valid points.")


# --- Main Process and GPIO Setup ---

def initialize_gpio():
    GPIO.setwarnings(False);
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT);
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH);
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)


def _graceful_stop(signum, frame, shared_data):
    try:
        shared_data['shutdown'].value = True
    except Exception:
        pass


def run_motor_control(shared_data, movement_queue):
    signal.signal(signal.SIGINT, lambda s, f: _graceful_stop(s, f, shared_data))
    signal.signal(signal.SIGTERM, lambda s, f: _graceful_stop(s, f, shared_data))

    print("[MotorControl] Starting...", flush=True)
    initialize_gpio()
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio not connected", flush=True)
        return

    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
    pi.set_PWM_frequency(SERVO_PIN, 50)
    pi.set_PWM_range(SERVO_PIN, 20000)

    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.start()

    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value:
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True
            if shared_data['tilt_up'].value: move(pi, 'up', 5.0, None, movement_queue, shared_data); shared_data[
                'tilt_up'].value = False
            if shared_data['tilt_down'].value: move(pi, 'down', 5.0, None, movement_queue, shared_data); shared_data[
                'tilt_down'].value = False
            if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data); shared_data[
                'pan_left'].value = False
            if shared_data['pan_right'].value: move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data); shared_data[
                'pan_right'].value = False
            if shared_data["go_to_target"].value:
                track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001,
                             movement_queue, shared_data)
                shared_data["go_to_target"].value = False

            # --- MODIFIED BEHAVIOR ---
            if shared_data["acquire_points"].value:
                intelligent_acquire_three(pi, shared_data, movement_queue)
                shared_data["acquire_points"].value = False  # Reset trigger after attempt

            if shared_data['ekf_running'].value:
                track_target(pi,
                             shared_data["predicted_azimuth"].value,
                             shared_data["predicted_elevation"].value,
                             0.0001, movement_queue, shared_data)

            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...", flush=True)
        try:
            print("[MotorControl] Returning to home position (90, 90)...", flush=True)
            track_target(pi, 90, 90, 0.0001, movement_queue, shared_data)
        except Exception as e:
            print(f"[MotorControl] Could not return to home: {e}", flush=True)

        try:
            pi.wave_tx_stop()
            pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            pi.set_PWM_dutycycle(SERVO_PIN, 0)
        except Exception:
            pass

        try:
            movement_queue.put_nowait(None)
        except Exception:
            pass
        stepper_process.join(timeout=3)
        if stepper_process.is_alive():
            print("[MotorControl] WARNING: stepper worker did not exit cleanly.", flush=True)

        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.stop()
        GPIO.cleanup()