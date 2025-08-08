import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic, time
import math
import signal


# --- Helper functions for coordinate conversion (unchanged) ---
def spherical_to_cartesian(az_rad, el_rad, dist):
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return az, el, dist


# --- Constants (unchanged) ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

# --- NEW: State machine constants ---
ACQUIRE_IDLE = 0
ACQUIRE_START = 1
ACQUIRE_PHASE1_SEARCH = 2
ACQUIRE_PHASE2_SPIRAL = 3
ACQUIRE_PHASE3_PREDICT = 4
ACQUIRE_FINALIZE = 5
ACQUIRE_FAILED = 6


# (stepper_worker is unchanged)
def stepper_worker(movement_queue, shared_data):
    # ... (no changes in this function)
    print("[WORKER] Stepper worker started.")
    pi = pigpio.pi()  # <-- create our own client
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
    # ... (no changes in this function)
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if step == 0: return

    # Loop to create a smooth movement effect
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step)
    for degrees in degrees_range:
        # Calculate the required pulse width in microseconds (500-2500 is typical for servos)
        pulse_width = 500 + (degrees / 0.09) + (28 / 0.09)
        # Instead of `set_servo_pulsewidth`, we use `set_PWM_dutycycle`.
        # This uses software timing and will not conflict with the hardware PWM on the stepper pin.
        pi.set_PWM_dutycycle(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

    # Send the final pulse width to ensure it lands exactly on the target.
    final_pulse_width = 500 + (target_degrees / 0.09) + (28 / 0.09)
    pi.set_PWM_dutycycle(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees


def move(pi, direction, degrees, delay, movement_queue, shared_data):
    # ... (no changes in this function)
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)


def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    # ... (no changes in this function)
    current_pan = shared_data["stepper_degrees"].value;
    current_tilt = shared_data["servo_degrees"].value;
    adjusted_azimuth = target_azimuth % 360;
    adjusted_elevation = max(0, min(180, target_elevation))
    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1: move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue,
                                  shared_data)
    if abs(adjusted_elevation - current_tilt) > 1: smooth_servo_move(pi, adjusted_elevation, shared_data)


def concentric_ring_search_smooth(pi, shared_data):
    # ... (no changes in this function)
    print("\n--- STARTING HIGH-FIDELITY CONCENTRIC RING SEARCH ---")
    pan_direction = 1;
    initial_pan_angle = shared_data['stepper_degrees'].value
    for radius in range(int(90.0), -1, -int(1.5)):
        if shared_data['shutdown'].value: break
        smooth_servo_move(pi, 90.0 - radius, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        scan_frequency_hz = 1778  # This can now be changed without affecting the servo.
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


# --- NEW: Non-Blocking Acquisition State Machine ---

def check_for_detection(shared_data):
    """ Non-blocking check for a satellite detection flag. """
    if shared_data['satellite_detected'].value:
        with shared_data["satellite_points"].get_lock():
            point = {
                "az": shared_data["satellite_points"][0], "el": shared_data["satellite_points"][1],
                "str": shared_data["satellite_points"][2], "dist": shared_data["satellite_points"][3] / 100.0,
                "time": time()
            }
            shared_data['satellite_detected'].value = False
        return point
    return None


def manage_acquisition_state(pi, shared_data, movement_queue):
    """
    Manages the non-blocking state machine for 3-point acquisition.
    This function should be called in every loop of run_motor_control.
    """
    state = shared_data['acquisition_state']
    internal = shared_data['acquisition_internal_state']
    points_buf = shared_data['points_buffer']
    points_count = shared_data['points_count']

    now = time()

    # --- State 1: Initialize Acquisition ---
    if state.value == ACQUIRE_START:
        print("[Acquisition] Starting 3-point acquisition.")
        points_count.value = 0
        internal[0] = now + 0.1  # next_action_time
        internal[1] = 0  # grid_index / spiral_angle
        internal[2] = shared_data['stepper_degrees'].value  # center_az
        internal[3] = shared_data['servo_degrees'].value  # center_el
        # Clear candidate buffer (represented by point 3 in the main buffer)
        points_buf[8] = 0  # Best strength found so far
        state.value = ACQUIRE_PHASE1_SEARCH
        return

    # --- State 2: Search for the Best First Point ---
    if state.value == ACQUIRE_PHASE1_SEARCH:
        if now < internal[0]: return  # Non-blocking delay

        grid_index = int(internal[1])
        if grid_index >= 9:  # Finished 3x3 grid
            if points_buf[8] > 0:  # Check if best_strength was updated
                print(f"[Acquisition] Phase 1 SUCCESS. Best point strength: {points_buf[8]:.0f}")
                # The best point is already stored, just increment count
                points_count.value = 1
                internal[0] = now + 0.1  # next_action_time
                internal[1] = 0  # reset for spiral_angle
                state.value = ACQUIRE_PHASE2_SPIRAL
            else:
                print("[Acquisition] Phase 1 FAILED: No points detected.")
                state.value = ACQUIRE_FAILED
            return

        # Move to next grid point
        search_radius = 4.0
        el_offset = np.linspace(-search_radius, search_radius, 3)[grid_index // 3]
        az_offset = np.linspace(-search_radius, search_radius, 3)[grid_index % 3]
        track_target(pi, internal[2] + az_offset, internal[3] + el_offset, 0.0001, movement_queue, shared_data)

        # Check for detection
        detection = check_for_detection(shared_data)
        if detection and detection['str'] > points_buf[8]:  # points_buf[8] stores best_strength
            print(f"  > Found new best candidate: str={detection['str']:.0f}")
            points_buf[0], points_buf[1], points_buf[2], points_buf[3] = detection['az'], detection['el'], detection[
                'dist'], detection['str']
            points_buf[8] = detection['str']  # Update best strength

        internal[1] += 1  # Move to next grid point
        internal[0] = now + 0.4  # Dwell time
        return

    # --- State 3: Spiral for the Second Point ---
    if state.value == ACQUIRE_PHASE2_SPIRAL:
        if now < internal[0]: return

        angle = internal[1]
        radius = 1.0 + 0.02 * angle  # Archimedean spiral
        if radius > 15.0:
            print("[Acquisition] Phase 2 FAILED: Spiral radius exceeded.")
            state.value = ACQUIRE_FAILED
            return

        center_az, center_el = points_buf[0], points_buf[1]
        az = center_az + radius * math.cos(math.radians(angle))
        el = max(0, min(90, center_el + radius * math.sin(math.radians(angle))))
        track_target(pi, az, el, 0.0001, movement_queue, shared_data)

        detection = check_for_detection(shared_data)
        if detection:
            print(f"[Acquisition] Phase 2 SUCCESS. Found 2nd point at az={detection['az']:.1f}")
            points_buf[4], points_buf[5], points_buf[6], points_buf[7] = detection['az'], detection['el'], detection[
                'dist'], detection['str']
            points_count.value = 2

            # Immediately move to predict phase
            try:
                p1_xyz = np.array(
                    spherical_to_cartesian(np.deg2rad(points_buf[0]), np.deg2rad(points_buf[1]), points_buf[2]))
                p2_xyz = np.array(
                    spherical_to_cartesian(np.deg2rad(points_buf[4]), np.deg2rad(points_buf[5]), points_buf[6]))
                dt = detection['time'] - check_for_detection.p1_time  # Requires storing p1 time
                if dt < 0.02: dt = 0.1
                velocity_xyz = (p2_xyz - p1_xyz) / dt
                p3_predicted_xyz = p2_xyz + velocity_xyz * 0.5
                az_pred, el_pred, _ = cartesian_to_spherical(*p3_predicted_xyz)
                print(
                    f"[Acquisition] Phase 3: Predicting target at az={np.rad2deg(az_pred):.1f}, el={np.rad2deg(el_pred):.1f}")
                track_target(pi, np.rad2deg(az_pred), np.rad2deg(el_pred), 0.0001, movement_queue, shared_data)
                internal[0] = now + 2.0  # prediction_deadline
                state.value = ACQUIRE_PHASE3_PREDICT
            except Exception as e:
                print(f"[Acquisition] Phase 3 Prediction ERROR: {e}")
                state.value = ACQUIRE_FAILED
            return

        internal[1] += 10  # Increment angle for next spiral step
        internal[0] = now + 0.05
        return
    # Attach time of first point detection for dt calculation in phase 3
    if state.value == ACQUIRE_PHASE1_SEARCH and 'detection' in locals() and detection:
        check_for_detection.p1_time = detection['time']

    # --- State 4: Wait for Predicted Third Point ---
    if state.value == ACQUIRE_PHASE3_PREDICT:
        detection = check_for_detection(shared_data)
        if detection:
            print("[Acquisition] Phase 3 SUCCESS. Captured third point.")
            points_buf[8], points_buf[9], points_buf[10], points_buf[11] = detection['az'], detection['el'], detection[
                'dist'], detection['str']
            points_count.value = 3
            state.value = ACQUIRE_FINALIZE
            return

        if now > internal[0]:  # Check against prediction_deadline
            print("[Acquisition] Phase 3 FAILED: Timeout waiting for predicted point.")
            state.value = ACQUIRE_FAILED
            return

    # --- State 5: Finalize and Trigger EKF ---
    if state.value == ACQUIRE_FINALIZE:
        print("[Acquisition] COMPLETE. Triggering EKF.")
        shared_data["ekf_start"].value = True
        state.value = ACQUIRE_IDLE  # Reset for next time
        return

    # --- State 6: Handle Failure ---
    if state.value == ACQUIRE_FAILED:
        print("[Acquisition] Resetting state machine.")
        state.value = ACQUIRE_IDLE  # Reset
        return


# --- Main Process and GPIO Setup (largely unchanged) ---

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
            # --- MODIFIED: Trigger state machine instead of a blocking function ---
            if shared_data["acquire_points"].value:
                if shared_data['acquisition_state'].value == ACQUIRE_IDLE:
                    shared_data['acquisition_state'].value = ACQUIRE_START
                shared_data["acquire_points"].value = False  # Consume trigger

            # --- NEW: Always run the state machine manager ---
            if shared_data['acquisition_state'].value != ACQUIRE_IDLE:
                manage_acquisition_state(pi, shared_data, movement_queue)

            # All other controls only run if acquisition is idle
            elif shared_data['acquisition_state'].value == ACQUIRE_IDLE:
                if shared_data['scan_trigger'].value:
                    concentric_ring_search_smooth(pi, shared_data)
                    shared_data['scan_trigger'].value = False
                    shared_data['save_background'].value = True
                if shared_data['tilt_up'].value: move(pi, 'up', 5.0, None, movement_queue, shared_data); shared_data[
                    'tilt_up'].value = False
                if shared_data['tilt_down'].value: move(pi, 'down', 5.0, None, movement_queue, shared_data);
                shared_data['tilt_down'].value = False
                if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_left'].value = False
                if shared_data['pan_right'].value: move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_right'].value = False
                if shared_data["go_to_target"].value:
                    track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001,
                                 movement_queue, shared_data)
                    shared_data["go_to_target"].value = False
                if shared_data['ekf_running'].value:
                    track_target(pi,
                                 shared_data["predicted_azimuth"].value,
                                 shared_data["predicted_elevation"].value,
                                 0.0001, movement_queue, shared_data)

            sleep(0.02)  # Main loop can sleep briefly
    finally:
        # ... (shutdown logic is unchanged) ...
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