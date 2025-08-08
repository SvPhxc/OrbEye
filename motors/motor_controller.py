# File: motors/motor_controller.py
import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math
import signal

# --- NEW: Define constants for GPIO pins for clarity ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

# (stepper_worker is unchanged)
def stepper_worker(movement_queue, shared_data):
    print("[WORKER] Stepper worker started.")
    pi = pigpio.pi()                  # <-- create our own client
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
        except: pass
        try:
            if pulse_wave_id != -1:
                pi.wave_delete(pulse_wave_id)
        except: pass
        if pi.connected:
            pi.stop()
        print("[WORKER] Stepper worker shutting down.")


# --- CHANGE 1: MODIFY SERVO FUNCTION TO USE SOFTWARE PWM ---
def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    """Moves the servo smoothly using software-timed PWM pulses."""
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if step == 0: return

    # Loop to create a smooth movement effect
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step)
    for degrees in degrees_range:
        # Calculate the required pulse width in microseconds (500-2500 is typical for servos)
        pulse_width = 500 + (degrees / 0.09) + (28/0.09)
        # Instead of `set_servo_pulsewidth`, we use `set_PWM_dutycycle`.
        # This uses software timing and will not conflict with the hardware PWM on the stepper pin.
        pi.set_PWM_dutycycle(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)
        
    # Send the final pulse width to ensure it lands exactly on the target.
    final_pulse_width = 500 + (target_degrees / 0.09) + (28/0.09)
    pi.set_PWM_dutycycle(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees

# (move and track_target are unchanged, they just call the modified servo function)
def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']: movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)
def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value; current_tilt = shared_data["servo_degrees"].value; adjusted_azimuth = target_azimuth % 360; adjusted_elevation = max(0, min(180, target_elevation))
    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1: move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue, shared_data)
    if abs(adjusted_elevation - current_tilt) > 1: smooth_servo_move(pi, adjusted_elevation, shared_data)

# (concentric_ring_search_smooth is unchanged, it correctly uses hardware PWM for the stepper)
def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING HIGH-FIDELITY CONCENTRIC RING SEARCH ---")
    pan_direction = 1; initial_pan_angle = shared_data['stepper_degrees'].value
    for radius in range(int(90.0), -1, -int(1.5)):
        if shared_data['shutdown'].value: break
        smooth_servo_move(pi, 90.0 - radius, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        scan_frequency_hz = 1778 # This can now be changed without affecting the servo.
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000)
        degrees_per_second = scan_frequency_hz * MICROSTEP_ANGLE; duration = 360.0 / degrees_per_second
        start_time = monotonic()
        while monotonic() - start_time < duration:
            if shared_data['shutdown'].value: break
            elapsed_time = monotonic() - start_time; degrees_turned = elapsed_time * degrees_per_second
            current_pan = (initial_pan_angle + degrees_turned * pan_direction) % 360
            with shared_data['stepper_degrees'].get_lock(): shared_data['stepper_degrees'].value = current_pan
            sleep(0.01)
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0); pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        initial_pan_angle = (initial_pan_angle + 360 * pan_direction) % 360
        shared_data['stepper_degrees'].value = initial_pan_angle
        if shared_data['shutdown'].value: break
        pan_direction *= -1
    print("\n--- HIGH-FIDELITY SEARCH FINISHED ---"); smooth_servo_move(pi, 90.0, shared_data); return True

def spiral_acquire_three(pi, shared_data, movement_queue):
    """Tight outward spiral, stop once 3 validated points are captured by LiDAR process."""
    center_az = shared_data['stepper_degrees'].value
    center_el = shared_data['servo_degrees'].value

    radius = 0.5   # degrees
    turns = 2
    step = 0.5
    direction = 1  # keep current pan direction
    shared_data["points_count"].value = 0

    for t in np.arange(0.0, turns*360.0, step):
        if shared_data['shutdown'].value: break
        if shared_data["points_count"].value >= 3: break

        # simple Archimedean spiral
        r = radius + 0.01*t
        az = center_az + direction * r * math.cos(math.radians(t))
        el = max(0, min(90, center_el + r * math.sin(math.radians(t))))

        track_target(pi, az, el, 0.0001, movement_queue, shared_data)
        sleep(0.03)

def initialize_gpio():
    GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT); GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH); GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

def _graceful_stop(signum, frame, shared_data):
    try:
        shared_data['shutdown'].value = True
    except Exception:
        pass

def run_motor_control(shared_data, movement_queue):
    # catch Ctrl-C / SIGTERM so we flip the flag instead of dying
    signal.signal(signal.SIGINT,  lambda s,f: _graceful_stop(s,f,shared_data))
    signal.signal(signal.SIGTERM, lambda s,f: _graceful_stop(s,f,shared_data))

    print("[MotorControl] Starting...", flush=True)
    initialize_gpio()
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio not connected", flush=True)
        return

    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)

    # Servo via software PWM
    pi.set_PWM_frequency(SERVO_PIN, 50)
    pi.set_PWM_range(SERVO_PIN, 20000)

    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.daemon = False
    stepper_process.start()

    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value: concentric_ring_search_smooth(pi, shared_data); shared_data['scan_trigger'].value = False; shared_data['save_background'].value = True
            if shared_data['tilt_up'].value: move(pi, 'up', 5.0, None, movement_queue, shared_data); shared_data['tilt_up'].value = False
            if shared_data['tilt_down'].value: move(pi, 'down', 5.0, None, movement_queue, shared_data); shared_data['tilt_down'].value = False
            if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data); shared_data['pan_left'].value = False
            if shared_data['pan_right'].value: move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data); shared_data['pan_right'].value = False
            if shared_data["go_to_target"].value: track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001, movement_queue, shared_data); shared_data["go_to_target"].value = False
            if shared_data["acquire_points"].value:
                spiral_acquire_three(pi, shared_data, movement_queue)
                shared_data["acquire_points"].value = False
                if shared_data["points_count"].value >= 3:
                    shared_data["ekf_start"].value = True
            if shared_data['ekf_running'].value:
                track_target(pi,
                    shared_data["predicted_azimuth"].value,
                    shared_data["predicted_elevation"].value,
                    0.0001, movement_queue, shared_data)
            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...", flush=True)
        # Don’t enqueue movements here—worker may already be exiting
        try:
            print("[MotorControl] Returning to home position (0,0)...", flush=True)
            track_target(pi, 0, 0, 0.0001, movement_queue, shared_data)
        except Exception as e:
            print(f"[MotorControl] Could not return to home: {e}", flush=True)
        try:
            pigpio.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        except Exception:
            pass
        try:
            pi.set_PWM_dutycycle(SERVO_PIN, 0)
        except Exception:
            pass

        # tell worker to stop and wait briefly
        try:
            movement_queue.put_nowait(None)
        except Exception:
            pass
        stepper_process.join(timeout=3)
        if stepper_process.is_alive():
            print("[MotorControl] WARNING: stepper worker didn’t exit", flush=True)

        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.stop()
        GPIO.cleanup()
