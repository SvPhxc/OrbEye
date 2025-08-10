# ==============================================================================
# motors/motor_controller.py
# ------------------------------------------------------------------------------
# Key Fixes:
# - The main `run_motor_control` loop no longer passively points the motors.
# - When the EKF is running, it now calls the `active_track_target` function
#   from the new `tracking.tracker` module to perform the hybrid search.
# ==============================================================================

import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math
import signal

# --- Import our new active tracking logic ---
from tracking.tracker import active_track_target
from tracking.acquisition import run_acquisition_sequence, run_manual_acquisition_sequence
from tracking.tle_utils import parse_tle_file

# --- Pin Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625


# region Low-Level Motor Functions
def stepper_worker(movement_queue, shared_data):
    """Dedicated process to handle stepper motor movements via pigpio waves."""
    print("[WORKER] Stepper worker started.")
    pi = pigpio.pi()
    if not pi.connected: return

    pulse_wave_id = -1
    try:
        us_delay = 500
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        pi.wave_add_generic([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay),
                             pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)])
        pulse_wave_id = pi.wave_create()

        while not shared_data['shutdown'].value:
            try:
                command = movement_queue.get(timeout=0.1)
                if command is None: break

                direction, degrees, _ = command
                total = (degrees / MICROSTEP_ANGLE) + shared_data['cumulative_error'].value
                steps = round(total)
                shared_data['cumulative_error'].value = total - steps
                if steps == 0: continue

                pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
                chain = [255, 0, pulse_wave_id, 255, 1, steps % 256, steps // 256]
                pi.wave_chain(chain)

                while pi.wave_tx_busy():
                    if shared_data['shutdown'].value: break
                    sleep(0.01)

                actual_deg = steps * MICROSTEP_ANGLE
                current_pos = shared_data['stepper_degrees'].value
                new_pos = (current_pos - actual_deg) if direction == 'left' else (current_pos + actual_deg)
                shared_data['stepper_degrees'].value = new_pos % 360
            except Exception:
                continue
    finally:
        if pi.connected:
            pi.wave_tx_stop()
            if pulse_wave_id != -1: pi.wave_delete(pulse_wave_id)
            pi.stop()
        print("[WORKER] Stepper worker shutting down.")


def smooth_servo_move(pi, target_degrees, shared_data):
    """Moves the servo smoothly to a target angle."""
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(90, target_degrees))  # Clamp to 0-90

    # Move in small increments for smoothness
    for deg in np.arange(current_degrees, target_degrees, np.sign(target_degrees - current_degrees)):
        pulse_width = 500 + (deg / 0.09) + (28 / 0.09)
        pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
        sleep(0.005)

    final_pulse_width = 500 + (target_degrees / 0.09) + (28 / 0.09)
    pi.set_servo_pulsewidth(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees


def move(pi, direction, degrees, delay, movement_queue, shared_data):
    """Generic move command for pan or tilt."""
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target, shared_data)


def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    """Low-level function to point the turret at a specific az/el coordinate."""
    current_pan = shared_data["stepper_degrees"].value
    delta_pan = (target_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1:
        move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue, shared_data)

    current_tilt = shared_data["servo_degrees"].value
    if abs(target_elevation - current_tilt) > 0.1:
        smooth_servo_move(pi, target_elevation, shared_data)


# endregion

def run_motor_control(shared_data, movement_queue):
    """Main process for controlling all physical movement and tracking logic."""
    signal.signal(signal.SIGINT, lambda s, f: shared_data['shutdown'].update(True))
    signal.signal(signal.SIGTERM, lambda s, f: shared_data['shutdown'].update(True))

    print("[MotorControl] Starting...")
    GPIO.setwarnings(False);
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT);
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH);
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

    pi = pigpio.pi()
    if not pi.connected: return

    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.start()

    try:
        while not shared_data['shutdown'].value:
            # --- Acquisition Logic ---
            if shared_data["acquire_points"].value:
                success = False
                if shared_data["debug_mode"].value:
                    success = run_manual_acquisition_sequence(pi, shared_data, movement_queue)
                else:
                    try:
                        tle_data = parse_tle_file("temp.tle")[0]
                        success = run_acquisition_sequence(pi, shared_data, movement_queue, tle_data)
                    except Exception as e:
                        print(f"[MotorControl] Could not run TLE acquisition: {e}")

                shared_data["acquire_points"].value = False
                if success:
                    shared_data["ekf_start"].value = True
                    print("[MotorControl] Handoff to EKF process initiated.")
                else:
                    print("[MotorControl] Acquisition sequence failed.")

            # --- HYBRID TRACKING LOGIC ---
            elif shared_data['ekf_running'].value:
                # Instead of just pointing, we now call the active search function.
                active_track_target(pi, shared_data, movement_queue)

            # --- Manual Override Logic ---
            else:
                if shared_data['tilt_up'].value: move(pi, 'up', 1.0, None, movement_queue, shared_data); shared_data[
                    'tilt_up'].value = False
                if shared_data['tilt_down'].value: move(pi, 'down', 1.0, None, movement_queue, shared_data);
                shared_data['tilt_down'].value = False
                if shared_data['pan_left'].value: move(pi, 'left', 1.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_left'].value = False
                if shared_data['pan_right'].value: move(pi, 'right', 1.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_right'].value = False
                if shared_data["go_to_target"].value: track_target(pi, shared_data["target_azimuth"].value,
                                                                   shared_data["target_elevation"].value, 0.0001,
                                                                   movement_queue, shared_data); shared_data[
                    "go_to_target"].value = False

            sleep(0.02)  # Main loop delay

    finally:
        print("[MotorControl] Shutting down...")
        try:
            movement_queue.put_nowait(None)
        except Exception:
            pass
        stepper_process.join(timeout=3)
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected: pi.stop()
        GPIO.cleanup()