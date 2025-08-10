# ==============================================================================
# motors/motor_utils.py (NEW FILE)
# ------------------------------------------------------------------------------
# This file contains all the low-level, direct motor control functions.
# It has no knowledge of the EKF or high-level tracking logic. This decoupling
# prevents circular import errors.
# ==============================================================================

import numpy as np
import pigpio
from time import sleep

# --- Pin Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625


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
    target_degrees = max(0, min(90, target_degrees))

    # Use a simple direct set for speed in tracking, smooth move is less critical here.
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