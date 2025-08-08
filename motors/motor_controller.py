import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic, time
import math

# --- Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

# --- State Definitions ---
STATE_SEARCHING = 0
STATE_CENTERING_P1 = 1
STATE_SPIRAL_P2 = 2
STATE_PREDICT_P3 = 3
STATE_COMPLETE = 4
STATE_TRACKING = 5


# --- NEW: Added the missing initialize_gpio function ---
def initialize_gpio():
    """Sets up the GPIO pins for the stepper motor driver."""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH)
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)


# --- Stepper Motor Worker Process ---
def stepper_worker(pi, movement_queue, shared_data):
    print("[WORKER] Stepper worker started.")
    pulse_wave_id = -1
    try:
        us_delay = 500
        pi.wave_add_generic(
            [pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)])
        pulse_wave_id = pi.wave_create()
        while not shared_data['shutdown'].value:
            try:
                command = movement_queue.get(timeout=0.1)
                if command is None: continue
            except Exception:
                continue

            direction, degrees_to_move, _ = command
            ideal_microsteps = degrees_to_move / MICROSTEP_ANGLE
            total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
            actual_microsteps_to_take = round(total_microsteps_to_consider)
            shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take

            if actual_microsteps_to_take == 0: continue

            pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
            repeats_lsb = actual_microsteps_to_take % 256
            repeats_msb = actual_microsteps_to_take // 256
            chain = [255, 0, pulse_wave_id, 255, 1, repeats_lsb, repeats_msb]
            pi.wave_chain(chain)

            while pi.wave_tx_busy():
                sleep(0.01)

            current_pos = shared_data['stepper_degrees'].value
            actual_degrees_this_move = actual_microsteps_to_take * MICROSTEP_ANGLE
            new_pos = (current_pos - actual_degrees_this_move) if direction == 'left' else (
                        current_pos + actual_degrees_this_move)
            shared_data['stepper_degrees'].value = new_pos % 360
    finally:
        if pulse_wave_id != -1 and pi.connected:
            pi.wave_delete(pulse_wave_id)
        print("[WORKER] Stepper worker shutting down.")


# --- Servo and Pan/Tilt Movement Functions ---
def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if abs(target_degrees - current_degrees) < 1: return

    for degrees in range(int(round(current_degrees)), int(round(target_degrees)), step):
        pulse_width = 500 + (degrees / 0.09) + (28 / 0.09)
        pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

    final_pulse_width = 500 + (target_degrees / 0.09) + (28 / 0.09)
    pi.set_servo_pulsewidth(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees


def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)


def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value
    current_tilt = shared_data["servo_degrees"].value
    adjusted_azimuth = target_azimuth % 360
    adjusted_elevation = max(0, min(180, target_elevation))

    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1:
        move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue, shared_data)

    if abs(adjusted_elevation - current_tilt) > 1:
        smooth_servo_move(pi, adjusted_elevation, shared_data)


def concentric_ring_search_smooth(pi, shared_data):
    # This function remains unchanged.
    pass


# --- State Machine Logic Functions (from previous step) ---
def center_for_point_1(pi, shared_data, movement_queue):
    print("[Acquisition] STATE 1: Centering for Point 1...")
    with shared_data["best_strength_point"].get_lock():
        shared_data["best_strength_point"][0] = 0.0
        shared_data["best_strength_point"][1] = 0.0
        shared_data["best_strength_point"][2] = 0.0
    with shared_data["satellite_points"].get_lock():
        center_az = shared_data["satellite_points"][0]
        center_el = shared_data["satellite_points"][1]

    scan_width = 10.0
    start_az = center_az - (scan_width / 2)
    print(f"[Acquisition] Scanning from {start_az:.1f}° to {start_az + scan_width:.1f}°")
    track_target(pi, start_az, center_el, 0.0001, movement_queue, shared_data)
    sleep(0.5)

    move(pi, 'right', scan_width, 0.0001, movement_queue, shared_data)
    sleep(1.0)

    with shared_data["best_strength_point"].get_lock():
        best_az = shared_data["best_strength_point"][0]
        best_el = shared_data["best_strength_point"][1]
        best_str = shared_data["best_strength_point"][2]

    if best_str > 0:
        print(f"[Acquisition] Found peak strength {best_str} at Az: {best_az:.1f}, El: {best_el:.1f}")
        track_target(pi, best_az, best_el, 0.0001, movement_queue, shared_data)
        sleep(0.2)
        with shared_data["initial_points"].get_lock(), shared_data["lidar_data"].get_lock():
            shared_data["initial_points"][0:4] = [best_az, best_el, shared_data["lidar_data"][0], time()]
        shared_data["acquisition_state"].value = STATE_SPIRAL_P2
    else:
        print("[Acquisition] Failed to find peak strength. Resetting.")
        shared_data["acquisition_state"].value = STATE_SEARCHING


def spiral_for_point_2(pi, shared_data, movement_queue):
    print("[Acquisition] STATE 2: Spiraling for Point 2...")
    with shared_data["initial_points"].get_lock():
        center_az, center_el = shared_data["initial_points"][0:2]
    shared_data["satellite_detected"].value = False

    t, start_time = 0.0, monotonic()
    while monotonic() - start_time < 5.0:
        if shared_data['shutdown'].value: break
        if shared_data["satellite_detected"].value:
            print("[Acquisition] Point 2 acquired.")
            with shared_data["initial_points"].get_lock(), shared_data["satellite_points"].get_lock():
                shared_data["initial_points"][4:8] = [shared_data["satellite_points"][0],
                                                      shared_data["satellite_points"][1],
                                                      shared_data["satellite_points"][3], time()]
            shared_data["acquisition_state"].value = STATE_PREDICT_P3
            return
        r = 1.0 + 0.05 * t
        if r > 15.0: break
        az = center_az + r * math.cos(math.radians(t))
        el = max(0, min(90, center_el + r * math.sin(math.radians(t))))
        track_target(pi, az, el, 0.0001, movement_queue, shared_data)
        sleep(0.01)
        t += 5.0

    if shared_data["acquisition_state"].value == STATE_SPIRAL_P2:
        print("[Acquisition] Failed to find Point 2. Resetting.")
        shared_data["acquisition_state"].value = STATE_SEARCHING


def predict_point_3(pi, shared_data, movement_queue):
    print("[Acquisition] STATE 3: Predicting Point 3...")
    with shared_data["initial_points"].get_lock():
        p1_az, _, _, p1_time = shared_data["initial_points"][0:4]
        p2_az, p2_el, _, p2_time = shared_data["initial_points"][4:8]
    dt = p2_time - p1_time
    if dt < 0.01:
        shared_data["acquisition_state"].value = STATE_SEARCHING
        return
    az_vel = (p2_az - p1_az) / dt
    el_vel = (p2_el - shared_data["initial_points"][1]) / dt
    pred_az = p2_az + az_vel * dt
    pred_el = p2_el + el_vel * dt
    track_target(pi, pred_az, pred_el, 0.0001, movement_queue, shared_data)
    shared_data["satellite_detected"].value = False

    start_time = monotonic()
    while monotonic() - start_time < 3.0:
        if shared_data["satellite_detected"].value:
            print("[Acquisition] Point 3 acquired!")
            with shared_data["initial_points"].get_lock(), shared_data["satellite_points"].get_lock():
                shared_data["initial_points"][8:12] = [shared_data["satellite_points"][0],
                                                       shared_data["satellite_points"][1],
                                                       shared_data["satellite_points"][3], time()]
            shared_data["acquisition_state"].value = STATE_COMPLETE
            return
        sleep(0.05)
    print("[Acquisition] Failed to find Point 3. Resetting.")
    shared_data["acquisition_state"].value = STATE_SEARCHING


# --- MAIN MOTOR CONTROL PROCESS ---
def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...")
    initialize_gpio()
    pi = pigpio.pi()
    if not pi.connected: return
    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
    pi.set_servo_pulsewidth(SERVO_PIN, 1500)  # Center servo

    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data))
    stepper_process.start()

    try:
        while not shared_data['shutdown'].value:
            state = shared_data['acquisition_state'].value
            # ... (manual controls as before) ...

            if shared_data['acquire_points'].value:
                if state == STATE_SEARCHING and shared_data["satellite_detected"].value:
                    shared_data['acquisition_state'].value = STATE_CENTERING_P1
                elif state == STATE_CENTERING_P1:
                    center_for_point_1(pi, shared_data, movement_queue)
                elif state == STATE_SPIRAL_P2:
                    spiral_for_point_2(pi, shared_data, movement_queue)
                elif state == STATE_PREDICT_P3:
                    predict_point_3(pi, shared_data, movement_queue)
                elif state == STATE_COMPLETE:
                    print("\n--- ACQUISITION COMPLETE: Initializing EKF. ---\n")
                    shared_data["ekf_start"].value = True
                    shared_data["acquire_points"].value = False
                    shared_data["acquisition_state"].value = STATE_TRACKING

            elif state == STATE_TRACKING and shared_data['ekf_running'].value:
                track_target(pi, shared_data["predicted_azimuth"].value, shared_data["predicted_elevation"].value,
                             0.0001, movement_queue, shared_data)

            else:
                if not shared_data['acquire_points'].value and state not in [STATE_SEARCHING, STATE_TRACKING]:
                    shared_data['acquisition_state'].value = STATE_SEARCHING

            sleep(0.05)

    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None)
        stepper_process.join()
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop servo
            pi.stop()
        GPIO.cleanup()