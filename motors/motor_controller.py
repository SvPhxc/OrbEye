# File: motors/motor_controller.py

from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Stepper Worker and Manual Movement Functions (Unchanged and Correct) ---
# This stable worker is now used by the precise scan function.
def stepper_worker(pi, movement_queue, shared_data):
    print("[WORKER] Stepper worker started.")
    STEPPER_PULSE_PIN = 19; STEPPER_DIR_PIN = 3
    pulse_wave_id = -1
    try:
        us_delay = 500
        pi.wave_add_generic([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)])
        pulse_wave_id = pi.wave_create()
        while not shared_data['shutdown'].value:
            try:
                command = movement_queue.get(timeout=0.1)
                if command is None: break
            except Exception: continue
            direction, degrees_to_move, _ = command
            ideal_microsteps = degrees_to_move / 0.05625
            total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
            actual_microsteps_to_take = round(total_microsteps_to_consider)
            shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
            if actual_microsteps_to_take == 0: continue
            pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
            repeats_lsb = actual_microsteps_to_take % 256
            repeats_msb = actual_microsteps_to_take // 256
            chain = [255, 0, pulse_wave_id, 255, 1, repeats_lsb, repeats_msb]
            pi.wave_chain(chain)
            while pi.wave_tx_busy(): sleep(0.01)
            current_pos = shared_data['stepper_degrees'].value
            actual_degrees_this_move = actual_microsteps_to_take * 0.05625
            new_pos = (current_pos - actual_degrees_this_move) if direction == 'left' else (current_pos + actual_degrees_this_move)
            shared_data['stepper_degrees'].value = new_pos % 360
    finally:
        if pulse_wave_id != -1 and pi.connected: pi.wave_delete(pulse_wave_id)
        print("[WORKER] Stepper worker shutting down.")

def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if step == 0: return
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step)
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09)
        pi.set_servo_pulsewidth(13, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)
    final_pulse_width = 500 + (target_degrees / 0.09)
    pi.set_servo_pulsewidth(13, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees

def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']: movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)

def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value; current_tilt = shared_data["servo_degrees"].value
    adjusted_azimuth = target_azimuth % 360; adjusted_elevation = max(0, min(180, target_elevation))
    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1: move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue, shared_data)
    if abs(adjusted_elevation - current_tilt) > 1: smooth_servo_move(pi, adjusted_elevation, shared_data)


# --- CONCENTRIC SEARCH (REWRITTEN FOR ACCURACY) ---

CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 90.0
TILT_STEP_DEGREES = 2.0  # Can make this slightly larger if needed
PAN_STEP_DEGREES = 2.0   # The step size for each pan movement

def concentric_ring_search_precise(pi, movement_queue, shared_data):
    """
    Performs a precise "Step-Scan" for the background map.
    It prioritizes accuracy over speed by moving, stopping, then allowing a
    LiDAR reading at a known, stable position.
    """
    print("\n--- STARTING PRECISE STEP-SCAN FOR BACKGROUND MAP ---")
    pan_direction = 1
    
    # Scan from the horizon (0 deg elevation) up to the center (90 deg)
    elevation_range = np.arange(0, MAX_TILT_RADIUS + 1, TILT_STEP_DEGREES)

    for elevation in elevation_range:
        if shared_data['shutdown'].value: break
        
        # 1. Move servo to the new elevation ring
        smooth_servo_move(pi, elevation, shared_data)
        print(f"\n--- Scanning ring at Elevation: {elevation:.1f}° ---")

        # 2. Perform a full 360-degree pan in discrete steps
        steps_for_360_pan = int(360 / PAN_STEP_DEGREES)
        for i in range(steps_for_360_pan):
            if shared_data['shutdown'].value: break
            
            # 2a. Command the stepper motor to move one small step
            direction = 'right' if pan_direction > 0 else 'left'
            move(pi, direction, PAN_STEP_DEGREES, 0.0001, movement_queue, shared_data)

            # 2b. **Wait for the move to complete.** This is the crucial synchronization step.
            # We poll the shared angle until it's close to our target.
            # This ensures the motor has physically stopped moving.
            target_pan = (shared_data['stepper_degrees'].value + PAN_STEP_DEGREES * pan_direction) % 360
            while abs(((shared_data['stepper_degrees'].value - target_pan + 540) % 360) - 180) > 180 - PAN_STEP_DEGREES:
                 sleep(0.01) # Wait for stepper_worker to finish

            # 2c. **Pause to guarantee LiDAR reading.**
            # The lidar_handler process is constantly running. This short pause gives it
            # a clean window to get a reading while the motors are perfectly still.
            sleep(0.02) # 20ms pause for measurement
            print(f"\rScanning... Az: {shared_data['stepper_degrees'].value:.1f}°, El: {elevation:.1f}°", end="")

        print() # Newline after each completed ring
        pan_direction *= -1 # Reverse direction for the next ring
        
    print("\n--- PRECISE STEP-SCAN FINISHED ---")
    smooth_servo_move(pi, CENTER_TILT_ANGLE, shared_data)
    return True


# --- Main Control Logic ---

def initialize_gpio():
    GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT); GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH); GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting..."); initialize_gpio() 
    pi = pigpio.pi()
    if not pi.connected: return

    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT); pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
    
    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data)); stepper_process.start()
    
    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        while not shared_data['shutdown'].value:
            # --- This now calls the PRECISE scan function ---
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting precise background scan.")
                # Pass the movement_queue so the scan can command the stepper_worker
                concentric_ring_search_precise(pi, movement_queue, shared_data)
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True
            
            # (The rest of the manual control logic is unchanged)
            if shared_data['tilt_up'].value: move(pi, 'up', 5.0, None, movement_queue, shared_data); shared_data['tilt_up'].value = False
            if shared_data['tilt_down'].value: move(pi, 'down', 5.0, None, movement_queue, shared_data); shared_data['tilt_down'].value = False
            if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data); shared_data['pan_left'].value = False
            if shared_data['pan_right'].value: move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data); shared_data['pan_right'].value = False
            if shared_data["go_to_target"].value:
                track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001, movement_queue, shared_data)
                shared_data["go_to_target"].value = False
            
            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None); stepper_process.join()
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected: pi.set_servo_pulsewidth(13, 0); pi.stop()
        GPIO.cleanup()
