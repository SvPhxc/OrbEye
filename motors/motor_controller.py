# File: motors/motor_controller.py

from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Stepper Worker (MODIFIED TO BE STABLE) ---
def stepper_worker(pi, movement_queue, shared_data):
    """
    Handles individual, non-blocking stepper movements using a persistent PIGPIO wave.
    This prevents the "non-existent wave id" error.
    """
    print("[WORKER] Stepper worker started.")
    
    STEPPER_PULSE_PIN = 19
    STEPPER_DIR_PIN = 3
    pulse_wave_id = -1 # Initialize to an invalid ID

    try:
        # --- Create ONE persistent wave for a single step ---
        # This wave will be reused for all movements.
        us_delay = 500 # A reasonable default microsecond delay for manual moves
        pi.wave_add_generic([
            pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay),
            pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)
        ])
        pulse_wave_id = pi.wave_create()
        print(f"[WORKER] Created persistent wave with ID: {pulse_wave_id}")

        while not shared_data['shutdown'].value:
            try:
                # Wait for a command from the main motor control loop
                command = movement_queue.get(timeout=0.1)
                if command is None: break # Shutdown signal
            except Exception: # Catches Queue.Empty
                continue

            direction, degrees_to_move, _ = command # Delay is now handled by the persistent wave

            # Calculate the exact number of microsteps to take
            ideal_microsteps = degrees_to_move / 0.05625
            total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
            actual_microsteps_to_take = round(total_microsteps_to_consider)
            shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
            
            if actual_microsteps_to_take == 0: continue

            # Set the motor direction
            pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
            
            # Use the wave_chain to repeat the persistent pulse wave
            repeats_lsb = actual_microsteps_to_take % 256
            repeats_msb = actual_microsteps_to_take // 256
            chain = [255, 0, pulse_wave_id, 255, 1, repeats_lsb, repeats_msb]
            
            pi.wave_chain(chain)
            
            # Wait until the transmission is complete
            while pi.wave_tx_busy():
                sleep(0.01)

            # Update the shared angle value
            current_pos = shared_data['stepper_degrees'].value
            actual_degrees_this_move = actual_microsteps_to_take * 0.05625
            new_pos = (current_pos - actual_degrees_this_move) if direction == 'left' else (current_pos + actual_degrees_this_move)
            shared_data['stepper_degrees'].value = new_pos % 360
            
    finally:
        # --- Clean up the persistent wave on exit ---
        if pulse_wave_id != -1 and pi.connected:
            print(f"[WORKER] Deleting persistent wave ID: {pulse_wave_id}")
            pi.wave_delete(pulse_wave_id)
        print("[WORKER] Stepper worker shutting down.")


# --- Other Functions (Unchanged) ---

def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    # Corrected range to handle moving up and down
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
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)

def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value
    current_tilt = shared_data["servo_degrees"].value
    
    # Simple, direct tracking without flip logic for now
    adjusted_azimuth = target_azimuth % 360
    adjusted_elevation = max(0, min(180, target_elevation))

    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1:
        pan_direction = "right" if delta_pan > 0 else "left"
        move(pi, pan_direction, abs(delta_pan), delay, movement_queue, shared_data)

    if abs(adjusted_elevation - current_tilt) > 1:
        smooth_servo_move(pi, adjusted_elevation, shared_data)

# --- Smooth Concentric Search ---

CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 90.0
TILT_STEP_DEGREES = 1.5
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction = 1
    
    # Scan from the horizon up to the center
    for radius in range(int(MAX_TILT_RADIUS), -1, -int(TILT_STEP_DEGREES)):
        if shared_data['shutdown'].value: break
        
        target_tilt = CENTER_TILT_ANGLE - radius
        smooth_servo_move(pi, target_tilt, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        
        scan_frequency_hz = 4000
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000)

        duration = 360.0 / (scan_frequency_hz * MICROSTEP_ANGLE)
        start_time = monotonic()
        
        # While spinning, continuously update the shared angle based on time
        while monotonic() - start_time < duration:
            if shared_data['shutdown'].value: break
            elapsed_time = monotonic() - start_time
            # Calculate theoretical position, the lidar_handler will use this
            with shared_data['stepper_degrees'].get_lock():
                 # This part is for estimation; the lidar handler uses its own instantaneous read
                 pass
            sleep(0.01)

        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT) # Release pin from hardware PWM
        
        if shared_data['shutdown'].value: break
        pan_direction *= -1
        
    print("\n--- SMOOTH CONCENTRIC RING SEARCH FINISHED ---")
    smooth_servo_move(pi, CENTER_TILT_ANGLE, shared_data)
    return True

# --- Main Control Logic ---

def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH)
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...")
    initialize_gpio() 
    
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio daemon not running. Exiting.")
        return

    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
    
    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data))
    stepper_process.start()

    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        print("[MotorControl] Idle, waiting for triggers...")
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting smooth scan")
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True
            
            if shared_data['tilt_up'].value:
                move(pi, 'up', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_up'].value = False

            if shared_data['tilt_down'].value:
                move(pi, 'down', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_down'].value = False

            if shared_data['pan_left'].value:
                move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_left'].value = False

            if shared_data['pan_right'].value:
                move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_right'].value = False
            
            if shared_data["go_to_target"].value:
                track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001, movement_queue, shared_data)
                shared_data["go_to_target"].value = False
            
            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None)
        stepper_process.join()
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.set_servo_pulsewidth(13, 0)
            pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shutdown complete.")
