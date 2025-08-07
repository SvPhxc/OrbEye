# File: motors/motor_controller.py

from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Stepper Worker and Movement Functions (Unchanged) ---
def stepper_worker(pi, movement_queue, shared_data):
    """
    Handles individual, non-blocking stepper movements using PIGPIO waves.
    This process runs independently to handle queued movement commands.
    """
    print("[WORKER] Stepper worker started (using PIGPIO waves)")
    
    STEPPER_PULSE_PIN = 19
    STEPPER_DIR_PIN = 3

    while not shared_data['shutdown'].value:
        try:
            command = movement_queue.get(timeout=0.1)
            if command is None: break
        except Exception: # Catches Queue.Empty
            continue

        direction, degrees_to_move, delay = command

        ideal_microsteps = degrees_to_move / 0.05625
        total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
        
        if actual_microsteps_to_take == 0: continue

        pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1)
        
        us_delay = int(delay * 1_000_000)
        if us_delay < 10: us_delay = 10
        
        pi.wave_clear()
        
        pulse = [ pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay) ]
        
        pi.wave_add_generic(pulse)
        wave_id = pi.wave_create()
        
        repeats_lsb = actual_microsteps_to_take % 256
        repeats_msb = actual_microsteps_to_take // 256
        chain = [255, 0, wave_id, 255, 1, repeats_lsb, repeats_msb]
        
        pi.wave_chain(chain)
        
        while pi.wave_tx_busy():
            sleep(0.01)
        
        pi.wave_delete(wave_id)

        current_pos = shared_data['stepper_degrees'].value
        actual_degrees_this_move = actual_microsteps_to_take * 0.05625
        shared_data['stepper_degrees'].value = (current_pos - actual_degrees_this_move) % 360 if direction == 'left' else (current_pos + actual_degrees_this_move) % 360
            
    print("[WORKER] Stepper worker shutting down.")


def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step_size if target_degrees > current_degrees else -step_size)
    
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09)
        pi.set_servo_pulsewidth(13, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)
    # Final precise move
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
    tilt_limit, hysteresis_margin = 90, 3

    current_pan = shared_data["stepper_degrees"].value
    current_tilt = shared_data["servo_degrees"].value
    flipped = shared_data["flipped"].value

    if target_elevation > (tilt_limit + hysteresis_margin) and not flipped:
        shared_data["flipped"].value = True
    elif target_elevation < (tilt_limit - hysteresis_margin) and flipped:
        shared_data["flipped"].value = False

    flipped = shared_data["flipped"].value
    
    adjusted_azimuth = (target_azimuth + 180) % 360 if flipped else target_azimuth
    adjusted_elevation = 180 - target_elevation if flipped else target_elevation

    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1:
        pan_direction = "right" if delta_pan > 0 else "left"
        move(pi, pan_direction, abs(delta_pan), delay, movement_queue, shared_data)

    if abs(adjusted_elevation - current_tilt) > 1:
        smooth_servo_move(pi, adjusted_elevation, shared_data)


# --- Smooth Concentric Search (MODIFIED) ---

SEARCH_FREQUENCY_HZ = 100.0
CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 90.0 # Will scan from 90 down to 0 degrees elevation
TILT_STEP_DEGREES = 1.5
LIDAR_READ_INTERVAL_DEGREES = 1.5
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

def concentric_ring_search_smooth(pi, shared_data):
    """Performs the fast background scan using hardware PWM."""
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction = 1
    
    MICROSTEPS_PER_DEGREE = 1 / MICROSTEP_ANGLE
    
    # Scan from the horizon (0 deg servo) up to the center (90 deg servo)
    # The servo moves from 90 (center) down to 0 (horizon)
    for radius in range(0, int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        if shared_data['shutdown'].value: break
        
        target_tilt = CENTER_TILT_ANGLE - radius
        smooth_servo_move(pi, target_tilt, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        
        total_steps_for_360_pan = int(360 * MICROSTEPS_PER_DEGREE)
        
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        
        # Use hardware PWM for fast, smooth scanning
        scan_frequency_hz = 4000
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000) # 50% duty cycle

        start_time = monotonic()
        # Let it run for the time it takes to complete one 360-degree rotation
        duration = 360.0 / (scan_frequency_hz * MICROSTEP_ANGLE)
        
        while monotonic() - start_time < duration:
            if shared_data['shutdown'].value: break
            # Update the shared angle based on elapsed time
            elapsed_time = monotonic() - start_time
            pan_change = elapsed_time * scan_frequency_hz * MICROSTEP_ANGLE * pan_direction
            new_pan = (shared_data['stepper_degrees'].value + pan_change) % 360
            # We don't write to shared_data here to avoid race conditions, LiDAR process reads it
            sleep(0.01) # LiDAR process will be sampling during this time

        # ***** THE FIX IS HERE *****
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0) # Stop the PWM signal
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT) # **RELEASE PIN FROM HARDWARE PWM**
        
        # Update the final angle after the spin
        shared_data['stepper_degrees'].value = (shared_data['stepper_degrees'].value + 360 * pan_direction) % 360
        
        if shared_data['shutdown'].value: break
        pan_direction *= -1 # Reverse direction for the next ring
        
    print("\n--- SMOOTH CONCENTRIC RING SEARCH FINISHED ---")
    # Return servo to center position
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
    
    # Start the dedicated stepper worker process
    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data))
    stepper_process.start()

    # Set initial servo position
    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        print("[MotorControl] Idle, waiting for triggers from GUI...")
        while not shared_data['shutdown'].value:
            # Check for background scan trigger
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting smooth scan")
                concentric_ring_search_smooth(pi, shared_data)
                # Reset scan trigger and set save trigger for lidar_handler
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True
            
            # Check for manual movement flags
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
            
            # Check for go_to_target trigger
            if shared_data["go_to_target"].value:
                track_target(pi, shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001, movement_queue, shared_data)
                shared_data["go_to_target"].value = False
            
            sleep(0.05)
    except Exception as e:
        print(f"[MotorControl] An error occurred: {e}")
    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None) # Signal worker to stop
        stepper_process.join()
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.set_servo_pulsewidth(13, 0)
            pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shutdown complete.")
