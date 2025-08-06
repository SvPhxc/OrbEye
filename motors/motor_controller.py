# File: motors/motor_controller.py
# This script is designed to be imported as a module by a main application.
# It has been corrected to use only the 'pigpio' library for stepper control.

from multiprocessing import Process, Queue
from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Pin Definitions & Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

# --- Pigpio-Based Stepper Worker ---

def stepper_worker(pi, movement_queue, shared_data):
    """
    Handles individual, non-blocking stepper movements using pigpio waves.
    This function now requires the 'pi' object from the pigpio library.
    """
    print("[WORKER] Stepper worker started (using pigpio)")
    
    # Ensure this process has a valid connection to the pigpio daemon
    if not pi.connected:
        print("[WORKER] Error: pigpio not connected. Worker cannot run.")
        return

    while True:
        if shared_data['shutdown'].value:
            break
        try:
            command = movement_queue.get(timeout=0.1)
            if command is None:
                break
        except Exception: # Catches Queue.Empty
            continue

        direction, degrees_to_move, delay_s = command

        # Set direction using RPi.GPIO (as it's just a simple HIGH/LOW)
        if direction == 'left':
            GPIO.output(STEPPER_DIR_PIN, GPIO.LOW)
        else:
            GPIO.output(STEPPER_DIR_PIN, GPIO.HIGH)

        # --- Generate Steps with Pigpio Waveform ---
        ideal_microsteps = degrees_to_move / MICROSTEP_ANGLE
        total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take

        if actual_microsteps_to_take == 0:
            continue

        # Delay between pulses in microseconds.
        # A smaller delay means a faster motor speed.
        us_delay = int(delay_s * 1_000_000)
        
        pi.wave_clear()
        
        # Create a single pulse: ON for us_delay, then OFF for us_delay
        pulse = [
            pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay),
            pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)
        ]
        
        pi.wave_add_generic(pulse)
        one_pulse_wave = pi.wave_create()

        # Chain the single pulse wave the required number of times
        # 255 is the code for a loop start, 255 0 is a loop forever, 255 1 x y is loop x y times
        # We send one wave, `actual_microsteps_to_take` times.
        chain = [255, 0, one_pulse_wave, 255, 1, actual_microsteps_to_take, 0]
        pi.wave_chain(chain)

        # Wait until the wave is finished
        while pi.wave_tx_busy():
            sleep(0.01)

        # Update position after the move is complete
        current_pos = shared_data['stepper_degrees'].value
        actual_degrees_this_move = actual_microsteps_to_take * MICROSTEP_ANGLE
        if direction == 'left':
            shared_data['stepper_degrees'].value = (current_pos - actual_degrees_this_move) % 360
        else:
            shared_data['stepper_degrees'].value = (current_pos + actual_degrees_this_move) % 360
            
        print(f"[WORKER] Moved {direction} by {actual_degrees_this_move:.2f} degrees.")

    print("[WORKER] Stepper worker shutting down.")

# --- Movement and Search Functions (Largely Unchanged) ---

def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    # Simplified range logic
    step = step_size if target_degrees > current_degrees else -step_size
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)) + step, step)
    
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09)
        pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']:
        command = (direction, degrees, delay)
        movement_queue.put(command)
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value
        target_degrees += degrees if direction == 'up' else -degrees
        target_degrees = max(0, min(180, target_degrees))
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
    if abs(delta_pan) > 1:
        pan_direction = "right" if delta_pan > 0 else "left"
        move(pi, pan_direction, abs(delta_pan), delay, movement_queue, shared_data)

    delta_tilt = adjusted_elevation - current_tilt
    if abs(delta_tilt) > 1:
        new_tilt = max(0, min(180, current_tilt + delta_tilt))
        smooth_servo_move(pi, new_tilt, shared_data)

def read_lidar(shared_data):
    distance = shared_data['lidar_data'][0]
    strength = shared_data['lidar_data'][1]
    return (distance, strength)

def set_servo_angle_absolute(pi, target_angle, shared_data):
    smooth_servo_move(pi, target_angle, shared_data)

def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction = 1
    MICROSTEPS_PER_DEGREE = 1 / MICROSTEP_ANGLE
    steps_per_lidar_read = int(LIDAR_READ_INTERVAL_DEGREES * MICROSTEPS_PER_DEGREE)

    for radius in range(int(TILT_STEP_DEGREES), int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        if shared_data['shutdown'].value: break
        
        target_tilt = 90.0 - radius
        set_servo_angle_absolute(pi, target_tilt, shared_data)
        print(f"\n--- Scanning new ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        
        total_steps_for_360_pan = int(360 * MICROSTEPS_PER_DEGREE)
        
        # Set pan direction with RPi.GPIO
        GPIO.output(STEPPER_DIR_PIN, GPIO.HIGH if pan_direction > 0 else GPIO.LOW)

        # Use pigpio for smooth motion
        scan_frequency_hz = 4000  # Higher frequency = faster pan
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000) # 50% duty cycle

        num_readings = total_steps_for_360_pan // steps_per_lidar_read
        for _ in range(num_readings):
            if shared_data['shutdown'].value: break
            distance, strength = read_lidar(shared_data)
            
            pan_change = LIDAR_READ_INTERVAL_DEGREES * pan_direction
            new_pan = (shared_data['stepper_degrees'].value + pan_change) % 360
            shared_data['stepper_degrees'].value = new_pan

            print(f"\rSearching... Pan: {new_pan:5.1f}°, Tilt: {shared_data['servo_degrees'].value:5.1f}° | LiDAR: {distance:.2f}", end="")

            if strength > 200 and distance < 50:
                print(f"\n\nTARGET ACQUIRED!")
                pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                return True
            
            sleep_duration = steps_per_lidar_read / scan_frequency_hz
            sleep(sleep_duration)
            
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        if shared_data['shutdown'].value: break
        pan_direction *= -1
        print()

    print("\n\nSEARCH FAILED: Target not found.")
    return False

def initialize_gpio():
    """Initializes GPIO pins using RPi.GPIO for simple HIGH/LOW settings."""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    # NOTE: STEPPER_PULSE_PIN is NOT configured here. pigpio will manage it.
    GPIO.setup([STEPPER_ENABLE_PIN, STEPPER_DIR_PIN, STEPPER_SLEEP_PIN], GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH)
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

# --- Main Process Function ---

def run_motor_control(shared_data, movement_queue):
    """Main loop for the motor control process."""
    print("[MotorControl] Starting...")
    # Initialize non-pulsing GPIOs first
    initialize_gpio()
    
    # Connect to pigpio daemon
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio daemon not running. Exiting.")
        return

    # Start the worker process, passing it the pi connection object
    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data))
    stepper_process.start()

    set_servo_angle_absolute(pi, shared_data['servo_degrees'].value, shared_data)

    try:
        print("[MotorControl] Idle, waiting for triggers...")
        while not shared_data['shutdown'].value:
            # The 'pi' object is now passed to all functions that need it
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting smooth scan")
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'].value = False
            
            if shared_data['tilt_up'].value:
                move(pi, 'up', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_up'].value = False

            if shared_data['tilt_down'].value:
                move(pi, 'down', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_down'].value = False

            if shared_data['pan_left'].value:
                move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data) # Delay is now for pulse width
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
            pi.set_servo_pulsewidth(SERVO_PIN, 0)
            pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shutdown complete.")
