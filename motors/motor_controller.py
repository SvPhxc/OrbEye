from multiprocessing import Process, Manager, Queue
from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math


# --- Stepper Worker and Movement Functions ---

def stepper_worker(movement_queue, shared_data):
    """Handles individual, non-blocking stepper movements."""
    print("[WORKER] Stepper worker started")
    while True:
        command = movement_queue.get()
        if command is None:
            break

        direction, degrees_to_move, delay = command

        ideal_microsteps = degrees_to_move / 0.05625
        # Correctly access shared data without .value
        total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error']
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        # Correctly update shared data without .value
        shared_data['cumulative_error'] = total_microsteps_to_consider - actual_microsteps_to_take
        
        # Correctly access shared data without .value
        current_pos = shared_data['stepper_degrees']
        actual_degrees_this_move = actual_microsteps_to_take * 0.05625

        if direction == 'left':
            GPIO.output(3, GPIO.LOW)
        else:
            GPIO.output(3, GPIO.HIGH)

        for _ in range(actual_microsteps_to_take):
            GPIO.output(2, GPIO.HIGH)
            sleep(delay)
            GPIO.output(2, GPIO.LOW)
            sleep(delay)

        # Correctly update shared data without .value
        if direction == 'left':
            shared_data['stepper_degrees'] = (current_pos - actual_degrees_this_move) % 360
        else:
            shared_data['stepper_degrees'] = (current_pos + actual_degrees_this_move) % 360

def smooth_servo_move(target_degrees, shared_data, step_delay=0.01, step_size=1):
    """Moves the servo smoothly to a target angle."""
    # Correctly access shared data without .value
    current_degrees = shared_data['servo_degrees']
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)) + (1 if target_degrees > current_degrees else -1), step_size if target_degrees > current_degrees else -step_size)
    
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09)
        pi.set_servo_pulsewidth(13, pulse_width)
        # Correctly update shared data without .value
        shared_data['servo_degrees'] = degrees
        sleep(step_delay)

def move(direction, degrees, delay, movement_queue, shared_data):
    """Unified function to command motors for discrete movements."""
    if direction in ['left', 'right']:
        command = (direction, degrees, delay)
        movement_queue.put(command)
    elif direction in ['up', 'down']:
        # Correctly access shared data without .value
        target_degrees = shared_data['servo_degrees']
        target_degrees += degrees if direction == 'up' else -degrees
        target_degrees = max(0, min(180, target_degrees))
        smooth_servo_move(target_degrees, shared_data)

def track_target(target_azimuth, target_elevation, delay, movement_queue, shared_data):
    """Moves the pan-tilt system to a specific azimuth and elevation."""
    tilt_limit, hysteresis_margin = 90, 3

    # Correctly access shared data without .value
    current_pan = shared_data["stepper_degrees"]
    current_tilt = shared_data["servo_degrees"]
    flipped = shared_data["flipped"] == 1

    if target_elevation > (tilt_limit + hysteresis_margin) and not flipped:
        flipped = True
        shared_data["flipped"] = 1
    elif target_elevation < (tilt_limit - hysteresis_margin) and flipped:
        flipped = False
        shared_data["flipped"] = 0

    adjusted_azimuth = (target_azimuth + 180) % 360 if flipped else target_azimuth
    adjusted_elevation = 180 - target_elevation if flipped else target_elevation

    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 1:
        pan_direction = "right" if delta_pan > 0 else "left"
        movement_queue.put((pan_direction, abs(delta_pan), delay))

    delta_tilt = adjusted_elevation - current_tilt
    if abs(delta_tilt) > 1:
        new_tilt = max(0, min(180, current_tilt + delta_tilt))
        smooth_servo_move(new_tilt, shared_data)

# --- Smooth Concentric Search & Control Logic ---

SEARCH_FREQUENCY_HZ = 100.0
CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 45.0
TILT_STEP_DEGREES = 1.5
LIDAR_READ_INTERVAL_DEGREES = 1.5 # How often to take a reading in degrees
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

def read_lidar():
    """Placeholder for reading the TF-MINI S sensor."""
    return (999, 0) # (distance, strength)

def set_servo_angle_absolute(target_angle, shared_data):
    """Sets an absolute servo angle."""
    smooth_servo_move(target_angle, shared_data)

def concentric_ring_search_smooth(pi, shared_data):
    """A smooth, continuous concentric search with precise LiDAR readings."""
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction = 1

    MICROSTEPS_PER_DEGREE = 1 / MICROSTEP_ANGLE
    steps_per_lidar_read = int(LIDAR_READ_INTERVAL_DEGREES * MICROSTEPS_PER_DEGREE)

    for radius in range(int(TILT_STEP_DEGREES), int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        target_tilt = CENTER_TILT_ANGLE - radius
        set_servo_angle_absolute(target_tilt, shared_data)
        # Correctly access shared data without .value
        print(f"\n--- Scanning new ring at Tilt: {shared_data['servo_degrees']:.1f}° ---")
        
        total_steps_for_360_pan = int(360 * MICROSTEPS_PER_DEGREE)
        
        pi.write(STEPPER_DIR_PIN, GPIO.HIGH if pan_direction > 0 else GPIO.LOW)

        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        pi.wave_clear()
        
        scan_frequency_hz = 2000 # Pulse frequency in Hz. Higher is faster. Adjust for your motor.
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000) # 50% duty cycle

        for step_count in range(0, total_steps_for_360_pan, steps_per_lidar_read):
            distance, strength = read_lidar()
            
            # Update position based on steps moved
            current_pan = shared_data['stepper_degrees']
            pan_change = LIDAR_READ_INTERVAL_DEGREES * pan_direction
            # Correctly update shared data without .value
            new_pan = (current_pan + pan_change) % 360
            shared_data['stepper_degrees'] = new_pan

            # Correctly access shared data without .value
            print(f"\rSearching... Pan: {new_pan:5.1f}°, Tilt: {shared_data['servo_degrees']:5.1f}°", end="")

            if strength > 200 and distance < 50:
                print(f"\n\nTARGET ACQUIRED!")
                pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                return True
                
            sleep_duration = steps_per_lidar_read / scan_frequency_hz
            sleep(sleep_duration)
            
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        pan_direction *= -1
        print()

    print("\n\nSEARCH FAILED: Target not found.")
    return False

def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_DIR_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_PULSE_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH)
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...")
    initialize_gpio()
    global pi
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio daemon not running. Exiting.")
        return

    # Initialize shared data correctly, without .value
    shared_data['stepper_degrees'] = 0.0
    shared_data['cumulative_error'] = 0.0
    shared_data['servo_degrees'] = 90.0
    shared_data['flipped'] = 0
    shared_data['scan_trigger'] = False
    shared_data['tilt_up'] = False
    shared_data['tilt_down'] = False
    shared_data['pan_left'] = False
    shared_data['pan_right'] = False
    shared_data['go_to_target'] = False
    
    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.start()

    set_servo_angle_absolute(90, shared_data)

    try:
        print("[MotorControl] Idle, waiting for triggers...")
        while not shared_data['shutdown']:
            # Check flags correctly, without .value
            if shared_data['scan_trigger']:
                print("[MotorControl] Trigger received: starting smooth scan")
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'] = False # Reset trigger
            
            if shared_data['tilt_up']:
                move('up', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_up'] = False

            if shared_data['tilt_down']:
                move('down', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_down'] = False

            if shared_data['pan_left']:
                move('left', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_left'] = False

            if shared_data['pan_right']:
                move('right', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_right'] = False

            if shared_data["go_to_target"]:
                track_target(shared_data["target_azimuth"], shared_data["target_elevation"], 0.0001, movement_queue, shared_data)
                shared_data["go_to_target"] = False
            
            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None)
        stepper_process.join()
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.set_servo_pulsewidth(13, 0)
            pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shutdown complete.")
