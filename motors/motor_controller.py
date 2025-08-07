# File: motors/motor_controller.py
# This script is designed to be imported as a module by a main application.

from multiprocessing import Process, Queue
from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Stepper Worker and Movement Functions ---

# ***** THIS IS THE KEY MODIFIED FUNCTION *****
def stepper_worker(pi, movement_queue, shared_data):
    """
    Handles individual, non-blocking stepper movements using PIGPIO waves.
    This resolves the conflict with the hardware PWM used by the smooth search.
    """
    print("[WORKER] Stepper worker started (using PIGPIO waves)")
    
    STEPPER_PULSE_PIN = 19
    STEPPER_DIR_PIN = 3

    while True:
        if shared_data['shutdown'].value:
            break
        try:
            command = movement_queue.get(timeout=0.1)
            if command is None:
                break
        except Exception: # Catches Queue.Empty
            continue

        direction, degrees_to_move, delay = command

        ideal_microsteps = degrees_to_move / 0.05625
        total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
        
        if actual_microsteps_to_take == 0:
            continue

        if direction == 'left':
            pi.write(STEPPER_DIR_PIN, 0) # GPIO.LOW
        else:
            pi.write(STEPPER_DIR_PIN, 1) # GPIO.HIGH
        
        us_delay = int(delay * 1_000_000)
        if us_delay < 10: us_delay = 10 # Prevent too high frequency
        
        pi.wave_clear()
        
        pulse = [
            pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay),
            pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)
        ]
        
        pi.wave_add_generic(pulse)
        wave_id = pi.wave_create()
        
        # ***** THE FIX IS HERE *****
        # We must split the total steps into a Most Significant Byte (MSB)
        # and a Least Significant Byte (LSB) for the wave chain command.
        
        repeats_lsb = actual_microsteps_to_take % 256
        repeats_msb = actual_microsteps_to_take // 256
        
        # The chain format is: [start, wave, repeat, lsb, msb]
        chain = [255, 0, wave_id, 255, 1, repeats_lsb, repeats_msb]
        
        pi.wave_chain(chain)
        
        while pi.wave_tx_busy():
            sleep(0.01)
        
        pi.wave_delete(wave_id)

        current_pos = shared_data['stepper_degrees'].value
        actual_degrees_this_move = actual_microsteps_to_take * 0.05625
        if direction == 'left':
            shared_data['stepper_degrees'].value = (current_pos - actual_degrees_this_move) % 360
        else:
            shared_data['stepper_degrees'].value = (current_pos + actual_degrees_this_move) % 360
            
    print("[WORKER] Stepper worker shutting down.")


def smooth_servo_move(target_degrees, shared_data, step_delay=0.01, step_size=1):
    """Moves the servo smoothly to a target angle."""
    current_degrees = shared_data['servo_degrees'].value
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)) + (1 if target_degrees > current_degrees else -1), step_size if target_degrees > current_degrees else -step_size)
    
    for degrees in degrees_range:
        pulse_width = 500 + (degrees / 0.09)
        pi.set_servo_pulsewidth(13, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

def move(direction, degrees, delay, movement_queue, shared_data):
    """Unified function for manual, discrete movements."""
    if direction in ['left', 'right']:
        command = (direction, degrees, delay)
        movement_queue.put(command)
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value
        target_degrees += degrees if direction == 'up' else -degrees
        target_degrees = max(0, min(180, target_degrees))
        smooth_servo_move(target_degrees, shared_data)

def track_target(target_azimuth, target_elevation, delay, movement_queue, shared_data):
    """Moves the pan-tilt system to a specific azimuth and elevation."""
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
    if abs(delta_pan) > 0.1: # Use a small tolerance
        pan_direction = "right" if delta_pan > 0 else "left"
        # The move function will now correctly handle large degree values
        move(pan_direction, abs(delta_pan), delay, movement_queue, shared_data)

    delta_tilt = adjusted_elevation - current_tilt
    if abs(delta_tilt) > 1:
        new_tilt = max(0, min(180, current_tilt + delta_tilt))
        smooth_servo_move(new_tilt, shared_data)

# --- Smooth Concentric Search & Control Logic (Unchanged) ---

SEARCH_FREQUENCY_HZ = 100.0
CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 45.0
TILT_STEP_DEGREES = 1.5
LIDAR_READ_INTERVAL_DEGREES = 1.5
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625

def read_lidar(shared_data):
    distance = shared_data['lidar_data'][0]
    strength = shared_data['lidar_data'][1]
    return (distance, strength)

def set_servo_angle_absolute(target_angle, shared_data):
    smooth_servo_move(target_angle, shared_data)

def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction = 1

    MICROSTEPS_PER_DEGREE = 1 / MICROSTEP_ANGLE
    steps_per_lidar_read = int(LIDAR_READ_INTERVAL_DEGREES * MICROSTEPS_PER_DEGREE)

    for radius in range(int(TILT_STEP_DEGREES), int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        if shared_data['shutdown'].value: break
        
        target_tilt = CENTER_TILT_ANGLE - radius
        set_servo_angle_absolute(target_tilt, shared_data)
        print(f"\n--- Scanning new ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        
        total_steps_for_360_pan = int(360 * MICROSTEPS_PER_DEGREE)
        
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        
        scan_frequency_hz = 4000
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000)

        for _ in range(0, total_steps_for_360_pan, steps_per_lidar_read):
            if shared_data['shutdown'].value: break

            distance, strength = read_lidar(shared_data)
            
            pan_change = LIDAR_READ_INTERVAL_DEGREES * pan_direction
            new_pan = (shared_data['stepper_degrees'].value + pan_change) % 360
            shared_data['stepper_degrees'].value = new_pan

            print(f"\rSearching... Pan: {new_pan:5.1f}°, Tilt: {shared_data['servo_degrees'].value:5.1f}° | LiDAR Dist: {distance:.2f}", end="")
                
            sleep_duration = steps_per_lidar_read / scan_frequency_hz
            sleep(sleep_duration)
            
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        if shared_data['shutdown'].value: break
        pan_direction *= -1
    return False

def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT)
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH)
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

# This is the main function that your main.py will run as a process
def run_motor_control(shared_data, movement_queue):
    """Main loop for the motor control process."""
    print("[MotorControl] Starting...")
    initialize_gpio() 
    
    global pi
    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio daemon not running. Exiting.")
        return

    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)

    # Pass the 'pi' object to the worker
    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data))
    stepper_process.start()

    set_servo_angle_absolute(shared_data['servo_degrees'].value, shared_data)

    try:
        print("[MotorControl] Idle, waiting for triggers from GUI...")
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting smooth scan")
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True
            
            if shared_data['tilt_up'].value:
                move('up', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_up'].value = False

            if shared_data['tilt_down'].value:
                move('down', 5.0, None, movement_queue, shared_data)
                shared_data['tilt_down'].value = False

            if shared_data['pan_left'].value:
                move('left', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_left'].value = False

            if shared_data['pan_right'].value:
                move('right', 5.0, 0.0001, movement_queue, shared_data)
                shared_data['pan_right'].value = False

            if shared_data["go_to_target"].value:
                track_target(shared_data["target_azimuth"].value, shared_data["target_elevation"].value, 0.0001, movement_queue, shared_data)
                shared_data["go_to_target"].value = False
            
            sleep(0.05)
    except Exception as e:
        print(f"[MotorControl] An error occurred: {e}")
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
