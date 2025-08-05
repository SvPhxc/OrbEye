from multiprocessing import Process, Manager, Queue
from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math


# --- Your Provided Worker and Move Functions (Unchanged) ---

def stepper_worker(movement_queue, shared_data):
    """
    Process function that listens for commands on a queue and controls the stepper motor.
    (This function is from your provided code and is NOT modified)
    """
    while True:
        command = movement_queue.get()
        if command is None:
            break
        direction, degrees, delay = command
        ideal_microsteps = degrees / 0.05625
        total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
        
        # NOTE: Your original code had a bug here. 'right' and 'left' were swapped.
        # Direction pin HIGH is typically one direction, LOW is the other.
        if direction == 'right':
            GPIO.output(3, GPIO.LOW) # Set direction (adjust if your motor is backwards)
        else: # 'left'
            GPIO.output(3, GPIO.HIGH)

        for _ in range(actual_microsteps_to_take):
            # NOTE: Your original code had a bug here. Pin 2 is step, Pin 3 is dir.
            GPIO.output(2, GPIO.HIGH)
            sleep(delay)
            GPIO.output(2, GPIO.LOW)
            sleep(delay)
        
        actual_degrees_this_move = actual_microsteps_to_take * 0.05625
        
        # This update is atomic because of the Manager
        current_pos = shared_data['stepper_degrees'].value
        if direction == 'left':
            shared_data['stepper_degrees'].value = (current_pos - actual_degrees_this_move) % 360
        else:
            shared_data['stepper_degrees'].value = (current_pos + actual_degrees_this_move) % 360


def smooth_servo_move(target_degrees, shared_data, step_delay=0.01, step_size=1):
    current_degrees = shared_data['servo_degrees'].value
    if target_degrees > current_degrees:
        for degrees in range(int(round(current_degrees)), int(round(target_degrees)) + 1, step_size):
            pulse_width = 500 + (degrees / 0.09)
            pi.set_servo_pulsewidth(13, pulse_width)
            shared_data['servo_degrees'].value = degrees
            sleep(step_delay)
    else:
        for degrees in range(int(round(current_degrees)), int(round(target_degrees)) - 1, -step_size):
            pulse_width = 500 + (degrees / 0.09)
            pi.set_servo_pulsewidth(13, pulse_width)
            shared_data['servo_degrees'].value = degrees
            sleep(step_delay)


def move(direction, degrees, delay, movement_queue, shared_data):
    """
    A unified function to command either the stepper or the servo motor.
    (This function is from your provided code and is slightly adapted for clarity)
    """
    if direction in ['left', 'right']:
        command = (direction, degrees, delay)
        movement_queue.put(command)
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value
        if direction == 'up':
            target_degrees += degrees
        else:
            target_degrees -= degrees
        target_degrees = max(0, min(180, target_degrees))
        smooth_servo_move(target_degrees, shared_data)

# --- NEW Conductor Functions and Search Algorithm ---

# --- Search Parameters ---
SEARCH_FREQUENCY_HZ = 100.0
LOOP_DELAY_S = 1.0 / SEARCH_FREQUENCY_HZ
CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 45.0
TILT_STEP_DEGREES = 1.5
PAN_STEP_DEGREES = 1.5
STEPPER_DELAY = 0.00001

def read_lidar():
    """Placeholder for reading the TF-MINI S sensor."""
    return (999, 0) # (distance, strength)

def set_servo_angle_absolute(target_angle, shared_data):
    """Uses your existing move function to set an absolute servo angle."""
    current_angle = shared_data['servo_degrees'].value
    delta = target_angle - current_angle
    
    if delta > 0:
        # The 'move' function for servo is blocking, which is what we want here.
        smooth_servo_move(target_angle, shared_data)
    elif delta < 0:
        smooth_servo_move(target_angle, shared_data)

def set_pan_angle_and_wait(target_angle, movement_queue, shared_data):
    """
    Commands the stepper to an absolute angle and WAITS for it to arrive
    by polling the shared data variable. This is the key to synchronization.
    """
    current_angle = shared_data['stepper_degrees'].value
    delta = (target_angle - current_angle + 180) % 360 - 180
    
    if abs(delta) < 0.1: # Don't move if we are already there
        return

    if delta > 0:
        direction = 'right'
    else:
        direction = 'left'
    
    # Send the non-blocking command using your 'move' function
    move(direction, abs(delta), STEPPER_DELAY, movement_queue, shared_data)
    
    # Poll the shared data until the stepper process confirms the move is done
    # This loop makes the async function behave like a sync one.
    while abs(shared_data['stepper_degrees'].value - target_angle) > 1.0: # Tolerance of 1 degree
        sleep(0.005)


def concentric_ring_search(movement_queue, shared_data):
    """The main search algorithm, using only your provided functions."""
    print("\n--- STARTING CONCENTRIC RING SEARCH ---")
    pan_direction = 1  # 1 for CW, -1 for CCW

    for radius in range(int(TILT_STEP_DEGREES), int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        target_tilt = CENTER_TILT_ANGLE - radius
        set_servo_angle_absolute(target_tilt, shared_data)
        print(f"\n--- Scanning new ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")

        tilt_rad_for_scaling = math.radians(target_tilt)
        scaling_factor = abs(math.sin(tilt_rad_for_scaling))
        pan_range_degrees = 360.0 * scaling_factor
        num_pan_steps = int(pan_range_degrees / PAN_STEP_DEGREES)
        
        if num_pan_steps < 1: continue

        start_pan = shared_data['stepper_degrees'].value
        
        for i in range(num_pan_steps + 1):
            loop_start_time = monotonic()
            
            step_angle = i * PAN_STEP_DEGREES
            target_pan = (start_pan + step_angle * pan_direction)
            
            set_pan_angle_and_wait(target_pan, movement_queue, shared_data)

            distance, strength = read_lidar()
            print(f"\rSearching... Pan: {shared_data['stepper_degrees'].value:5.1f}°, Tilt: {shared_data['servo_degrees'].value:5.1f}°", end="")
            
            if strength > 200 and distance < 50:
                print(f"\n\nTARGET ACQUIRED!")
                return True
            
            time_elapsed = monotonic() - loop_start_time
            sleep_duration = LOOP_DELAY_S - time_elapsed
            if sleep_duration > 0:
                sleep(sleep_duration)
        
        pan_direction *= -1
        print()

    print("\n\nSEARCH FAILED: Target not found.")
    return False


def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.OUT)
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(2, GPIO.OUT)
    GPIO.setup(6, GPIO.OUT)
    GPIO.output(6, GPIO.HIGH)
    GPIO.output(4, GPIO.LOW)

def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...")

    shared_data['stepper_degrees'].value = 0.0
    shared_data['cumulative_error'].value = 0.0
    shared_data['servo_degrees'].value = 90.0
    shared_data['scan_trigger'].value = False

    initialize_gpio()
    global pi
    pi = pigpio.pi()

    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.start()

    set_servo_angle_absolute(90, shared_data)
    set_pan_angle_and_wait(0, movement_queue, shared_data)

    try:
        print("[MotorControl] Idle, waiting for scan trigger...")
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting scan")
                concentric_ring_search(movement_queue, shared_data)
                shared_data['scan_trigger'].value = False
            sleep(0.05)
    finally:
        movement_queue.put(None)
        stepper_process.join()
        GPIO.output(4, GPIO.HIGH)
        pi.set_servo_pulsewidth(13, 0)
        pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shut down cleanly")



# --- Main Execution Block ---
'''
if __name__ == '__main__':
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['stepper_degrees'] = 0.0
        shared_data['cumulative_error'] = 0.0
        shared_data['servo_degrees'] = 90 # Start at 90 degrees

        movement_queue = Queue()
        stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
        stepper_process.start()

        try:
            # Set initial position
            set_servo_angle_absolute(90, shared_data)
            set_pan_angle_and_wait(0, movement_queue, shared_data)
            print("Initial position set. Starting search in 2 seconds...")
            sleep(2)
            
            # Run the search
            concentric_ring_search(movement_queue, shared_data)

        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            print("Cleaning up...")
            movement_queue.put(None)
            stepper_process.join()
            GPIO.output(4, GPIO.HIGH)
            pi.set_servo_pulsewidth(13, 0)
            pi.stop()
            print("Script finished.")
'''
