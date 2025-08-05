from multiprocessing import Process, Manager, Queue
from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- Your Provided Worker and Move Functions (Unchanged for servo) ---

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

def set_servo_angle_absolute(target_angle, shared_data):
    """Uses your existing move function to set an absolute servo angle."""
    smooth_servo_move(target_angle, shared_data)


# --- NEW Conductor Functions and Search Algorithm ---

# --- Search Parameters ---
CENTER_TILT_ANGLE = 90.0
MAX_TILT_RADIUS = 45.0
TILT_STEP_DEGREES = 1.0
PAN_DEGREES_PER_SECOND = 45.0  # Base speed for panning
STEPPER_PULSE_PIN = 2 # The GPIO pin for the stepper pulses
STEPPER_DIR_PIN = 3   # The GPIO pin for the stepper direction
MICROSTEPS_PER_REVOLUTION = 3200 # 1.8 degree motor with 1/16 microstepping

def read_lidar():
    """Placeholder for reading the TF-MINI S sensor."""
    return (999, 0) # (distance, strength)

def degrees_to_hz(degrees_per_second):
    """Converts rotational speed in degrees/sec to frequency in Hz for the stepper."""
    return (degrees_per_second / 360) * MICROSTEPS_PER_REVOLUTION

def concentric_ring_search_smooth(shared_data):
    """The main search algorithm, modified for smooth sweeping motion."""
    print("\n--- STARTING SMOOTH CONCENTRIC RING SEARCH ---")
    pan_direction_is_cw = True  # True for CW, False for CCW

    for radius in range(int(TILT_STEP_DEGREES), int(MAX_TILT_RADIUS) + 1, int(TILT_STEP_DEGREES)):
        target_tilt = CENTER_TILT_ANGLE - radius
        set_servo_angle_absolute(target_tilt, shared_data)
        print(f"\n--- Scanning new ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")

        # Calculate the sweep range and duration for this ring
        tilt_rad_for_scaling = math.radians(90 - target_tilt) # Angle from the pole
        scaling_factor = abs(math.cos(tilt_rad_for_scaling))
        pan_range_degrees = 360.0
        
        # Adjust pan speed based on the tilt angle to maintain consistent surface speed
        # Slower pan speed when tilted further away from the center
        current_pan_speed_dps = PAN_DEGREES_PER_SECOND * scaling_factor
        if current_pan_speed_dps < 1.0: # Prevent extremely slow speeds
            current_pan_speed_dps = 1.0

        sweep_duration_seconds = pan_range_degrees / current_pan_speed_dps
        scan_frequency_hz = degrees_to_hz(current_pan_speed_dps)

        # Set the panning direction
        if pan_direction_is_cw:
            GPIO.output(STEPPER_DIR_PIN, GPIO.LOW) # Adjust if your motor is backwards
        else:
            GPIO.output(STEPPER_DIR_PIN, GPIO.HIGH)

        print(f"Sweep Duration: {sweep_duration_seconds:.2f}s, Pan Speed: {current_pan_speed_dps:.1f}°/s")
        
        # Start hardware PWM for smooth motion
        pi.hardware_PWM(STEPPER_PULSE_PIN, int(scan_frequency_hz), 500000) # 50% duty cycle

        scan_start_time = monotonic()
        while (monotonic() - scan_start_time) < sweep_duration_seconds:
            distance, strength = read_lidar()
            # We no longer have a precise real-time degree, but this is inherent to sweep-scanning
            print(f"\rSearching... Tilt: {shared_data['servo_degrees'].value:5.1f}° (Sweeping)", end="")

            if strength > 200 and distance < 50:
                pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0) # Stop the motor
                print(f"\n\nTARGET ACQUIRED!")
                return True
            
            sleep(0.01) # Poll the sensor at 100Hz

        # Stop the motor at the end of the sweep
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        
        # Reverse direction for the next ring
        pan_direction_is_cw = not pan_direction_is_cw
        print()

    print("\n\nSEARCH FAILED: Target not found.")
    return False

def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.OUT) # Enable pin
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(19, GPIO.OUT)
    GPIO.setup(6, GPIO.OUT)
    GPIO.output(6, GPIO.HIGH)
    GPIO.output(4, GPIO.LOW) # Enable driver

def run_motor_control(shared_data):
    print("[MotorControl] Starting...")

    shared_data['servo_degrees'].value = 90.0
    shared_data['scan_trigger'].value = False

    initialize_gpio()
    global pi
    pi = pigpio.pi()

    # Set initial position
    set_servo_angle_absolute(90, shared_data)
    sleep(1) # Wait for servo to settle

    try:
        print("[MotorControl] Idle, waiting for scan trigger...")
        while not shared_data.get('shutdown', False):
            if shared_data['scan_trigger'].value:
                print("[MotorControl] Trigger received: starting scan")
                concentric_ring_search_smooth(shared_data)
                shared_data['scan_trigger'].value = False
            sleep(0.1)
    finally:
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0) # Ensure motor is off
        GPIO.output(4, GPIO.HIGH) # Disable driver
        pi.set_servo_pulsewidth(13, 0) # Disable servo
        pi.stop()
        GPIO.cleanup()  
        print("[MotorControl] Shut down cleanly")

# --- Main Execution Block ---
if __name__ == '__main__':
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['servo_degrees'] = 90.0

        # For demonstration, we'll trigger the scan immediately.
        shared_data['scan_trigger'] = True 
        
        # The motor control logic now runs in the main process for simplicity
        run_motor_control(shared_data)
