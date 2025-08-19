#!/usr/bin/env python3
"""
Hardware Controller for Stepper Motor (Pan) and Servo (Tilt) Control
Uses multiprocessing with shared data and PID control for smooth movement
"""

import time
import math
import pigpio
import serial
import queue
import numpy as np
from multiprocessing import Process, Manager, Value, Array
import threading

# Hardware Configuration
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625  # degrees per step


# Control Parameters
class MotorParams:
    # Stepper Parameters
    STEPPER_MAX_SPEED = 1000  # max steps per second
    STEPPER_MIN_SPEED = 50  # min steps per second
    STEPPER_ACCEL_DISTANCE = 10.0  # degrees to start/stop acceleration

    # PID Parameters for stepper speed control
    KP = 0.8  # Proportional gain
    KI = 0.1  # Integral gain
    KD = 0.2  # Derivative gain

    # Servo Parameters
    SERVO_MIN_PULSE = 500  # microseconds
    SERVO_MAX_PULSE = 2500  # microseconds
    SERVO_MIN_ANGLE = 0  # degrees
    SERVO_MAX_ANGLE = 180  # degrees
    SERVO_DISPLACEMENT = 90.0  # degrees offset for 0 point (pointing straight forward)


# Scan Parameters
SCAN_AZIMUTH_STEP = 2.0  # degrees per step for background scan
SCAN_ELEVATION_STEP = 5.0  # degrees per step for background scan
SCAN_TILT_MAX = 90.0  # start elevation
SCAN_TILT_MIN = 0.0  # end elevation
LIDAR_SAMPLE_RATE = 1000  # Hz - LiDAR sampling rate
LIDAR_SAMPLE_TIME = 1.0 / LIDAR_SAMPLE_RATE  # Time per sample
SCAN_SAMPLES_PER_POSITION = 5  # number of samples to average per position


class PIDController:
    """Simple PID controller for stepper speed regulation"""

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def calculate(self, error):
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            return 0

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt

        # Calculate output
        output = proportional + integral + derivative

        # Update for next iteration
        self.prev_error = error
        self.last_time = current_time

        return output


class LidarController:
    """Controls LiDAR sensor and data collection"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.ser = None
        self.lidar_queue = queue.Queue(maxsize=100)
        self.shutdown_event = threading.Event()
        self.lidar_thread = None

        # Initialize serial connection
        try:
            # Decode byte string for serial port
            port = self.shared_data["lidar_port"].value.decode()
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            print(f"[HWCtrl-LIDAR] LiDAR initialized on port {port}")

            # Start reader thread
            self.lidar_thread = threading.Thread(target=self._lidar_reader_thread)
            self.lidar_thread.daemon = True
            self.lidar_thread.start()

        except Exception as e:
            print(f"[HWCtrl-LIDAR] Failed to initialize LiDAR: {e}")

    def _lidar_reader_thread(self):
        """Background thread to read LiDAR data"""
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    try:
                        self.lidar_queue.put_nowait(
                            (frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8), time.time()))
                    except queue.Full:
                        pass
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set():
                    print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def get_lidar_data(self):
        """Get latest LiDAR data and update shared data"""
        try:
            dist, strength, ts = self.lidar_queue.get_nowait()
            with self.shared_data["lidar_data"].get_lock():
                self.shared_data["lidar_data"][:] = [dist, strength, ts]
            return dist, strength, ts
        except queue.Empty:
            return None, None, None

    def stop(self):
        """Stop LiDAR controller"""
        self.shutdown_event.set()
        if self.lidar_thread and self.lidar_thread.is_alive():
            self.lidar_thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[HWCtrl-LIDAR] LiDAR controller stopped")


class BackgroundScanner:
    """Handles background scanning functionality"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.current_scan_az = 0.0
        self.current_scan_el = SCAN_TILT_MAX
        self.background_data_buffer = []
        self.scan_direction = 1  # 1 for forward, -1 for backward

    def execute_background_scan(self, lidar_controller):
        """Execute one step of the background scan at LiDAR sampling rate"""
        if not self.shared_data["background_scan_active"].value:
            return

        # Move to current scan position
        self.shared_data["target_azimuth"].value = self.current_scan_az
        self.shared_data["target_elevation"].value = self.current_scan_el

        # Wait for movement to complete with timeout
        move_timeout = time.time() + 5.0  # 5 second timeout
        while (abs(self.shared_data["stepper_degrees"].value - self.current_scan_az) > MICROSTEP_ANGLE * 2 or
               abs(self.shared_data["servo_degrees"].value - self.current_scan_el) > 1.0):
            if time.time() > move_timeout:
                print(f"[HWCtrl] Movement timeout at Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}°")
                break
            time.sleep(0.01)

        # Allow settling time
        time.sleep(0.05)

        # Collect samples at this position at LiDAR rate (1000Hz)
        samples = []
        sample_start_time = time.time()

        for i in range(SCAN_SAMPLES_PER_POSITION):
            sample_time = sample_start_time + (i * LIDAR_SAMPLE_TIME)

            # Wait for precise timing
            while time.time() < sample_time:
                time.sleep(0.0001)  # 0.1ms precision

            dist, strength, ts = lidar_controller.get_lidar_data()
            if dist is not None and dist > 0:  # Valid reading
                samples.append([self.current_scan_az, self.current_scan_el, dist, strength])

        # Add averaged sample to buffer if we got valid data
        if samples:
            avg_sample = np.mean(samples, axis=0)
            self.background_data_buffer.append(avg_sample)
            if len(self.background_data_buffer) % 50 == 0:  # Progress every 50 points
                print(f"[HWCtrl] Scan progress: {len(self.background_data_buffer)} points - "
                      f"Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}° - {len(samples)} valid samples")

        # Update scan position
        self._update_scan_position()

    def _update_scan_position(self):
        """Update scan position for next measurement"""
        # Move azimuth in current direction
        self.current_scan_az += SCAN_AZIMUTH_STEP * self.scan_direction

        # Check if we've completed a full ring
        ring_complete = False
        if self.scan_direction == 1 and self.current_scan_az >= 360.0:
            # Completed forward sweep
            self.current_scan_az = 360.0
            ring_complete = True
        elif self.scan_direction == -1 and self.current_scan_az <= 0.0:
            # Completed backward sweep
            self.current_scan_az = 0.0
            ring_complete = True

        # If ring is complete, move to next elevation and reverse direction
        if ring_complete:
            self.current_scan_el -= SCAN_ELEVATION_STEP
            self.scan_direction *= -1  # Reverse direction for next ring

            print(f"[HWCtrl] Completed elevation ring at {self.current_scan_el + SCAN_ELEVATION_STEP:.1f}°, "
                  f"next ring direction: {'forward' if self.scan_direction == 1 else 'backward'}")

        # Check if entire scan is complete
        if self.current_scan_el < SCAN_TILT_MIN:
            print("[HWCtrl] BACKGROUND_SCAN finished.")
            if self.background_data_buffer:
                # Expected rows: [azimuth, elevation, distance_cm, strength]
                # Decode byte string for file path
                path = self.shared_data["background_path"].value.decode()
                np.save(path, np.array(self.background_data_buffer))
                print(f"[HWCtrl] Saved {len(self.background_data_buffer)} scan points to {path}")
                self.background_data_buffer = []
            self.shared_data["background_scan_active"].value = False

            # Reset scan parameters for next scan
            self.current_scan_az = 0.0
            self.current_scan_el = SCAN_TILT_MAX
            self.scan_direction = 1


# --- FIX: Added missing class definition ---
class StepperController:
    """Controls stepper motor with PID speed control"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.pid = PIDController(MotorParams.KP, MotorParams.KI, MotorParams.KD)
        self.running = True

        # Setup GPIO pins
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        # Enable stepper driver
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Active low
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake up driver

        print("[HWCtrl] Stepper controller initialized")

    def calculate_target_speed(self, distance_to_target):
        """Calculate target speed based on distance using PID and acceleration profile"""

        # Acceleration/deceleration profile
        if abs(distance_to_target) <= MotorParams.STEPPER_ACCEL_DISTANCE:
            # Close to target - decelerate
            speed_factor = abs(distance_to_target) / MotorParams.STEPPER_ACCEL_DISTANCE
            base_speed = MotorParams.STEPPER_MIN_SPEED + (
                    (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * speed_factor
            )
        else:
            # Far from target - full speed
            base_speed = MotorParams.STEPPER_MAX_SPEED

        # Apply PID correction
        pid_output = self.pid.calculate(distance_to_target)
        target_speed = base_speed + abs(pid_output) * 100  # Scale PID output

        # Clamp speed to limits
        return max(MotorParams.STEPPER_MIN_SPEED,
                   min(MotorParams.STEPPER_MAX_SPEED, target_speed))

    def move_to_target(self):
        """Move stepper to target azimuth with PID speed control"""

        target_pos = self.shared_data["target_azimuth"].value
        current_pos = self.shared_data["stepper_degrees"].value

        while abs(target_pos - current_pos) >= MICROSTEP_ANGLE and self.running:
            error = target_pos - current_pos

            # Determine direction
            direction = 1 if error > 0 else 0
            self.pi.write(STEPPER_DIR_PIN, direction)

            # Calculate speed based on distance to target
            target_speed = self.calculate_target_speed(error)
            step_delay = 1.0 / (2 * target_speed)  # Half period for pulse

            # Send pulse
            self.pi.write(STEPPER_PULSE_PIN, 1)
            time.sleep(step_delay)
            self.pi.write(STEPPER_PULSE_PIN, 0)
            time.sleep(step_delay)

            # Update position
            step_change = MICROSTEP_ANGLE if direction else -MICROSTEP_ANGLE
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value += step_change
                current_pos = self.shared_data["stepper_degrees"].value

        # Target reached
        self.shared_data["target_reached"].value = True

    def stop(self):
        """Stop the stepper controller"""
        self.running = False
        self.pi.write(STEPPER_ENABLE_PIN, 1)  # Disable stepper
        print("[HWCtrl] Stepper controller stopped")


class ServoController:
    """Controls servo motor for tilt movement"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True

        # Initialize servo
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        print("[HWCtrl] Servo controller initialized")

    def angle_to_pulse_width(self, angle):
        """Convert angle to servo pulse width with displacement correction"""
        # Apply displacement correction (subtract to make 0 point straight forward)
        corrected_angle = angle + MotorParams.SERVO_DISPLACEMENT

        # Clamp angle to servo limits
        corrected_angle = max(MotorParams.SERVO_MIN_ANGLE,
                              min(MotorParams.SERVO_MAX_ANGLE, corrected_angle))

        # Convert to pulse width
        pulse_range = MotorParams.SERVO_MAX_PULSE - MotorParams.SERVO_MIN_PULSE
        angle_range = MotorParams.SERVO_MAX_ANGLE - MotorParams.SERVO_MIN_ANGLE

        pulse_width = MotorParams.SERVO_MIN_PULSE + (
                (corrected_angle - MotorParams.SERVO_MIN_ANGLE) / angle_range * pulse_range
        )

        return int(pulse_width)

    def control_servo(self):
        """Control servo position based on target elevation"""

        while self.running:
            target_elevation = self.shared_data["target_elevation"].value
            current_servo = self.shared_data["servo_degrees"].value

            # Check if servo needs to move
            if abs(target_elevation - current_servo) > 0.5:  # 0.5 degree tolerance
                pulse_width = self.angle_to_pulse_width(target_elevation)
                self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

                # Update current position
                with self.shared_data["servo_degrees"].get_lock():
                    self.shared_data["servo_degrees"].value = target_elevation

            time.sleep(0.05)  # 20Hz update rate

    def stop(self):
        """Stop the servo controller"""
        self.running = False
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop servo signal
        print("[HWCtrl] Servo controller stopped")


def combined_controller_process(shared_data):
    """Combined process for controlling stepper, servo, LiDAR, and background scanning"""
    pi = None
    stepper = None
    servo = None
    lidar = None

    try:
        pi = pigpio.pi()
        if not pi.connected:
            print("[HWCtrl] Failed to connect to pigpio daemon")
            return

        # Initialize controllers
        stepper = StepperController(pi, shared_data)
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = BackgroundScanner(shared_data)

        print("[HWCtrl] Combined controller initialized")

        # Start servo control thread
        servo_thread = threading.Thread(target=servo.control_servo)
        servo_thread.daemon = True
        servo_thread.start()

        # Main control loop
        while not shared_data["shutdown"].value:
            # Update LiDAR data continuously
            lidar.get_lidar_data()

            # Handle background scanning
            if shared_data["background_scan_active"].value:
                scanner.execute_background_scan(lidar)
            else:
                # Normal stepper movement
                if shared_data["go_to_target"].value:
                    # Reset target reached flag
                    shared_data["target_reached"].value = False

                    # Move stepper to target
                    stepper.move_to_target()

                    # Clear go_to_target flag when done
                    shared_data["go_to_target"].value = False

            time.sleep(0.001)  # 1ms main loop for responsive control

    except Exception as e:
        print(f"[HWCtrl] Combined controller error: {e}")
    finally:
        print("[HWCtrl] Shutting down...")
        if stepper:
            stepper.stop()
        if servo:
            servo_thread.join(timeout=1.0)  # Ensure thread is joined
            servo.stop()
        if lidar:
            lidar.stop()
        if pi and pi.connected:
            pi.stop()
        print("[HWCtrl] Shutdown complete.")


# --- FIX: Renamed function to be called by main.py ---
def run_hardware_controller(shared_data):
    """Main function to start the combined hardware controller"""

    print("[HWCtrl] Starting hardware controller process...")
    try:
        combined_controller_process(shared_data)
    except KeyboardInterrupt:
        print("[HWCtrl] Process interrupted by user.")
    print("[HWCtrl] Hardware controller process stopped.")


if __name__ == "__main__":
    """Test the hardware controller independently"""
    import ctypes

    # Create shared data structure
    manager = Manager()
    shared_data = manager.dict({
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "target_azimuth": Value('d', 90.0),
        "target_elevation": Value('d', 45.0),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),
        # LiDAR parameters
        "lidar_port": Value(ctypes.c_char_p, b'/dev/ttyUSB0'),
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),  # [distance, strength, timestamp]
        # Background scan parameters
        "background_scan_active": Value('b', False),
        "background_path": Value(ctypes.c_char_p, b'background_scan.npy'),
        # System control
        "shutdown": Value('b', False),
    })

    # Start the controller process for testing
    controller_proc = Process(target=run_hardware_controller, args=(shared_data,))
    controller_proc.start()
    print("Hardware controller test process started.")

    try:
        # Test movement
        print("\n--- Test 1: Moving to 180°, 60° ---")
        shared_data["target_azimuth"].value = 180.0
        shared_data["target_elevation"].value = 60.0
        shared_data["go_to_target"].value = True

        # Wait for movement to complete
        while shared_data["go_to_target"].value:
            time.sleep(0.1)
        print("--- Test 1: Movement complete. ---\n")
        time.sleep(2)

        # Optional: Test background scan
        # print("--- Test 2: Starting background scan ---")
        # shared_data["background_scan_active"].value = True
        # while shared_data["background_scan_active"].value:
        #     time.sleep(1)
        # print("--- Test 2: Background scan complete. ---\n")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Requesting shutdown of test process...")
        shared_data["shutdown"].value = True
        controller_proc.join(timeout=5)
        if controller_proc.is_alive():
            controller_proc.terminate()
        print("Test finished.")