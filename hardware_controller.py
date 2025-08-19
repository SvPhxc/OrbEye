#!/usr/bin/env python3
"""
Hardware Controller for Stepper Motor (Pan) and Servo (Tilt) Control
Uses multiprocessing with shared data and PID control for smooth movement
PWM-based stepper control for smooth operation with precise position tracking
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
    # Stepper Parameters - DRV8825 optimized
    STEPPER_MAX_SPEED = 5000  # max steps per second (DRV8825 can handle up to 250kHz)
    STEPPER_MIN_SPEED = 100   # min steps per second
    STEPPER_ACCEL_DISTANCE = 2.0  # degrees to start/stop acceleration (reduced for faster scanning)
    STEPPER_CRUISE_SPEED = 3000  # optimal cruise speed for DRV8825

    # Fast transition parameters
    MAX_FREQ_CHANGE_RATE = 2000  # Hz per millisecond (much faster transitions)

    # PID Parameters for stepper speed control (tuned for higher speeds)
    KP = 6.0   # Increased proportional gain for faster response
    KI = 0.05  # Increased integral gain
    KD = 0.01  # Reduced derivative to prevent oscillation at high speeds

    # Servo Parameters
    SERVO_MIN_PULSE = 500+(23*0.09)  # microseconds
    SERVO_MAX_PULSE = 1750  # microseconds
    SERVO_MIN_ANGLE = 0  # degrees
    SERVO_MAX_ANGLE = 90  # degrees
    SERVO_DISPLACEMENT = 0.0  # degrees offset for 0 point (pointing straight forward)


# Scan Parameters - Optimized for faster DRV8825 operation
SCAN_AZIMUTH_STEP = 1.0  # Smaller steps for higher resolution (was 2.0)
SCAN_ELEVATION_STEP = 2.0  # Smaller steps for higher resolution (was 5.0)
SCAN_TILT_MAX = 90.0  # start elevation
SCAN_TILT_MIN = 0.0  # end elevation
LIDAR_SAMPLE_RATE = 1000  # Hz - LiDAR sampling rate
LIDAR_SAMPLE_TIME = 1.0 / LIDAR_SAMPLE_RATE  # Time per sample
SCAN_SAMPLES_PER_POSITION = 3  # More samples per position (was 2)


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

        print(f"[HWCtrl] Scan step: Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}°")

        # Set target position and trigger movement
        self.shared_data["target_azimuth"].value = self.current_scan_az
        self.shared_data["target_elevation"].value = self.current_scan_el
        self.shared_data["go_to_target"].value = True  # Trigger movement
        self.shared_data["target_reached"].value = False

        # Wait for movement to complete with timeout
        move_timeout = time.time() + 5.0  # 5 second timeout
        movement_started = False

        while not self.shared_data["target_reached"].value:
            if time.time() > move_timeout:
                print(f"[HWCtrl] Movement timeout at Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}°")
                break

            # Check if movement has started
            if not movement_started and not self.shared_data["go_to_target"].value:
                movement_started = True
                print(f"[HWCtrl] Movement started to Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}°")

            time.sleep(0.01)

        print(f"[HWCtrl] Reached position Az:{self.shared_data['stepper_degrees'].value:.1f}° El:{self.shared_data['servo_degrees'].value:.1f}°")

        # Allow shorter settling time due to faster, more precise movement
        time.sleep(0.05)  # Reduced from 0.1s since DRV8825 moves more precisely

        # Collect samples at this position
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
            print(f"[HWCtrl] Collected {len(samples)} samples at Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}° - Avg dist: {avg_sample[2]:.1f}cm")

            if len(self.background_data_buffer) % 10 == 0:  # Progress every 10 points
                print(f"[HWCtrl] Scan progress: {len(self.background_data_buffer)} points total")
        else:
            print(f"[HWCtrl] No valid samples at Az:{self.current_scan_az:.1f}° El:{self.current_scan_el:.1f}°")

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


class PWMStepperController:
    """PWM-based stepper motor controller with precise position tracking"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.pid = PIDController(MotorParams.KP, MotorParams.KI, MotorParams.KD)
        self.running = True

        # Position tracking
        self.step_count = 0
        self.current_frequency = 0
        self.target_frequency = 0
        self.last_update_time = time.time()

        # PWM callback for step counting
        self.step_callback = None

        # Setup GPIO pins
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        # Enable stepper driver
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Active low
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake up driver

        # Initialize PWM with DRV8825 optimized settings
        # DRV8825 minimum pulse width is 1.9μs, so we can use higher frequencies
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)  # 50% duty cycle, start at 0 Hz

        print("[HWCtrl] PWM Stepper controller initialized for DRV8825")

    def _step_counter_callback(self, gpio, level, tick):
        """Callback to count steps for position tracking"""
        if level == 1:  # Rising edge
            direction = self.pi.read(STEPPER_DIR_PIN)
            if direction:
                self.step_count += 1
            else:
                self.step_count -= 1

            # Update shared position data
            degrees = self.step_count * MICROSTEP_ANGLE
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value = degrees

    def calculate_target_frequency(self, distance_to_target):
        """Calculate target PWM frequency based on distance and PID"""

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
        target_speed = max(MotorParams.STEPPER_MIN_SPEED,
                           min(MotorParams.STEPPER_MAX_SPEED, target_speed))

        return target_speed

    def smooth_frequency_transition(self, target_freq):
        """Smoothly transition PWM frequency optimized for DRV8825"""
        current_time = time.time()
        dt = current_time - self.last_update_time

        if dt > 0:
            # DRV8825 can handle much faster transitions
            max_change = MotorParams.MAX_FREQ_CHANGE_RATE * dt * 1000  # Convert to seconds
            freq_diff = target_freq - self.current_frequency

            if abs(freq_diff) > max_change:
                # Limit the frequency change rate
                if freq_diff > 0:
                    self.current_frequency += max_change
                else:
                    self.current_frequency -= max_change
            else:
                self.current_frequency = target_freq

        self.last_update_time = current_time
        return self.current_frequency

    def move_to_target(self):
        """Move stepper to target azimuth using PWM control"""

        # Set up step counting callback
        if not self.step_callback:
            self.step_callback = self.pi.callback(STEPPER_PULSE_PIN, pigpio.RISING_EDGE,
                                                  self._step_counter_callback)

        target_pos = self.shared_data["target_azimuth"].value
        current_pos = self.shared_data["stepper_degrees"].value

        print(f"[HWCtrl] Moving from {current_pos:.3f}° to {target_pos:.3f}°")

        while abs(target_pos - current_pos) >= MICROSTEP_ANGLE and self.running:
            error = target_pos - current_pos

            # Determine direction
            direction = 1 if error > 0 else 0
            self.pi.write(STEPPER_DIR_PIN, direction)

            # Calculate target frequency
            target_freq = self.calculate_target_frequency(error)

            # Apply smooth frequency transition
            smooth_freq = self.smooth_frequency_transition(target_freq)

            # Update PWM frequency
            if smooth_freq > 0:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(smooth_freq), 500000)  # 50% duty cycle
            else:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)

            # Update current position from step counter
            current_pos = self.shared_data["stepper_degrees"].value

            # Faster update rate for DRV8825 - it can handle rapid changes
            time.sleep(0.0005)  # 0.5ms for 2kHz update rate

        # Stop PWM when target is reached
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)

        # Final position update
        final_pos = self.shared_data["stepper_degrees"].value
        print(f"[HWCtrl] Target reached. Final position: {final_pos:.3f}°, Error: {abs(target_pos - final_pos):.3f}°")

        # Target reached
        self.shared_data["target_reached"].value = True

    def stop(self):
        """Stop the PWM stepper controller"""
        self.running = False

        # Stop PWM
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)

        # Clean up callback
        if self.step_callback:
            self.step_callback.cancel()

        # Disable stepper
        self.pi.write(STEPPER_ENABLE_PIN, 1)  # Disable stepper
        print("[HWCtrl] PWM Stepper controller stopped")


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

            time.sleep(0.001)  # 50Hz update rate

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
    servo_thread = None

    try:
        pi = pigpio.pi()
        if not pi.connected:
            print("[HWCtrl] Failed to connect to pigpio daemon")
            return

        # Initialize controllers
        stepper = PWMStepperController(pi, shared_data)  # Use PWM stepper controller
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = BackgroundScanner(shared_data)

        print("[HWCtrl] Combined controller initialized with PWM stepper")

        # Start servo control thread
        servo_thread = threading.Thread(target=servo.control_servo)
        servo_thread.daemon = True
        servo_thread.start()

        # Main control loop
        while not shared_data["shutdown"].value:
            # Update LiDAR data continuously
            lidar.get_lidar_data()

            # Handle background scanning - give it priority
            if shared_data["background_scan_active"].value:
                scanner.execute_background_scan(lidar)
                # Small delay between scan steps to prevent overwhelming the system
                time.sleep(0.05)

            # Handle normal stepper movement only when not scanning
            elif shared_data["go_to_target"].value:
                # Reset target reached flag
                shared_data["target_reached"].value = False

                # Move stepper to target
                stepper.move_to_target()

                # Clear go_to_target flag when done
                shared_data["go_to_target"].value = False

            else:
                # Idle - just update LiDAR data
                time.sleep(0.001)  # 1ms main loop for responsive control

    except Exception as e:
        print(f"[HWCtrl] Combined controller error: {e}")
    finally:
        print("[HWCtrl] Shutting down...")
        if stepper:
            stepper.stop()
        if servo:
            servo.stop()
        if servo_thread:
            servo_thread.join(timeout=1.0)  # Ensure thread is joined
        if lidar:
            lidar.stop()
        if pi and pi.connected:
            pi.stop()
        print("[HWCtrl] Shutdown complete.")


def run_hardware_controller(shared_data):
    """Main function to start the combined hardware controller"""

    print("[HWCtrl] Starting PWM-based hardware controller process...")
    try:
        combined_controller_process(shared_data)
    except KeyboardInterrupt:
        print("[HWCtrl] Process interrupted by user.")
    print("[HWCtrl] Hardware controller process stopped.")