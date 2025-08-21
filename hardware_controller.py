#!/usr/bin/env python3
"""
Hardware Controller for Stepper Motor (Pan) and Servo (Tilt) Control
Uses multiprocessing with shared data and PID control for smooth movement
PWM-based stepper control for smooth operation with precise position tracking
FIXED: Continuous background scanning with data collection during movement
FIXED: Proper servo movement and angle limiting
FIXED: Background scan completion and movement after scan
FIXED: Overshoot issue for small movements
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
    # Stepper Parameters - DRV8825 optimized for speed
    STEPPER_MAX_SPEED = 8000  # Increased max (DRV8825 can handle up to 250kHz)
    STEPPER_MIN_SPEED = 100  # min steps per second
    STEPPER_ACCEL_DISTANCE = 1.5  # Reduced for faster acceleration
    STEPPER_CRUISE_SPEED = 6400  # Higher cruise speed for scans
    ACCEL_STEPS = 150  # e.g., use 800 steps to ramp up and 800 to ramp down

    # Ultra-fast transition for scanning
    MAX_FREQ_CHANGE_RATE = 3000  # Hz per millisecond (very fast transitions)

    # PID Parameters - tuned for high-speed operation
    KP = 0.2  # Higher proportional for faster response
    KI = 0.0  # Slightly higher integral
    KD = 0.0  # Lower derivative for stability at high speed

    # Servo Parameters
    SERVO_MIN_PULSE = 670  # microseconds (standard servo min)
    SERVO_MAX_PULSE = 1670  # microseconds (standard servo max)
    SERVO_MIN_ANGLE = 0  # degrees (physical limit)
    SERVO_MAX_ANGLE = 90  # degrees (physical limit)

    # Servo mounting offset - adjust this to set your zero point
    # If servo points 15 degrees up when at "0", set this to -15
    # This ensures displayed angles are always positive (0-90)
    SERVO_ZERO_OFFSET = 0  # degrees - adjust for your mounting

    # Pan angle limits
    PAN_MIN_ANGLE = 0.0  # degrees
    PAN_MAX_ANGLE = 360.0  # degrees


# Continuous Scan Parameters
# Speed Profiles for different scan modes
class ScanProfiles:
    # FAST SCAN - Lower resolution, high speed
    FAST = {
        "azimuth_speed": 360.0,  # 90 deg/sec (4 seconds per rotation)
        "elevation_step": 1.0,  # 5 degree steps (18 rings for 0-90)
        "data_rate": 1000,  # 30 Hz sampling
        "servo_move_time": 0.05,  # Faster servo movement
        "servo_settle_time": 0.05,  # Minimal settling
    }

    # NORMAL SCAN - Balanced speed and resolution
    NORMAL = {
        "azimuth_speed": 60.0,  # 60 deg/sec (6 seconds per rotation)
        "elevation_step": 3.0,  # 3 degree steps (30 rings)
        "data_rate": 40,  # 40 Hz sampling
        "servo_move_time": 0.8,  # Normal servo movement
        "servo_settle_time": 0.2,  # Short settling
    }

    # HIGH QUALITY - Higher resolution, slower speed
    HIGH_QUALITY = {
        "azimuth_speed": 30.0,  # 30 deg/sec (12 seconds per rotation)
        "elevation_step": 2.0,  # 2 degree steps (45 rings)
        "data_rate": 50,  # 50 Hz sampling
        "servo_move_time": 1.0,  # Safe servo movement
        "servo_settle_time": 0.3,  # Good settling
    }

    # ULTRA FAST - Maximum speed, minimum resolution
    ULTRA_FAST = {
        "azimuth_speed": 120.0,  # 120 deg/sec (3 seconds per rotation)
        "elevation_step": 10.0,  # 10 degree steps (9 rings only)
        "data_rate": 25,  # 25 Hz sampling
        "servo_move_time": 0.3,  # Very fast servo
        "servo_settle_time": 0.05,  # Almost no settling
    }


# Select scan profile here
CURRENT_SCAN_PROFILE = ScanProfiles.FAST  # Change this to select speed

# Apply selected profile
SCAN_AZIMUTH_SPEED = CURRENT_SCAN_PROFILE["azimuth_speed"]
SCAN_ELEVATION_STEP = CURRENT_SCAN_PROFILE["elevation_step"]
SCAN_DATA_RATE = CURRENT_SCAN_PROFILE["data_rate"]
SERVO_MOVE_TIME = CURRENT_SCAN_PROFILE["servo_move_time"]
SERVO_SETTLE_TIME = CURRENT_SCAN_PROFILE["servo_settle_time"]

# Fixed parameters
SCAN_AZIMUTH_STEP = 1  # Not used in continuous mode
SCAN_TILT_MAX = 90.0  # start elevation
SCAN_TILT_MIN = 0.0  # end elevation


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
            port = "/dev/serial0"
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            self.ser.write(bytearray([0x5A, 0x05, 0x11, 0x70]))
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
        # Drain the queue to get the most recent measurement.
        # This reduces latency between when a measurement is taken and
        # when it is correlated with the system's current position.
        last_data = None
        while not self.lidar_queue.empty():
            try:
                # Keep getting items until the queue is empty, storing the last one
                last_data = self.lidar_queue.get_nowait()
            except queue.Empty:
                # This can happen in a multithreaded environment, it's safe to ignore.
                break

        # If we successfully retrieved at least one data point, process the last one.
        if last_data:
            dist, strength, ts = last_data
            with self.shared_data["lidar_data"].get_lock():
                self.shared_data["lidar_data"][:] = [dist, strength, ts]
            return dist, strength, ts
        else:
            # The queue was empty, so no new data is available.
            return None, None, None

    def stop(self):
        """Stop LiDAR controller"""
        self.shutdown_event.set()
        if self.lidar_thread and self.lidar_thread.is_alive():
            self.lidar_thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[HWCtrl-LIDAR] LiDAR controller stopped")


class ContinuousBackgroundScanner:
    """Scanner using improved position tracking."""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.current_elevation = 90.0  # Start from top
        self.background_data_buffer = []
        self.scan_direction = 1
        self.scan_active = False
        self.scan_start_time = None

    def start_continuous_scan(self, lidar_controller, stepper_controller, servo_controller):
        """Start scanning with improved position tracking."""
        if not self.shared_data["background_scan_active"].value:
            return False

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        print(f"[HWCtrl] Scanning at elevation {self.current_elevation:.1f}°")

        # Move servo to elevation
        self.shared_data["target_elevation"].value = self.current_elevation

        # Wait for servo
        start_wait = time.time()
        while time.time() - start_wait < 0.5:
            current_servo = self.shared_data["servo_degrees"].value
            if abs(current_servo - self.current_elevation) < 1.0:
                break
            time.sleep(0.001)

        # Perform azimuth sweep
        if self.scan_direction == 1:
            start_az, end_az = 0.0, 360.0
        else:
            start_az, end_az = 360.0, 0.0

        # Use improved continuous movement
        stepper_controller.continuous_movement_scan(start_az, end_az, 90.0)  # 90°/sec

        # Collect data during movement
        self._collect_scan_data(lidar_controller, start_az, end_az)

        # Next elevation
        self.current_elevation -= 5.0  # 5° steps

        if self.current_elevation < 0:
            print("[HWCtrl] Scan complete")
            self._save_scan_data()
            stepper_controller.reset_hardware_state()
            self._reset_scan()
            return False

        self.scan_direction *= -1
        return True

    def _collect_scan_data(self, lidar_controller, start_az, end_az):
        """Collect data during scan movement."""
        collection_rate = 40  # Hz
        interval = 1.0 / collection_rate

        duration = abs(end_az - start_az) / 90.0  # 90°/sec scan speed
        samples = int(duration * collection_rate)

        for _ in range(samples):
            # Get current position from shared data (now accurate!)
            az = self.shared_data["stepper_degrees"].value
            el = self.shared_data["servo_degrees"].value

            # Get LiDAR data
            dist, strength, ts = lidar_controller.get_lidar_data()

            if dist and dist > 0:
                self.background_data_buffer.append([az, el, dist, strength])

            time.sleep(interval)

    def _save_scan_data(self):
        """Save scan data."""
        if self.background_data_buffer:
            path = self.shared_data["background_path"].value
            np.save(path, np.array(self.background_data_buffer))
            print(f"[HWCtrl] Saved {len(self.background_data_buffer)} points")

    def _reset_scan(self):
        """Reset for next scan."""
        self.current_elevation = 90.0
        self.scan_direction = 1
        self.background_data_buffer = []
        self.shared_data["background_scan_active"].value = False
        self.scan_start_time = None


class PWMStepperController:
    """
    Hybrid stepper controller using step counting for precision movements
    and time-based tracking for high-speed continuous movements.
    """

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.step_count = 0

        # Time-based tracking state
        self.time_tracking_active = False
        self.time_tracking_start = None
        self.time_tracking_start_pos = 0
        self.time_tracking_speed = 0  # steps per second
        self.time_tracking_direction = 1  # 1 or -1

        # Setup GPIO pins
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        # Enable stepper driver
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Active low
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake up
        time.sleep(0.001)

        # Step counting callback (only for low-speed movements)
        self.step_callback = None
        self._setup_step_counter()

        # Position update thread for time-based tracking
        self.position_updater = threading.Thread(target=self._position_updater_thread)
        self.position_updater.daemon = True
        self.position_updater.start()

        print("[HWCtrl] Hybrid stepper controller initialized")

    def _setup_step_counter(self):
        """Setup step counting callback for low-speed precision movements."""
        if self.step_callback:
            self.step_callback.cancel()
        self.step_callback = self.pi.callback(
            STEPPER_PULSE_PIN, pigpio.RISING_EDGE,
            self._step_counter_callback
        )

    def _step_counter_callback(self, gpio, level, tick):
        """Callback for counting steps (only used at low speeds)."""
        if level == 1 and not self.time_tracking_active:
            direction = self.pi.read(STEPPER_DIR_PIN)
            if direction:
                self.step_count += 1
            else:
                self.step_count -= 1

            # Wrap around at 360 degrees
            steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
            self.step_count = self.step_count % steps_per_rotation

    def _position_updater_thread(self):
        """Thread that updates position based on time during high-speed movements."""
        while self.running:
            if self.time_tracking_active:
                # Calculate current position based on elapsed time
                elapsed = time.time() - self.time_tracking_start
                steps_moved = int(self.time_tracking_speed * elapsed)

                # Update step count and shared position
                current_steps = self.time_tracking_start_pos + (steps_moved * self.time_tracking_direction)
                steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
                current_steps = current_steps % steps_per_rotation

                # Update both internal count and shared data
                self.step_count = current_steps
                degrees = current_steps * MICROSTEP_ANGLE

                with self.shared_data["stepper_degrees"].get_lock():
                    self.shared_data["stepper_degrees"].value = degrees

            time.sleep(0.001)  # 1ms update rate

    def start_time_tracking(self, speed_steps_per_sec, direction):
        """Start time-based position tracking for high-speed movements."""
        self.time_tracking_start_pos = self.step_count
        self.time_tracking_start = time.time()
        self.time_tracking_speed = speed_steps_per_sec
        self.time_tracking_direction = 1 if direction else -1
        self.time_tracking_active = True
        print(f"[HWCtrl] Started time-based tracking at {speed_steps_per_sec:.0f} steps/sec")

    def stop_time_tracking(self):
        """Stop time-based tracking and sync final position."""
        if self.time_tracking_active:
            # Calculate final position
            elapsed = time.time() - self.time_tracking_start
            steps_moved = int(self.time_tracking_speed * elapsed)
            final_steps = self.time_tracking_start_pos + (steps_moved * self.time_tracking_direction)

            steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
            self.step_count = final_steps % steps_per_rotation

            # Update shared data with final position
            degrees = self.step_count * MICROSTEP_ANGLE
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value = degrees

            self.time_tracking_active = False
            print(f"[HWCtrl] Stopped time-based tracking at {degrees:.1f}°")

    def continuous_movement_scan(self, start_az, end_az, speed_deg_per_sec):
        """
        Optimized continuous movement for scanning using time-based tracking.
        """
        # Stop any existing movements
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.wave_tx_stop()
        self.stop_time_tracking()

        # Calculate movement parameters
        total_distance = abs(end_az - start_az)
        steps_per_second = speed_deg_per_sec / MICROSTEP_ANGLE

        # Limit to max speed
        steps_per_second = min(steps_per_second, MotorParams.STEPPER_MAX_SPEED)

        # Set direction
        direction = 1 if end_az > start_az else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        print(f"[HWCtrl] Continuous scan: {start_az:.1f}° -> {end_az:.1f}° at {steps_per_second:.0f} steps/sec")

        # Start time-based tracking
        self.start_time_tracking(steps_per_second, direction)

        # Start hardware PWM
        frequency = int(steps_per_second)
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, frequency, 500000)  # 50% duty cycle

        # Calculate movement duration
        movement_time = total_distance / speed_deg_per_sec

        # Wait for movement to complete
        start_time = time.time()
        while (time.time() - start_time) < movement_time and self.running:
            if self.shared_data["shutdown"].value:
                break
            time.sleep(0.001)

        # Stop movement
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.stop_time_tracking()

        print(f"[HWCtrl] Scan movement complete. Final: {self.shared_data['stepper_degrees'].value:.1f}°")

    def move_to_angle(self, target_angle):
        """
        Move to target angle using appropriate method based on speed requirements.
        """
        # Stop any existing movements
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.wave_tx_stop()
        self.stop_time_tracking()

        # Calculate movement
        target_angle = max(MotorParams.PAN_MIN_ANGLE, min(MotorParams.PAN_MAX_ANGLE, target_angle))
        current_pos_steps = self.step_count
        target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
        error_steps = target_pos_steps - current_pos_steps

        # Use shortest path
        steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
        if abs(error_steps) > (steps_per_rotation / 2):
            if error_steps > 0:
                error_steps -= steps_per_rotation
            else:
                error_steps += steps_per_rotation

        total_steps = abs(error_steps)

        if total_steps < 1:
            print("[HWCtrl] Already at target position")
            self.shared_data["target_reached"].value = True
            return

        direction = 1 if error_steps > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        print(f"[HWCtrl] Moving to {target_angle:.1f}° ({total_steps} steps)")

        # For large movements, use time-based tracking
        if total_steps > 500:
            self._high_speed_move(total_steps, direction)
        else:
            self._precision_move(total_steps, direction)

        self.shared_data["target_reached"].value = True

    def _high_speed_move(self, total_steps, direction):
        """High-speed movement using time-based tracking."""
        # Simple trapezoidal profile
        accel_steps = min(MotorParams.ACCEL_STEPS, total_steps // 3)
        decel_steps = accel_steps
        cruise_steps = total_steps - (accel_steps + decel_steps)

        # Start time tracking
        self.start_time_tracking(MotorParams.STEPPER_CRUISE_SPEED, direction)

        # Acceleration phase
        for i in range(1, accel_steps + 1):
            if not self.running: break
            speed = MotorParams.STEPPER_MIN_SPEED + (
                        MotorParams.STEPPER_CRUISE_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / accel_steps)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
            time.sleep(1.0 / speed)

        # Cruise phase
        if cruise_steps > 0:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, MotorParams.STEPPER_CRUISE_SPEED, 500000)
            time.sleep(cruise_steps / MotorParams.STEPPER_CRUISE_SPEED)

        # Deceleration phase
        for i in range(decel_steps, 0, -1):
            if not self.running: break
            speed = MotorParams.STEPPER_MIN_SPEED + (
                        MotorParams.STEPPER_CRUISE_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / decel_steps)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
            time.sleep(1.0 / speed)

        # Stop
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.stop_time_tracking()

    def _precision_move(self, total_steps, direction):
        """Precision movement using waveforms and step counting."""
        # Build waveform
        pulses = []
        for _ in range(total_steps):
            delay_us = int(500000 / MotorParams.STEPPER_MIN_SPEED)
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        # Execute waveform
        self.pi.wave_clear()
        self.pi.wave_add_generic(pulses)
        wave_id = self.pi.wave_create()

        if wave_id >= 0:
            self.pi.wave_send_once(wave_id)
            while self.pi.wave_tx_busy():
                # Update position during movement
                degrees = self.step_count * MICROSTEP_ANGLE
                with self.shared_data["stepper_degrees"].get_lock():
                    self.shared_data["stepper_degrees"].value = degrees
                time.sleep(0.001)
            self.pi.wave_delete(wave_id)

        # Final position update
        degrees = self.step_count * MICROSTEP_ANGLE
        with self.shared_data["stepper_degrees"].get_lock():
            self.shared_data["stepper_degrees"].value = degrees

    def move_to_target(self):
        """Move to target azimuth from shared data."""
        target_pos = self.shared_data["target_azimuth"].value
        self.move_to_angle(target_pos)

    def reset_hardware_state(self):
        """Reset hardware state after scanning."""
        print("[HWCtrl] Resetting stepper hardware state...")

        # Stop all operations
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        self.stop_time_tracking()

        # Reset pin mode
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.write(STEPPER_PULSE_PIN, 0)

        # Re-setup step counter
        self._setup_step_counter()

        time.sleep(0.05)
        print("[HWCtrl] Hardware reset complete")

    def stop(self):
        """Stop the controller."""
        self.running = False
        self.stop_time_tracking()

        if self.step_callback:
            self.step_callback.cancel()

        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.write(STEPPER_ENABLE_PIN, 1)

        print("[HWCtrl] Stepper controller stopped")
class ServoController:
    """Controls servo motor for tilt movement"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True

        # Initialize servo
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

        # Set initial position to middle
        initial_angle = 45.0  # Start at 45 degrees
        self.move_to_angle(initial_angle)

        print(f"[HWCtrl] Servo controller initialized (zero offset: {MotorParams.SERVO_ZERO_OFFSET}°)")

    def angle_to_pulse_width(self, angle):
        """Convert angle to servo pulse width with mounting offset correction"""
        # Clamp input angle to valid range (0-90 degrees as seen by user)
        user_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, angle))

        # Apply mounting offset to get physical servo angle
        physical_angle = user_angle - MotorParams.SERVO_ZERO_OFFSET

        # Ensure physical angle is within servo's physical limits
        physical_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, physical_angle))

        # Convert to pulse width
        pulse_range = MotorParams.SERVO_MAX_PULSE - MotorParams.SERVO_MIN_PULSE
        angle_range = MotorParams.SERVO_MAX_ANGLE - MotorParams.SERVO_MIN_ANGLE

        pulse_width = MotorParams.SERVO_MIN_PULSE + (
            (physical_angle - MotorParams.SERVO_MIN_ANGLE) / angle_range * pulse_range
        )

        return int(pulse_width)

    def move_to_angle(self, target_angle):
        """Direct movement to target angle with angle limiting"""
        # Ensure angle is within valid range (what user sees)
        target_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, target_angle))

        print(f"[HWCtrl-Servo] Moving to {target_angle:.1f}° (user angle)")

        pulse_width = self.angle_to_pulse_width(target_angle)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

        # Update shared data with user-visible angle
        with self.shared_data["servo_degrees"].get_lock():
            self.shared_data["servo_degrees"].value = target_angle

        physical_angle = target_angle - MotorParams.SERVO_ZERO_OFFSET
        print(f"[HWCtrl-Servo] Physical angle: {physical_angle:.1f}°, Pulse: {pulse_width}µs")

    def control_servo(self):
        """Control servo position based on target elevation"""
        while self.running:
            target_elevation = self.shared_data["target_elevation"].value
            current_servo = self.shared_data["servo_degrees"].value

            # Check if servo needs to move (with hysteresis to prevent hunting)
            if abs(target_elevation - current_servo) > 0.5:  # 0.5 degree tolerance
                self.move_to_angle(target_elevation)

            time.sleep(0.001)  # 1000Hz update rate

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

        print(f"[HWCtrl] Initializing with servo zero offset: {MotorParams.SERVO_ZERO_OFFSET}°")
        print(f"[HWCtrl] Pan limits: {MotorParams.PAN_MIN_ANGLE}° - {MotorParams.PAN_MAX_ANGLE}°")
        print(f"[HWCtrl] Tilt limits: {MotorParams.SERVO_MIN_ANGLE}° - {MotorParams.SERVO_MAX_ANGLE}°")
        print(f"[HWCtrl] Scan speed profile: {SCAN_AZIMUTH_SPEED}°/s azimuth, {SCAN_ELEVATION_STEP}° elevation steps")

        # Initialize controllers
        stepper = PWMStepperController(pi, shared_data)
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = ContinuousBackgroundScanner(shared_data)

        print("[HWCtrl] Combined controller initialized with continuous scanning")

        # Start servo control thread
        servo_thread = threading.Thread(target=servo.control_servo)
        servo_thread.daemon = True
        servo_thread.start()

        # Main control loop
        while not shared_data["shutdown"].value:
            # Update LiDAR data continuously
            lidar.get_lidar_data()

            # Handle continuous background scanning
            if shared_data["background_scan_active"].value:
                scan_continues = scanner.start_continuous_scan(lidar, stepper, servo)
                if not scan_continues:
                    # Scan completed, ensure stepper is ready for normal operation
                    print("[HWCtrl] Background scan completed, ready for normal movement")
                    # Small delay before accepting new commands
                    time.sleep(0.1)

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
                time.sleep(0.001)  # 1ms main loop

    except Exception as e:
        import traceback
        print(f"[HWCtrl] Combined controller error: {e}")
        traceback.print_exc()
    finally:
        print("[HWCtrl] Shutting down...")
        if stepper:
            stepper.stop()
        if servo:
            servo.stop()
        if servo_thread:
            servo_thread.join(timeout=1.0)
        if lidar:
            lidar.stop()
        if pi and pi.connected:
            pi.stop()
        print("[HWCtrl] Shutdown complete.")


def run_hardware_controller(shared_data):
    """Main function to start the combined hardware controller"""
    print("[HWCtrl] Starting continuous scanning hardware controller process...")
    try:
        combined_controller_process(shared_data)
    except KeyboardInterrupt:
        print("[HWCtrl] Process interrupted by user.")
    except Exception as e:
        import traceback
        print(f"[HWCtrl] Unexpected error: {e}")
        traceback.print_exc()
    print("[HWCtrl] Hardware controller process stopped.")