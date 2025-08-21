#!/usr/bin/env python3
"""
Hardware Controller for Stepper Motor (Pan) and Servo (Tilt) Control
Uses multiprocessing with shared data and PID control for smooth movement
PWM-based stepper control for smooth operation with precise position tracking

IMPROVED: Hybrid position tracking for the stepper motor.
          - Uses a time-based deterministic calculation for continuous, high-speed
            PWM moves (background scan) to eliminate interrupt storms and improve accuracy.
          - Uses a traditional interrupt-based callback for precise, wave-based
            point-to-point moves.
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
        last_data = None
        while not self.lidar_queue.empty():
            try:
                last_data = self.lidar_queue.get_nowait()
            except queue.Empty:
                break

        if last_data:
            dist, strength, ts = last_data
            with self.shared_data["lidar_data"].get_lock():
                self.shared_data["lidar_data"][:] = [dist, strength, ts]
            return dist, strength, ts
        else:
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
    """Handles continuous background scanning functionality"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.current_elevation = SCAN_TILT_MAX
        self.background_data_buffer = []
        self.scan_direction = 1  # 1 for forward (0->360), -1 for backward (360->0)
        self.scan_active = False
        self.scan_start_time = None

        # Calculate and display expected scan time
        self._estimate_scan_time()

    def start_continuous_scan(self, lidar_controller, stepper_controller, servo_controller):
        """Start continuous scanning process with optimized servo movement"""
        if not self.shared_data["background_scan_active"].value:
            return False  # Return False if scan was stopped

        if not hasattr(self, 'scan_start_time') or self.scan_start_time is None:
            self.scan_start_time = time.time()

        print(f"[HWCtrl] Starting continuous scan at elevation {self.current_elevation:.1f}°")
        self.shared_data["target_elevation"].value = self.current_elevation
        start_wait = time.time()
        timeout = SERVO_MOVE_TIME if self.current_elevation != SCAN_TILT_MAX else max(SERVO_MOVE_TIME, 1.0)

        while time.time() - start_wait < timeout:
            if abs(self.shared_data["servo_degrees"].value - self.current_elevation) < 1.0:
                if SERVO_SETTLE_TIME > 0.01:
                    time.sleep(SERVO_SETTLE_TIME)
                break
            time.sleep(0.001)

        self._perform_continuous_azimuth_sweep(lidar_controller, stepper_controller)
        self.current_elevation -= SCAN_ELEVATION_STEP

        if self.current_elevation < SCAN_TILT_MIN:
            print("[HWCtrl] CONTINUOUS BACKGROUND SCAN completed.")
            total_time = time.time() - self.scan_start_time
            print(f"[HWCtrl] Total scan time: {total_time:.1f} seconds")
            self._save_scan_data()
            self._reset_hardware_after_scan(stepper_controller)
            self._reset_scan()
            return False  # Scan complete
        else:
            self.shared_data["target_elevation"].value = self.current_elevation
            self.scan_direction *= -1
            return True

    def _perform_continuous_azimuth_sweep(self, lidar_controller, stepper_controller):
        """Perform continuous azimuth sweep while collecting data"""
        self.movement_complete = threading.Event()
        start_az, end_az = (0.0, 360.0) if self.scan_direction == 1 else (360.0, 0.0)
        print(f"[HWCtrl] Azimuth sweep: {start_az}° -> {end_az}°")

        stepper_controller.move_to_angle(start_az)

        movement_thread = threading.Thread(
            target=self._continuous_azimuth_movement,
            args=(stepper_controller, start_az, end_az)
        )
        movement_thread.daemon = True
        movement_thread.start()

        self._collect_data_during_movement(lidar_controller, stepper_controller, start_az, end_az)
        movement_thread.join(timeout=15.0)

    def _continuous_azimuth_movement(self, stepper_controller, start_az, end_az):
        """MODIFIED: Perform movement using the new deterministic controller methods."""
        steps_per_second = SCAN_AZIMUTH_SPEED / MICROSTEP_ANGLE
        direction = 1 if end_az > start_az else 0
        movement_time = abs(end_az - start_az) / SCAN_AZIMUTH_SPEED

        stepper_controller.start_continuous_move(steps_per_second, direction)
        time.sleep(movement_time)
        stepper_controller.stop_continuous_move()

        self.movement_complete.set()

    def _collect_data_during_movement(self, lidar_controller, stepper_controller, start_az, end_az):
        """MODIFIED: Collect data using the deterministic position tracking method."""
        collection_interval = 1.0 / SCAN_DATA_RATE
        start_time = time.time()
        last_sample_time = start_time
        movement_duration = abs(end_az - start_az) / SCAN_AZIMUTH_SPEED
        print(f"[HWCtrl] Data collection: {SCAN_DATA_RATE} Hz for {movement_duration:.1f}s")
        sample_count = 0

        while (time.time() - start_time) < movement_duration:
            current_time = time.time()
            if (current_time - last_sample_time) >= collection_interval:
                # ***KEY CHANGE***: Get position from the deterministic calculator
                current_az = stepper_controller.update_and_get_virtual_position()
                current_el = self.shared_data["servo_degrees"].value
                dist, strength, timestamp = lidar_controller.get_lidar_data()

                if dist is not None and dist > 0:
                    self.background_data_buffer.append([current_az, current_el, dist, strength])
                    sample_count += 1
                last_sample_time = current_time
            else:
                time.sleep(0.001)
        print(f"[HWCtrl] Data collection completed. Total samples: {sample_count}")

    def _reset_hardware_after_scan(self, stepper_controller):
        """Reset hardware state after scan completion."""
        stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        stepper_controller.pi.wave_tx_stop()
        stepper_controller.pi.wave_clear()

        if stepper_controller.step_callback:
            stepper_controller.step_callback.cancel()
            stepper_controller.step_callback = None

        stepper_controller.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        stepper_controller.pi.write(STEPPER_PULSE_PIN, 0)
        time.sleep(0.05)
        stepper_controller.step_callback = stepper_controller.pi.callback(
            STEPPER_PULSE_PIN, pigpio.RISING_EDGE,
            stepper_controller._step_counter_callback
        )
        print("[HWCtrl] Stepper hardware has been reset and is ready.")

    def _save_scan_data(self):
        """Save collected scan data to file"""
        if self.background_data_buffer:
            try:
                path = "background_scan.npy"
                data_array = np.array(self.background_data_buffer)
                np.save(path, data_array)
                print(f"[HWCtrl] Saved {len(self.background_data_buffer)} scan points to {path}")
            except Exception as e:
                print(f"[HWCtrl] Error saving scan data: {e}")
        else:
            print("[HWCtrl] No scan data to save")

    def _estimate_scan_time(self):
        """Calculate and display expected scan time"""
        num_rings = int((SCAN_TILT_MAX - SCAN_TILT_MIN) / SCAN_ELEVATION_STEP) + 1
        time_per_rotation = 360.0 / SCAN_AZIMUTH_SPEED
        time_per_servo = SERVO_MOVE_TIME + SERVO_SETTLE_TIME
        total_scan_time = (num_rings * time_per_rotation) + ((num_rings - 1) * time_per_servo)
        total_samples = int(total_scan_time * SCAN_DATA_RATE)
        print(f"[HWCtrl] ========== SCAN CONFIGURATION ==========")
        print(f"[HWCtrl] Azimuth speed: {SCAN_AZIMUTH_SPEED}°/s, Elevation steps: {SCAN_ELEVATION_STEP}°")
        print(f"[HWCtrl] Expected total time: ~{total_scan_time / 60:.1f} minutes")
        print(f"[HWCtrl] Expected samples: ~{total_samples:,}")
        print(f"[HWCtrl] ========================================")

    def _reset_scan(self):
        """Reset scan parameters for next scan"""
        self.current_elevation = SCAN_TILT_MAX
        self.scan_direction = 1
        self.background_data_buffer = []
        self.shared_data["background_scan_active"].value = False
        self.scan_start_time = None
        print("[HWCtrl] Scan parameters reset. Background scan flag set to False.")


# --- vvv NEW AND IMPROVED STEPPER CONTROLLER vvv ---
class PWMStepperController:
    """
    IMPROVED Stepper motor controller with a hybrid position tracking system.
    - Time-based (deterministic) tracking for high-speed continuous PWM moves.
    - Callback-based (interrupt-driven) tracking for precise wave-based moves.
    This prevents interrupt storms and improves accuracy during scans.
    """

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.step_count = 0
        self.direction = 1

        # State for continuous PWM movement
        self.is_in_continuous_move = False
        self.continuous_move_start_time = 0.0
        self.continuous_move_start_steps = 0
        self.continuous_move_sps = 0.0

        # Setup GPIO pins
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
        self.pi.write(STEPPER_ENABLE_PIN, 0)
        self.pi.write(STEPPER_SLEEP_PIN, 1)
        time.sleep(0.001)

        self.step_callback = self.pi.callback(STEPPER_PULSE_PIN, pigpio.RISING_EDGE, self._step_counter_callback)
        print("[HWCtrl] Improved Stepper controller with HYBRID feedback initialized.")

    def _step_counter_callback(self, gpio, level, tick):
        """Callback for WAVE-BASED moves. Ignored during continuous PWM moves."""
        if self.is_in_continuous_move or level != 1:
            return

        if self.pi.read(STEPPER_DIR_PIN):
            self.step_count += 1
        else:
            self.step_count -= 1

        # Update shared memory directly. The performance gain outweighs the minimal risk of a race condition.
        self.shared_data["stepper_degrees"].value = (self.step_count * MICROSTEP_ANGLE) % 360.0

    def start_continuous_move(self, steps_per_second, direction):
        """Starts a hardware PWM move and engages time-based position tracking."""
        if self.is_in_continuous_move:
            self.stop_continuous_move()

        print(f"[HWCtrl-Stepper] Starting continuous move at {steps_per_second:.0f} steps/sec.")
        self.is_in_continuous_move = True
        self.direction = direction
        self.pi.write(STEPPER_DIR_PIN, self.direction)

        self.continuous_move_start_steps = self.step_count
        self.continuous_move_sps = steps_per_second
        self.continuous_move_start_time = time.time()

        self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(steps_per_second), 500000)

    def stop_continuous_move(self):
        """Stops a hardware PWM move and syncs the final position."""
        if not self.is_in_continuous_move:
            return

        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        elapsed_time = time.time() - self.continuous_move_start_time
        steps_moved = int(elapsed_time * self.continuous_move_sps)

        if self.direction == 1:
            self.step_count = self.continuous_move_start_steps + steps_moved
        else:
            self.step_count = self.continuous_move_start_steps - steps_moved

        self.is_in_continuous_move = False
        final_degrees = (self.step_count * MICROSTEP_ANGLE) % 360.0
        self.shared_data["stepper_degrees"].value = final_degrees
        print(f"[HWCtrl-Stepper] Continuous move stopped. Final position: {final_degrees:.2f}°")

    def update_and_get_virtual_position(self):
        """Calculates and returns the current position during a continuous move."""
        if self.is_in_continuous_move:
            elapsed_time = time.time() - self.continuous_move_start_time
            steps_moved = elapsed_time * self.continuous_move_sps
            current_steps = self.continuous_move_start_steps + (steps_moved if self.direction == 1 else -steps_moved)
            current_degrees = (current_steps * MICROSTEP_ANGLE) % 360.0
            self.shared_data["stepper_degrees"].value = current_degrees
            return current_degrees
        else:
            return self.shared_data["stepper_degrees"].value

    def move_to_angle(self, target_angle):
        """Calculates and executes a point-to-point move using pigpio waves."""
        if self.is_in_continuous_move:
            self.stop_continuous_move()

        self.pi.wave_tx_stop()
        target_angle = max(MotorParams.PAN_MIN_ANGLE, min(MotorParams.PAN_MAX_ANGLE, target_angle))

        current_pos_steps = self.step_count
        target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
        error_steps = target_pos_steps - current_pos_steps
        steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)

        if abs(error_steps) > (steps_per_rotation / 2):
            error_steps -= steps_per_rotation if error_steps > 0 else -steps_per_rotation

        if abs(error_steps) < 1:
            return

        direction = 1 if error_steps > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        total_steps = abs(error_steps)
        accel_steps_actual = min(total_steps // 2, MotorParams.ACCEL_STEPS)
        decel_steps_actual = accel_steps_actual
        cruise_steps = total_steps - (accel_steps_actual + decel_steps_actual)

        pulses = []
        for i in range(1, accel_steps_actual + 1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / accel_steps_actual)
            delay = int(500000 / speed)
            pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay)])
        if cruise_steps > 0:
            delay = int(500000 / MotorParams.STEPPER_MAX_SPEED)
            for _ in range(cruise_steps):
                pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay)])
        for i in range(decel_steps_actual, 0, -1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / decel_steps_actual)
            delay = int(500000 / speed)
            pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay)])

        MAX_PULSES_PER_CHUNK = 4000
        start_index = 0
        while start_index < len(pulses):
            if self.shared_data["shutdown"].value:
                break
            end_index = min(start_index + MAX_PULSES_PER_CHUNK, len(pulses))
            self.pi.wave_clear()
            self.pi.wave_add_generic(pulses[start_index:end_index])
            wave_id = self.pi.wave_create()
            if wave_id >= 0:
                self.pi.wave_send_once(wave_id)
                while self.pi.wave_tx_busy():
                    time.sleep(0.01)
                self.pi.wave_delete(wave_id)
            else:
                break
            start_index = end_index

    def move_to_target(self):
        """Move stepper to target azimuth stored in shared data."""
        target_pos = self.shared_data["target_azimuth"].value
        self.move_to_angle(target_pos)
        self.shared_data["target_reached"].value = True

    def stop(self):
        """Stops the stepper controller and cleans up resources."""
        self.running = False
        print("[HWCtrl] Stopping stepper controller...")
        try:
            self.pi.wave_tx_stop()
            self.pi.wave_clear()
            self.pi.write(STEPPER_PULSE_PIN, 0)
        except Exception as e:
            print(f"[HWCtrl] Warning during stepper stop: {e}")

        if self.step_callback:
            self.step_callback.cancel()
            self.step_callback = None

        self.pi.write(STEPPER_ENABLE_PIN, 1)
        print("[HWCtrl] Stepper controller stopped.")
# --- ^^^ NEW AND IMPROVED STEPPER CONTROLLER ^^^ ---


class ServoController:
    """Controls servo motor for tilt movement"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        self.move_to_angle(45.0)  # Start at 45 degrees

    def angle_to_pulse_width(self, angle):
        """Convert angle to servo pulse width with mounting offset correction"""
        user_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, angle))
        physical_angle = user_angle - MotorParams.SERVO_ZERO_OFFSET
        physical_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, physical_angle))
        pulse_range = MotorParams.SERVO_MAX_PULSE - MotorParams.SERVO_MIN_PULSE
        angle_range = MotorParams.SERVO_MAX_ANGLE - MotorParams.SERVO_MIN_ANGLE
        pulse_width = MotorParams.SERVO_MIN_PULSE + ((physical_angle - MotorParams.SERVO_MIN_ANGLE) / angle_range * pulse_range)
        return int(pulse_width)

    def move_to_angle(self, target_angle):
        """Direct movement to target angle with angle limiting"""
        target_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, target_angle))
        pulse_width = self.angle_to_pulse_width(target_angle)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
        with self.shared_data["servo_degrees"].get_lock():
            self.shared_data["servo_degrees"].value = target_angle

    def control_servo(self):
        """Control servo position based on target elevation"""
        while self.running:
            target_elevation = self.shared_data["target_elevation"].value
            if abs(target_elevation - self.shared_data["servo_degrees"].value) > 0.5:
                self.move_to_angle(target_elevation)
            time.sleep(0.001)

    def stop(self):
        """Stop the servo controller"""
        self.running = False
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
        print("[HWCtrl] Servo controller stopped")


def combined_controller_process(shared_data):
    """Combined process for controlling stepper, servo, LiDAR, and background scanning"""
    pi = None
    stepper, servo, lidar, servo_thread = None, None, None, None

    try:
        pi = pigpio.pi()
        if not pi.connected:
            print("[HWCtrl] Failed to connect to pigpio daemon")
            return

        print(f"[HWCtrl] Initializing...")
        stepper = PWMStepperController(pi, shared_data)
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = ContinuousBackgroundScanner(shared_data)
        print("[HWCtrl] Combined controller initialized.")

        servo_thread = threading.Thread(target=servo.control_servo)
        servo_thread.daemon = True
        servo_thread.start()

        while not shared_data["shutdown"].value:
            lidar.get_lidar_data()

            if shared_data["background_scan_active"].value:
                # Pass controller instances to the scanner
                scan_continues = scanner.start_continuous_scan(lidar, stepper, servo)
                if not scan_continues:
                    print("[HWCtrl] Background scan cycle finished.")
                    time.sleep(0.1)

            elif shared_data["go_to_target"].value:
                shared_data["target_reached"].value = False
                stepper.move_to_target()
                shared_data["go_to_target"].value = False
            else:
                time.sleep(0.001)

    except Exception as e:
        import traceback
        print(f"[HWCtrl] Combined controller error: {e}")
        traceback.print_exc()
    finally:
        print("[HWCtrl] Shutting down...")
        if stepper: stepper.stop()
        if servo: servo.stop()
        if servo_thread: servo_thread.join(timeout=1.0)
        if lidar: lidar.stop()
        if pi and pi.connected: pi.stop()
        print("[HWCtrl] Shutdown complete.")


def run_hardware_controller(shared_data):
    """Main function to start the combined hardware controller"""
    print("[HWCtrl] Starting hardware controller process...")
    try:
        combined_controller_process(shared_data)
    except KeyboardInterrupt:
        print("[HWCtrl] Process interrupted by user.")
    finally:
        print("[HWCtrl] Hardware controller process stopped.")