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
    KP = 0.2 # Higher proportional for faster response
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

        # Record scan start time on first run
        if not hasattr(self, 'scan_start_time') or self.scan_start_time is None:
            self.scan_start_time = time.time()

        print(f"[HWCtrl] Starting continuous scan at elevation {self.current_elevation:.1f}°")
        print(f"[HWCtrl] Scan profile: {SCAN_AZIMUTH_SPEED}°/s, {SCAN_ELEVATION_STEP}° steps")

        # Move servo to current elevation using shared data system
        print(f"[HWCtrl] Moving servo to {self.current_elevation:.1f}°")

        # Set target elevation in shared data for servo thread to handle
        self.shared_data["target_elevation"].value = self.current_elevation

        # Adaptive wait based on scan profile
        start_wait = time.time()
        timeout = SERVO_MOVE_TIME

        # For first move, always wait full time
        if self.current_elevation == SCAN_TILT_MAX:
            timeout = max(SERVO_MOVE_TIME, 1.0)

        while time.time() - start_wait < timeout:
            current_servo = self.shared_data["servo_degrees"].value
            if abs(current_servo - self.current_elevation) < 1.0:  # Within 1 degree
                print(f"[HWCtrl] Servo reached {current_servo:.1f}° (target: {self.current_elevation:.1f}°)")
                if SERVO_SETTLE_TIME > 0.01:  # Only settle if needed
                    time.sleep(SERVO_SETTLE_TIME)
                break
            time.sleep(0.001)  # Faster polling
        else:
            # Don't stop scan on timeout for fast profiles
            if SCAN_AZIMUTH_SPEED < 60:  # Only warn for slower scans
                print(f"[HWCtrl] Warning: Servo move timeout. Current: {self.shared_data['servo_degrees'].value:.1f}°")

        # Perform continuous azimuth sweep
        self._perform_continuous_azimuth_sweep(lidar_controller, stepper_controller)

        # Update elevation for next ring
        self.current_elevation -= SCAN_ELEVATION_STEP

        # Check if scan is complete (after processing the last ring at elevation 0)
        if self.current_elevation < SCAN_TILT_MIN:
            print("[HWCtrl] CONTINUOUS BACKGROUND SCAN completed.")
            total_time = time.time() - self.scan_start_time
            print(f"[HWCtrl] Total scan time: {total_time:.1f} seconds")
            self._save_scan_data()
            self._reset_hardware_after_scan(stepper_controller)
            self._reset_scan()
            return False  # Scan complete
        else:
            # Pre-move servo to next elevation during data processing
            print(f"[HWCtrl] Pre-positioning servo to next elevation: {self.current_elevation:.1f}°")
            self.shared_data["target_elevation"].value = self.current_elevation
            self.scan_direction *= -1  # Alternate direction for each ring
            return True  # Scan continues

    def _perform_continuous_azimuth_sweep(self, lidar_controller, stepper_controller):
        """Perform continuous azimuth sweep while collecting data"""

        # Determine start and end positions based on scan direction
        if self.scan_direction == 1:
            start_az = 0.0
            end_az = 360.0
            print(f"[HWCtrl] Forward sweep: 0° -> 360°")
        else:
            start_az = 360.0
            end_az = 0.0
            print(f"[HWCtrl] Backward sweep: 360° -> 0°")

        # Move to start position
        stepper_controller.move_to_angle(start_az)

        # Start continuous movement
        print(f"[HWCtrl] Starting continuous movement from {start_az:.1f}° to {end_az:.1f}°")
        movement_thread = threading.Thread(
            target=self._continuous_azimuth_movement,
            args=(stepper_controller, start_az, end_az)
        )
        movement_thread.daemon = True
        movement_thread.start()

        # Collect data during movement
        self._collect_data_during_movement(lidar_controller, start_az, end_az)

        # Wait for movement to complete
        movement_thread.join(timeout=15.0)  # 15 second timeout for full rotation

    def _continuous_azimuth_movement(self, stepper_controller, start_az, end_az):
        """Perform continuous azimuth movement in separate thread with optimized speed"""

        # Calculate movement parameters
        total_distance = abs(end_az - start_az)
        movement_time = total_distance / SCAN_AZIMUTH_SPEED
        steps_per_second = SCAN_AZIMUTH_SPEED / MICROSTEP_ANGLE

        print(f"[HWCtrl] Movement: {total_distance:.1f}° in {movement_time:.1f}s at {steps_per_second:.0f} steps/sec")

        # Set direction
        direction = 1 if end_az > start_az else 0
        stepper_controller.pi.write(STEPPER_DIR_PIN, direction)

        # Start continuous PWM at calculated frequency
        # Use higher speed for fast scans
        if SCAN_AZIMUTH_SPEED > 60:
            # For high-speed scans, start at cruise speed immediately
            frequency = min(int(steps_per_second), MotorParams.STEPPER_MAX_SPEED)
            stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, frequency, 500000)
        else:
            # For normal scans, ramp up
            target_freq = min(int(steps_per_second), MotorParams.STEPPER_MAX_SPEED)
            ramp_steps = 10
            for i in range(ramp_steps):
                freq = int((target_freq / ramp_steps) * (i + 1))
                stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)
                time.sleep(0.001)

        # Setup step counting for position tracking
        if not stepper_controller.step_callback:
            stepper_controller.step_callback = stepper_controller.pi.callback(
                STEPPER_PULSE_PIN, pigpio.RISING_EDGE, stepper_controller._step_counter_callback
            )

        # Run for calculated time
        start_time = time.time()
        while (time.time() - start_time) < movement_time and stepper_controller.running:
            time.sleep(0.001)

        # Stop movement with deceleration for slower scans
        if SCAN_AZIMUTH_SPEED <= 60:
            # Gentle deceleration
            current_freq = int(steps_per_second)
            while current_freq > 1000:
                current_freq = int(current_freq * 0.9)
                stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, current_freq, 500000)
                time.sleep(0.001)

        # Full stop
        stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)

        final_pos = stepper_controller.shared_data["stepper_degrees"].value
        print(f"[HWCtrl] Continuous movement completed. Final position: {final_pos:.1f}°")

    def _reset_hardware_after_scan(self, stepper_controller):
        """Reset hardware state after scan completion"""
        # Ensure PWM is fully stopped
        stepper_controller.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Clear any pending waves
        stepper_controller.pi.wave_tx_stop()
        stepper_controller.pi.wave_clear()

        # Small delay for hardware to settle
        time.sleep(0.05)

        # Ensure callback is properly set up for normal operation
        if stepper_controller.step_callback:
            stepper_controller.step_callback.cancel()

        stepper_controller.step_callback = stepper_controller.pi.callback(
            STEPPER_PULSE_PIN, pigpio.RISING_EDGE,
            stepper_controller._step_counter_callback
        )


    def _collect_data_during_movement(self, lidar_controller, start_az, end_az):
        """Collect LiDAR data during continuous movement"""

        collection_interval = 1.0 / SCAN_DATA_RATE  # Time between samples
        start_time = time.time()
        last_sample_time = start_time

        total_distance = abs(end_az - start_az)
        movement_duration = total_distance / SCAN_AZIMUTH_SPEED

        print(f"[HWCtrl] Data collection: {SCAN_DATA_RATE} Hz for {movement_duration:.1f}s")

        sample_count = 0

        while (time.time() - start_time) < movement_duration:
            current_time = time.time()

            # Check if it's time for next sample
            if (current_time - last_sample_time) >= collection_interval:

                # Get current position
                current_az = self.shared_data["stepper_degrees"].value
                current_el = self.shared_data["servo_degrees"].value  # Use actual servo position

                # Get LiDAR data
                dist, strength, timestamp = lidar_controller.get_lidar_data()

                if dist is not None and dist > 0:
                    # Store sample
                    sample = [current_az, current_el, dist, strength]
                    self.background_data_buffer.append(sample)
                    sample_count += 1



                last_sample_time = current_time
            else:
                # Short sleep to prevent CPU spinning
                time.sleep(0.001)

        print(f"[HWCtrl] Data collection completed. Total samples: {sample_count}")

    def _save_scan_data(self):
        """Save collected scan data to file"""
        if self.background_data_buffer:
            try:
                # Use the path from shared_data
                path = self.shared_data["background_path"].value
                data_array = np.array(self.background_data_buffer)
                np.save(path, data_array)
                print(f"[HWCtrl] Saved {len(self.background_data_buffer)} scan points to {path}")
                print(f"[HWCtrl] Data shape: {data_array.shape}")
                print(f"[HWCtrl] Azimuth range: {data_array[:, 0].min():.1f}° to {data_array[:, 0].max():.1f}°")
                print(f"[HWCtrl] Elevation range: {data_array[:, 1].min():.1f}° to {data_array[:, 1].max():.1f}°")
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

        samples_per_rotation = int(time_per_rotation * SCAN_DATA_RATE)
        total_samples = samples_per_rotation * num_rings

        print(f"[HWCtrl] ========== SCAN CONFIGURATION ==========")
        print(
            f"[HWCtrl] Profile: {'ULTRA_FAST' if SCAN_AZIMUTH_SPEED >= 120 else 'FAST' if SCAN_AZIMUTH_SPEED >= 90 else 'NORMAL' if SCAN_AZIMUTH_SPEED >= 60 else 'HIGH_QUALITY'}")
        print(f"[HWCtrl] Azimuth speed: {SCAN_AZIMUTH_SPEED}°/s")
        print(f"[HWCtrl] Elevation steps: {SCAN_ELEVATION_STEP}°")
        print(f"[HWCtrl] Number of rings: {num_rings}")
        print(f"[HWCtrl] Time per rotation: {time_per_rotation:.1f}s")
        print(f"[HWCtrl] Expected total time: {total_scan_time:.1f}s ({total_scan_time / 60:.1f} minutes)")
        print(f"[HWCtrl] Expected samples: ~{total_samples:,}")
        print(f"[HWCtrl] ========================================")

    def _reset_scan(self):
        """Reset scan parameters for next scan"""
        self.current_elevation = SCAN_TILT_MAX
        self.scan_direction = 1
        self.background_data_buffer = []
        self.shared_data["background_scan_active"].value = False
        self.scan_start_time = None
        print("[HWCtrl] Scan parameters reset for next scan")
        print("[HWCtrl] Background scan flag set to False")


class PWMStepperController:
    """
    Open-loop stepper motor controller using pre-calculated motion profiles
    and pigpio waveforms for precise, hardware-timed execution.
    This version includes a step-counting callback to provide LIVE angle updates.
    FIXED: Now handles moves >180 degrees without crashing pigpio.
    """

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.step_count = 0  # Tracks the motor's absolute position in steps

        # Setup GPIO pins
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        # Enable stepper driver
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Active low
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake up driver
        time.sleep(0.001)

        # Set up the callback for live position tracking
        self.step_callback = self.pi.callback(STEPPER_PULSE_PIN, pigpio.RISING_EDGE, self._step_counter_callback)

        print("[HWCtrl] Open-Loop Stepper controller with LIVE feedback initialized.")

    def _step_counter_callback(self, gpio, level, tick):
        """
        Callback function that triggers on every step.
        This provides the LIVE position updates.
        """
        # This callback is triggered by the hardware pulses from the waveform
        if level == 1:  # Rising edge
            direction = self.pi.read(STEPPER_DIR_PIN)
            if direction:
                self.step_count += 1
            else:
                self.step_count -= 1

            # Wrap around at 360 degrees (6400 steps)
            steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
            self.step_count = self.step_count % steps_per_rotation

            degrees = self.step_count * MICROSTEP_ANGLE

            # Update shared position data immediately
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value = degrees

    def move_to_angle(self, target_angle):
        """
        Moves the stepper motor using hardware PWM for large moves and waveforms for small moves.
        This prevents pigpio crashes on moves >180 degrees.
        """
        # Stop any existing hardware PWM before starting
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()

        # Clamp target angle
        target_angle = max(MotorParams.PAN_MIN_ANGLE, min(MotorParams.PAN_MAX_ANGLE, target_angle))

        print(f"[HWCtrl-Stepper] Moving to {target_angle:.3f}°")

        # Calculate the move
        current_pos_steps = self.step_count
        target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
        error_steps = target_pos_steps - current_pos_steps

        # Find shortest path
        steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
        if abs(error_steps) > (steps_per_rotation / 2):
            if error_steps > 0:
                error_steps -= steps_per_rotation
            else:
                error_steps += steps_per_rotation

        total_steps = abs(error_steps)

        if total_steps < 1:
            print("[HWCtrl-Stepper] Already at target position.")
            return

        # Set direction
        direction = 1 if error_steps > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)
        print(f"[HWCtrl-Stepper] Steps to move: {total_steps}, Direction: {'CW' if direction else 'CCW'}")

        # CRITICAL: Use different methods based on move size
        # Threshold for waveform vs PWM (adjust as needed)
        WAVEFORM_THRESHOLD = 1600  # About 90 degrees of movement

        if total_steps > WAVEFORM_THRESHOLD:
            # Use hardware PWM for large moves
            self._move_with_pwm(total_steps, direction)
        else:
            # Use waveform for small, precise moves
            self._move_with_waveform(total_steps, direction)

        # Log final position
        final_pos_deg = self.shared_data["stepper_degrees"].value
        print(f"[HWCtrl-Stepper] Movement complete. Final position: {final_pos_deg:.3f}°")

    def _move_with_pwm(self, total_steps, direction):
        """
        Execute large moves using hardware PWM with step counting verification.
        More reliable for moves >90 degrees with maintained accuracy.
        """
        print(f"[HWCtrl-Stepper] Using PWM mode for {total_steps} steps")

        # Record starting position for accuracy verification
        start_step_count = self.step_count
        target_step_count = start_step_count + (total_steps if direction else -total_steps)

        # Calculate acceleration parameters
        accel_steps = min(MotorParams.ACCEL_STEPS, total_steps // 3)
        decel_steps = min(MotorParams.ACCEL_STEPS, total_steps // 3)
        cruise_steps = total_steps - accel_steps - decel_steps

        # Acceleration phase - count actual steps
        print("[HWCtrl-Stepper] Accelerating...")
        accel_start = self.step_count
        for i in range(1, accel_steps + 1):
            if not self.running:
                break
            speed = MotorParams.STEPPER_MIN_SPEED + (
                    (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                    (i / accel_steps)
            )
            freq = min(int(speed), MotorParams.STEPPER_MAX_SPEED)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)

            # Wait for one step at this frequency
            time.sleep(1.0 / freq)

        accel_actual = abs(self.step_count - accel_start)

        # Cruise phase - monitor actual step count
        if cruise_steps > 0 and self.running:
            print(f"[HWCtrl-Stepper] Cruising for {cruise_steps} steps...")
            cruise_start = self.step_count
            cruise_freq = min(MotorParams.STEPPER_MAX_SPEED, MotorParams.STEPPER_CRUISE_SPEED)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, cruise_freq, 500000)

            # Monitor actual steps instead of just time
            steps_needed = cruise_steps
            timeout = time.time() + (steps_needed / cruise_freq) * 1.5  # 50% safety margin

            while self.running and time.time() < timeout:
                current_steps = abs(self.step_count - cruise_start)
                if current_steps >= steps_needed:
                    break
                time.sleep(0.001)  # Fast polling for accuracy

            cruise_actual = abs(self.step_count - cruise_start)
            if cruise_actual < cruise_steps * 0.95:  # Warning if >5% steps missed
                print(f"[HWCtrl-Stepper] Warning: Expected {cruise_steps} cruise steps, got {cruise_actual}")

        # Deceleration phase
        print("[HWCtrl-Stepper] Decelerating...")
        decel_start = self.step_count
        for i in range(decel_steps, 0, -1):
            if not self.running:
                break
            speed = MotorParams.STEPPER_MIN_SPEED + (
                    (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                    (i / decel_steps)
            )
            freq = min(int(speed), MotorParams.STEPPER_MAX_SPEED)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)

            time.sleep(1.0 / freq)

        # Stop PWM
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)

        # Verify final accuracy
        actual_steps_moved = abs(self.step_count - start_step_count)
        accuracy = (actual_steps_moved / total_steps) * 100

        print(
            f"[HWCtrl-Stepper] PWM move complete. Target: {total_steps}, Actual: {actual_steps_moved} ({accuracy:.1f}% accuracy)")

        # If significant error, do a correction move
        error_steps = abs(target_step_count - self.step_count)
        if error_steps > 2:  # More than 2 steps off
            print(f"[HWCtrl-Stepper] Correcting position error of {error_steps} steps")
            # Use waveform for precise correction
            self._correction_move(error_steps, target_step_count > self.step_count)

    def _move_with_waveform(self, total_steps, direction):
        """
        Execute small moves using pigpio waveforms for precise control.
        Best for moves <90 degrees.
        """
        print(f"[HWCtrl-Stepper] Using waveform mode for {total_steps} steps")

        # Calculate acceleration/deceleration steps
        if total_steps <= MotorParams.ACCEL_STEPS * 2:
            accel_steps_actual = total_steps // 2
            decel_steps_actual = total_steps - accel_steps_actual
        else:
            accel_steps_actual = MotorParams.ACCEL_STEPS
            decel_steps_actual = MotorParams.ACCEL_STEPS

        # Build the waveform
        pulses = []

        # Limit check to prevent excessive pulses
        MAX_PULSES = 3000  # pigpio limit is around 3000-4000 pulses
        if total_steps * 2 > MAX_PULSES:
            print(f"[HWCtrl-Stepper] Warning: Move too large for waveform ({total_steps * 2} pulses)")
            # Fall back to PWM
            self._move_with_pwm(total_steps, direction)
            return

        # Acceleration
        for i in range(1, accel_steps_actual + 1):
            speed = MotorParams.STEPPER_MIN_SPEED + (
                    (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                    (i / accel_steps_actual)
            )
            delay_us = int(500000 / speed)
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        # Cruise
        cruise_steps = total_steps - (accel_steps_actual + decel_steps_actual)
        if cruise_steps > 0:
            delay_us = int(500000 / MotorParams.STEPPER_MAX_SPEED)
            for _ in range(cruise_steps):
                pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
                pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        # Deceleration
        for i in range(decel_steps_actual, 0, -1):
            speed = MotorParams.STEPPER_MIN_SPEED + (
                    (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) *
                    (i / decel_steps_actual)
            )
            delay_us = int(500000 / speed)
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        # Add wave to pigpio
        try:
            self.pi.wave_add_generic(pulses)
            wave_id = self.pi.wave_create()

            if wave_id >= 0:
                print(f"[HWCtrl-Stepper] Sending wave {wave_id} with {len(pulses) // 2} pulses")
                self.pi.wave_send_once(wave_id)

                # Wait for completion
                while self.pi.wave_tx_busy() and self.running:
                    time.sleep(0.01)

                # Clean up
                self.pi.wave_delete(wave_id)
                print("[HWCtrl-Stepper] Wave complete")
            else:
                print(f"[HWCtrl-Stepper] Wave creation failed (id={wave_id}), falling back to PWM")
                self._move_with_pwm(total_steps, direction)

        except Exception as e:
            print(f"[HWCtrl-Stepper] Waveform error: {e}, falling back to PWM")
            self._move_with_pwm(total_steps, direction)

    def _correction_move(self, steps, forward):
        """
        Small correction move using precise waveform control.
        Used to correct any positioning errors after PWM moves.
        """
        if steps < 1 or steps > 100:  # Safety limit for correction
            return

        direction = 1 if forward else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        # Simple constant speed correction
        freq = 1000  # 1kHz for correction moves
        delay_us = int(500000 / freq)

        pulses = []
        for _ in range(steps):
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        self.pi.wave_add_generic(pulses)
        wave_id = self.pi.wave_create()

        if wave_id >= 0:
            self.pi.wave_send_once(wave_id)
            while self.pi.wave_tx_busy():
                time.sleep(0.001)
            self.pi.wave_delete(wave_id)
            print(f"[HWCtrl-Stepper] Correction complete")

    def move_to_target(self):
        """Move stepper to target azimuth."""
        target_pos = self.shared_data["target_azimuth"].value
        self.move_to_angle(target_pos)
        self.shared_data["target_reached"].value = True

    def stop(self):
        """Stops the stepper controller and cleans up resources."""
        self.running = False
        print("[HWCtrl] Stopping stepper controller...")

        # Stop any active waveforms or PWM
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()

        # Cancel the step counting callback
        if self.step_callback:
            self.step_callback.cancel()
            self.step_callback = None

        # Failsafe to ensure the pulse pin is off
        self.pi.write(STEPPER_PULSE_PIN, 0)

        # Disable the stepper driver
        self.pi.write(STEPPER_ENABLE_PIN, 1)
        print("[HWCtrl] Stepper controller stopped.")


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
        user_angle = max(MotorParams.SERVO_MIN_ANGLE,
                         min(MotorParams.SERVO_MAX_ANGLE, angle))

        # Apply mounting offset to get physical servo angle
        # If mounting causes servo to point up when it should be at 0,
        # we subtract the offset to compensate
        physical_angle = user_angle - MotorParams.SERVO_ZERO_OFFSET

        # Ensure physical angle is within servo's physical limits
        physical_angle = max(MotorParams.SERVO_MIN_ANGLE,
                             min(MotorParams.SERVO_MAX_ANGLE, physical_angle))

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
        target_angle = max(MotorParams.SERVO_MIN_ANGLE,
                           min(MotorParams.SERVO_MAX_ANGLE, target_angle))

        print(f"[HWCtrl-Servo] Moving to {target_angle:.1f}° (user angle)")

        pulse_width = self.angle_to_pulse_width(target_angle)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

        # Update shared data with user-visible angle
        with self.shared_data["servo_degrees"].get_lock():
            self.shared_data["servo_degrees"].value = target_angle

        physical_angle = target_angle - MotorParams.SERVO_ZERO_OFFSET
        print(f"[HWCtrl-Servo] Physical angle: {physical_angle:.1f}°, Pulse: {pulse_width}μs")

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
        print(f"[HWCtrl] Combined controller error: {e}")
        import traceback
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
        print(f"[HWCtrl] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    print("[HWCtrl] Hardware controller process stopped.")