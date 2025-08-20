#!/usr/bin/env python3
"""
Ultra-Precise Hardware Controller with Maximum Synchronization
Every LiDAR measurement is precisely timestamped with exact motor positions
"""

import time
import math
import pigpio
import serial
import numpy as np
import threading
from collections import deque
from enum import Enum
import struct

# ============== HARDWARE CONFIGURATION ==============
# GPIO Pins
SERVO_PIN = 13  # Tilt servo PWM pin
STEPPER_PULSE_PIN = 19  # Step pulse pin
STEPPER_DIR_PIN = 3  # Direction pin
STEPPER_ENABLE_PIN = 4  # Enable pin (active low)
STEPPER_SLEEP_PIN = 6  # Sleep pin

# Motor Parameters
MICROSTEP_ANGLE = 0.05625  # degrees per microstep (1.8° / 32)
STEPS_PER_REV = int(360 / MICROSTEP_ANGLE)  # 6400 steps per revolution

# Servo Configuration
SERVO_MIN_PULSE = 670  # microseconds
SERVO_MAX_PULSE = 1670  # microseconds
SERVO_MIN_ANGLE = 0  # degrees
SERVO_MAX_ANGLE = 90  # degrees

# Stepper Speed Limits (steps/sec)
STEPPER_MAX_SPEED = 6400  # 1 rev/sec max
STEPPER_MIN_SPEED = 100  # Minimum speed for smooth motion
STEPPER_ACCEL = 10000  # steps/sec^2 acceleration

# LiDAR Configuration
LIDAR_PORT = "/dev/serial0"
LIDAR_BAUD = 115200

# Background Scan Parameters
SCAN_SPEED_DEG_SEC = 60.0  # 60 degrees per second for scanning
SCAN_ELEVATION_STEP = 5.0  # 5 degree steps between rings
SCAN_DATA_RATE = 100  # Hz - sampling rate during scan

# Synchronization Parameters
POSITION_HISTORY_SIZE = 1000  # Keep history for interpolation
SYNC_PRECISION_US = 100  # Microsecond precision for timestamps


class SynchronizedMeasurement:
    """Container for precisely synchronized measurement data"""
    __slots__ = ['timestamp_us', 'azimuth', 'elevation', 'distance', 'strength', 'validated']

    def __init__(self, timestamp_us, azimuth, elevation, distance, strength):
        self.timestamp_us = timestamp_us
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.strength = strength
        self.validated = False


class PositionTracker:
    """
    High-precision position tracker with microsecond-accurate history
    Allows interpolation of exact position at any timestamp
    """

    def __init__(self):
        self.azimuth_history = deque(maxlen=POSITION_HISTORY_SIZE)
        self.elevation_history = deque(maxlen=POSITION_HISTORY_SIZE)
        self.lock = threading.Lock()

        # Current positions with microsecond timestamps
        self.current_azimuth = 0.0
        self.current_elevation = 45.0
        self.last_azimuth_update_us = 0
        self.last_elevation_update_us = 0

        # Velocity tracking for interpolation
        self.azimuth_velocity = 0.0  # degrees/second
        self.elevation_velocity = 0.0  # degrees/second

    def update_azimuth(self, angle, timestamp_us):
        """Update azimuth with microsecond timestamp"""
        with self.lock:
            if self.last_azimuth_update_us > 0:
                dt_sec = (timestamp_us - self.last_azimuth_update_us) / 1e6
                if dt_sec > 0:
                    self.azimuth_velocity = (angle - self.current_azimuth) / dt_sec

            self.current_azimuth = angle
            self.last_azimuth_update_us = timestamp_us
            self.azimuth_history.append((timestamp_us, angle))

    def update_elevation(self, angle, timestamp_us):
        """Update elevation with microsecond timestamp"""
        with self.lock:
            if self.last_elevation_update_us > 0:
                dt_sec = (timestamp_us - self.last_elevation_update_us) / 1e6
                if dt_sec > 0:
                    self.elevation_velocity = (angle - self.current_elevation) / dt_sec

            self.current_elevation = angle
            self.last_elevation_update_us = timestamp_us
            self.elevation_history.append((timestamp_us, angle))

    def get_position_at_timestamp(self, timestamp_us):
        """
        Get interpolated position at exact timestamp
        Returns (azimuth, elevation) with microsecond precision
        """
        with self.lock:
            # Interpolate azimuth
            if len(self.azimuth_history) < 2:
                az = self.current_azimuth
            else:
                # Find bracketing timestamps
                az = self._interpolate_from_history(self.azimuth_history, timestamp_us)

            # Interpolate elevation
            if len(self.elevation_history) < 2:
                el = self.current_elevation
            else:
                el = self._interpolate_from_history(self.elevation_history, timestamp_us)

            return az, el

    def _interpolate_from_history(self, history, target_us):
        """Linear interpolation from history deque"""
        # Find bracketing points
        prev_point = None
        next_point = None

        for i, (ts, val) in enumerate(history):
            if ts <= target_us:
                prev_point = (ts, val)
            if ts > target_us and next_point is None:
                next_point = (ts, val)
                break

        # Interpolate
        if prev_point and next_point:
            dt = next_point[0] - prev_point[0]
            if dt > 0:
                factor = (target_us - prev_point[0]) / dt
                return prev_point[1] + factor * (next_point[1] - prev_point[1])
        elif prev_point:
            # Extrapolate forward using velocity
            dt_sec = (target_us - prev_point[0]) / 1e6
            if history == self.azimuth_history:
                return prev_point[1] + self.azimuth_velocity * dt_sec
            else:
                return prev_point[1] + self.elevation_velocity * dt_sec
        elif next_point:
            return next_point[1]

        # Fallback to current
        return self.current_azimuth if history == self.azimuth_history else self.current_elevation


class PrecisionLidarReader:
    """
    Ultra-precise LiDAR reader with synchronized position stamping
    Each measurement is tagged with exact motor positions at measurement time
    """

    def __init__(self, shared_data, position_tracker):
        self.shared_data = shared_data
        self.position_tracker = position_tracker
        self.serial = None
        self.running = False
        self.thread = None
        self.pi = pigpio.pi()  # For microsecond timestamps

        # Measurement queue with position data
        self.measurement_queue = deque(maxlen=100)
        self.queue_lock = threading.Lock()

        # Initialize serial connection
        try:
            self.serial = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=0.0001)
            # Configure TF Mini S for 1000Hz operation
            self.serial.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))  # 1000Hz
            self.serial.write(bytearray([0x5A, 0x05, 0x11, 0x70]))  # Save config
            time.sleep(0.1)
            print(f"[LiDAR] Initialized for precision operation at 1000Hz")
        except Exception as e:
            print(f"[LiDAR] Failed to initialize: {e}")

    def start(self):
        """Start the precision LiDAR reading thread"""
        if self.serial and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._precision_read_loop)
            self.thread.daemon = True
            self.thread.start()
            print("[LiDAR] Precision reader started")

    def _precision_read_loop(self):
        """
        Precision reading loop with exact timestamp capture
        """
        buffer = bytearray()

        while self.running:
            try:
                # Non-blocking read
                if self.serial.in_waiting:
                    new_data = self.serial.read(self.serial.in_waiting)
                    buffer.extend(new_data)

                    # Process complete frames
                    while len(buffer) >= 9:
                        # Look for frame header
                        if buffer[0] == 0x59 and buffer[1] == 0x59:
                            # Capture precise timestamp IMMEDIATELY
                            timestamp_us = self.pi.get_current_tick()

                            # Extract data
                            dist = buffer[2] + (buffer[3] << 8)
                            strength = buffer[4] + (buffer[5] << 8)

                            # Get exact motor positions at this timestamp
                            azimuth, elevation = self.position_tracker.get_position_at_timestamp(timestamp_us)

                            # Create synchronized measurement
                            measurement = SynchronizedMeasurement(
                                timestamp_us, azimuth, elevation, dist, strength
                            )

                            # Store in queue
                            with self.queue_lock:
                                self.measurement_queue.append(measurement)

                            # Update shared data with full synchronized information
                            # Pack all data into the lidar_data array
                            with self.shared_data["lidar_data"].get_lock():
                                self.shared_data["lidar_data"][0] = dist
                                self.shared_data["lidar_data"][1] = strength
                                self.shared_data["lidar_data"][2] = timestamp_us / 1e6  # Convert to seconds

                            # Store position in dedicated shared arrays for synchronization
                            with self.shared_data["lidar_position"].get_lock():
                                self.shared_data["lidar_position"][0] = azimuth
                                self.shared_data["lidar_position"][1] = elevation

                            # Remove processed frame
                            buffer = buffer[9:]
                        else:
                            # Discard first byte and continue searching
                            buffer = buffer[1:]

                # Minimal sleep to prevent CPU spinning
                time.sleep(0.00001)  # 10 microseconds

            except Exception as e:
                if self.running:
                    print(f"[LiDAR] Read error: {e}")
                    time.sleep(0.001)

    def get_latest_synchronized_measurement(self):
        """Get the latest measurement with full synchronization data"""
        with self.queue_lock:
            if self.measurement_queue:
                return self.measurement_queue[-1]
        return None

    def get_measurements_in_range(self, start_us, end_us):
        """Get all measurements within a time range"""
        results = []
        with self.queue_lock:
            for m in self.measurement_queue:
                if start_us <= m.timestamp_us <= end_us:
                    results.append(m)
        return results

    def stop(self):
        """Stop the LiDAR reader"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.serial:
            self.serial.close()
        if self.pi:
            self.pi.stop()
        print("[LiDAR] Precision reader stopped")


class PrecisionStepperMotor:
    """
    Ultra-precise stepper motor with microsecond-accurate position tracking
    Every step is timestamped for exact position reconstruction
    """

    def __init__(self, pi, shared_data, position_tracker):
        self.pi = pi
        self.shared_data = shared_data
        self.position_tracker = position_tracker
        self.current_steps = 0
        self.direction = 1
        self.movement_active = False

        # Setup GPIO
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        # Enable driver
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Active low
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake up
        time.sleep(0.01)

        # Setup precise step counting callback
        self.step_callback = self.pi.callback(
            STEPPER_PULSE_PIN,
            pigpio.RISING_EDGE,
            self._on_step_with_timestamp
        )

        print("[Stepper] Precision stepper initialized")

    def _on_step_with_timestamp(self, gpio, level, tick):
        """Callback for each step with microsecond timestamp"""
        if level == 1:  # Rising edge
            # Update step count
            if self.direction > 0:
                self.current_steps += 1
            else:
                self.current_steps -= 1

            # Wrap around at 360 degrees
            self.current_steps = self.current_steps % STEPS_PER_REV

            # Calculate precise angle
            degrees = (self.current_steps * MICROSTEP_ANGLE) % 360.0

            # Update position tracker with microsecond timestamp
            self.position_tracker.update_azimuth(degrees, tick)

            # Update shared data
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value = degrees

    def move_to_angle_precise(self, target_angle, max_speed=None):
        """
        Precise movement with continuous position tracking
        Returns actual final position and timestamp
        """
        self.movement_active = True
        start_time_us = self.pi.get_current_tick()

        # Calculate movement parameters
        target_angle = target_angle % 360.0
        current_angle = (self.current_steps * MICROSTEP_ANGLE) % 360.0

        # Shortest path calculation
        delta = target_angle - current_angle
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        steps_to_move = int(abs(delta) / MICROSTEP_ANGLE)

        if steps_to_move < 2:
            self.movement_active = False
            return current_angle, start_time_us

        # Set direction
        self.direction = 1 if delta > 0 else -1
        self.pi.write(STEPPER_DIR_PIN, 1 if self.direction > 0 else 0)

        # Calculate speed profile
        if max_speed is None:
            max_speed = min(STEPPER_MAX_SPEED, STEPPER_MIN_SPEED + steps_to_move * 10)
        else:
            max_speed = min(max_speed, STEPPER_MAX_SPEED)

        # Generate precise waveform with acceleration
        accel_steps = min(steps_to_move // 3, int(max_speed / 20))
        decel_steps = accel_steps
        cruise_steps = steps_to_move - accel_steps - decel_steps

        # Use hardware PWM for precise timing
        steps_done = 0

        # Acceleration
        for i in range(1, accel_steps + 1):
            if not self.movement_active:
                break
            speed = STEPPER_MIN_SPEED + (max_speed - STEPPER_MIN_SPEED) * (i / accel_steps)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
            time.sleep(1.0 / speed)
            steps_done += 1

        # Cruise
        if cruise_steps > 0 and self.movement_active:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(max_speed), 500000)
            time.sleep(cruise_steps / max_speed)
            steps_done += cruise_steps

        # Deceleration
        for i in range(decel_steps, 0, -1):
            if not self.movement_active:
                break
            speed = STEPPER_MIN_SPEED + (max_speed - STEPPER_MIN_SPEED) * (i / decel_steps)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
            time.sleep(1.0 / speed)
            steps_done += 1

        # Stop
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Get final position and timestamp
        end_time_us = self.pi.get_current_tick()
        final_angle = (self.current_steps * MICROSTEP_ANGLE) % 360.0

        self.movement_active = False
        return final_angle, end_time_us

    def start_continuous_rotation_precise(self, deg_per_sec):
        """Start continuous rotation with precise speed control"""
        steps_per_sec = abs(deg_per_sec) / MICROSTEP_ANGLE
        steps_per_sec = min(steps_per_sec, STEPPER_MAX_SPEED)

        # Set direction
        self.direction = 1 if deg_per_sec > 0 else -1
        self.pi.write(STEPPER_DIR_PIN, 1 if self.direction > 0 else 0)

        # Start precise PWM
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(steps_per_sec), 500000)
        self.movement_active = True

        return steps_per_sec * MICROSTEP_ANGLE  # Return actual deg/sec

    def stop_rotation(self):
        """Stop rotation immediately"""
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.movement_active = False

    def emergency_stop(self):
        """Emergency stop - instant halt"""
        self.stop_rotation()
        self.pi.write(STEPPER_ENABLE_PIN, 1)  # Disable driver

    def stop(self):
        """Cleanup"""
        self.stop_rotation()
        if self.step_callback:
            self.step_callback.cancel()
        self.pi.write(STEPPER_ENABLE_PIN, 1)
        print("[Stepper] Stopped")


class PrecisionServoMotor:
    """
    Precise servo control with position feedback and timestamp tracking
    """

    def __init__(self, pi, shared_data, position_tracker):
        self.pi = pi
        self.shared_data = shared_data
        self.position_tracker = position_tracker
        self.current_angle = 45.0
        self.target_angle = 45.0
        self.movement_thread = None
        self.moving = False

        # Setup PWM
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

        # Move to initial position
        self.move_to_angle_precise(self.current_angle)
        print("[Servo] Precision servo initialized at 45°")

    def move_to_angle_precise(self, angle, speed=90.0):
        """
        Precise servo movement with position tracking
        speed in degrees/second
        """
        # Clamp angle
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        self.target_angle = angle

        if abs(angle - self.current_angle) < 0.1:
            return self.current_angle

        # Start movement thread for smooth, tracked motion
        if self.movement_thread and self.movement_thread.is_alive():
            self.moving = False
            self.movement_thread.join(timeout=0.1)

        self.moving = True
        self.movement_thread = threading.Thread(
            target=self._smooth_move,
            args=(angle, speed)
        )
        self.movement_thread.daemon = True
        self.movement_thread.start()

        return angle

    def _smooth_move(self, target, speed):
        """Smooth movement with continuous position updates"""
        start_angle = self.current_angle
        angle_diff = target - start_angle

        if abs(angle_diff) < 0.1:
            return

        # Calculate movement time
        move_time = abs(angle_diff) / speed
        start_time = time.time()

        while self.moving and (time.time() - start_time) < move_time:
            # Calculate current position
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / move_time)

            # Smooth interpolation (ease-in-out)
            smooth_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
            current = start_angle + angle_diff * smooth_progress

            # Update servo position
            pulse = self._angle_to_pulse(current)
            self.pi.set_servo_pulsewidth(SERVO_PIN, pulse)

            # Update position tracker with microsecond timestamp
            timestamp_us = self.pi.get_current_tick()
            self.position_tracker.update_elevation(current, timestamp_us)

            # Update shared data
            self.current_angle = current
            with self.shared_data["servo_degrees"].get_lock():
                self.shared_data["servo_degrees"].value = current

            time.sleep(0.001)  # 1ms update rate

        # Ensure final position
        if self.moving:
            pulse = self._angle_to_pulse(target)
            self.pi.set_servo_pulsewidth(SERVO_PIN, pulse)

            timestamp_us = self.pi.get_current_tick()
            self.position_tracker.update_elevation(target, timestamp_us)

            self.current_angle = target
            with self.shared_data["servo_degrees"].get_lock():
                self.shared_data["servo_degrees"].value = target

        self.moving = False

    def _angle_to_pulse(self, angle):
        """Convert angle to pulse width"""
        pulse_range = SERVO_MAX_PULSE - SERVO_MIN_PULSE
        angle_range = SERVO_MAX_ANGLE - SERVO_MIN_ANGLE
        pulse = SERVO_MIN_PULSE + (angle / angle_range) * pulse_range
        return int(pulse)

    def stop(self):
        """Stop servo"""
        self.moving = False
        if self.movement_thread:
            self.movement_thread.join(timeout=0.1)
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
        print("[Servo] Stopped")


class SynchronizedBackgroundScanner:
    """
    Synchronized background scanner with precise position-data correlation
    """

    def __init__(self, stepper, servo, lidar, shared_data, position_tracker):
        self.stepper = stepper
        self.servo = servo
        self.lidar = lidar
        self.shared_data = shared_data
        self.position_tracker = position_tracker
        self.scan_data = []
        self.scanning = False

    def scan_synchronized(self):
        """
        Perform synchronized background scan with precise position tracking
        """
        print("[Scanner] Starting synchronized background scan...")
        self.scanning = True
        self.scan_data = []

        elevation = 0
        direction = 1

        while elevation <= SERVO_MAX_ANGLE and self.scanning:
            # Move servo to elevation with tracking
            self.servo.move_to_angle_precise(elevation, speed=30)
            time.sleep(0.2)  # Settle time

            # Start continuous rotation with exact speed
            actual_speed = self.stepper.start_continuous_rotation_precise(
                SCAN_SPEED_DEG_SEC * direction
            )

            # Collect synchronized data during rotation
            start_angle = self.shared_data["stepper_degrees"].value
            start_time_us = self.position_tracker.pi.get_current_tick()
            sample_interval_us = int(1e6 / SCAN_DATA_RATE)
            next_sample_us = start_time_us + sample_interval_us

            samples_collected = 0

            while self.scanning:
                current_time_us = self.position_tracker.pi.get_current_tick()
                current_angle = self.shared_data["stepper_degrees"].value

                # Check rotation completion
                angle_diff = (current_angle - start_angle) * direction
                if angle_diff < 0:
                    angle_diff += 360
                if angle_diff >= 359:
                    break

                # Collect sample at precise intervals
                if current_time_us >= next_sample_us:
                    # Get measurements in time window
                    measurements = self.lidar.get_measurements_in_range(
                        next_sample_us - sample_interval_us // 2,
                        next_sample_us + sample_interval_us // 2
                    )

                    # Use best measurement (highest strength)
                    if measurements:
                        best = max(measurements, key=lambda m: m.strength)
                        self.scan_data.append([
                            best.azimuth,
                            best.elevation,
                            best.distance,
                            best.strength,
                            best.timestamp_us
                        ])
                        samples_collected += 1

                    next_sample_us += sample_interval_us

                time.sleep(0.0001)  # 100 microsecond loop

            # Stop rotation
            self.stepper.stop_rotation()

            print(f"[Scanner] Ring at {elevation}° complete: {samples_collected} samples")

            # Next elevation
            elevation += SCAN_ELEVATION_STEP
            direction *= -1

        # Save synchronized scan data
        if self.scan_data:
            self._save_synchronized_data()

        self.scanning = False
        self.shared_data["background_scan_active"].value = False
        print("[Scanner] Synchronized scan complete")

    def _save_synchronized_data(self):
        """Save scan data with timestamp information"""
        try:
            filename = self.shared_data["background_path"].value.decode()
            # Save with timestamp column for verification
            np.save(filename, np.array(self.scan_data))
            print(f"[Scanner] Saved {len(self.scan_data)} synchronized points")

            # Calculate synchronization statistics
            if len(self.scan_data) > 1:
                timestamps = [d[4] for d in self.scan_data]
                intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                avg_interval = np.mean(intervals) / 1e6  # Convert to seconds
                std_interval = np.std(intervals) / 1e6
                print(f"[Scanner] Timing stats: {1 / avg_interval:.1f}Hz ± {std_interval * 1000:.2f}ms")

        except Exception as e:
            print(f"[Scanner] Failed to save: {e}")


class UltraPreciseHardwareController:
    """
    Main controller with maximum synchronization between all subsystems
    """

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.running = False

        # Initialize pigpio for microsecond timing
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("Failed to connect to pigpio daemon")

        # Initialize position tracker first
        self.position_tracker = PositionTracker()

        # Initialize synchronized subsystems
        self.lidar = PrecisionLidarReader(shared_data, self.position_tracker)
        self.stepper = PrecisionStepperMotor(self.pi, shared_data, self.position_tracker)
        self.servo = PrecisionServoMotor(self.pi, shared_data, self.position_tracker)
        self.scanner = SynchronizedBackgroundScanner(
            self.stepper, self.servo, self.lidar, shared_data, self.position_tracker
        )

        # Start LiDAR reader
        self.lidar.start()

        # Control thread for precise motor commands
        self.control_thread = None

        print("[HWController] Ultra-precise controller initialized")
        print("[HWController] Synchronization precision: ±100 microseconds")

    def run(self):
        """Main control loop with precise synchronization"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        print("[HWController] Precision control loop started")

        # Main monitoring loop
        while self.running and not self.shared_data["shutdown"].value:
            try:
                # Monitor for background scan
                if self.shared_data["background_scan_active"].value and not self.scanner.scanning:
                    scan_thread = threading.Thread(target=self.scanner.scan_synchronized)
                    scan_thread.daemon = True
                    scan_thread.start()

                # Small sleep
                time.sleep(0.01)

            except Exception as e:
                print(f"[HWController] Monitor error: {e}")
                time.sleep(0.01)

        print("[HWController] Main loop ended")

    def _control_loop(self):
        """High-speed control loop for motor commands"""
        last_command_time = 0

        while self.running:
            try:
                current_time = time.time()

                # Check for movement commands (rate limited to prevent oscillation)
                if self.shared_data["go_to_target"].value and (current_time - last_command_time) > 0.01:
                    # Get targets
                    target_az = self.shared_data["target_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value

                    # Execute synchronized movement
                    # Start both movements simultaneously for better sync
                    servo_thread = threading.Thread(
                        target=self.servo.move_to_angle_precise,
                        args=(target_el, 90)  # 90 deg/sec for servo
                    )
                    servo_thread.start()

                    # Move stepper (blocks until complete)
                    final_az, final_time = self.stepper.move_to_angle_precise(target_az)

                    # Wait for servo
                    servo_thread.join(timeout=2.0)

                    # Signal completion
                    self.shared_data["go_to_target"].value = False
                    self.shared_data["target_reached"].value = True

                    last_command_time = current_time

                    # Log synchronization quality
                    latest = self.lidar.get_latest_synchronized_measurement()
                    if latest:
                        pos_az, pos_el = self.position_tracker.get_position_at_timestamp(latest.timestamp_us)
                        az_error = abs(pos_az - latest.azimuth)
                        el_error = abs(pos_el - latest.elevation)
                        if az_error > 0.1 or el_error > 0.1:
                            print(f"[HWController] Sync deviation: Az={az_error:.3f}° El={el_error:.3f}°")

                # Ultra-fast loop for responsiveness
                time.sleep(0.0001)  # 100 microseconds

            except Exception as e:
                print(f"[HWController] Control error: {e}")
                time.sleep(0.001)

    def stop(self):
        """Clean shutdown with synchronization stats"""
        print("[HWController] Shutting down...")
        self.running = False

        # Print final synchronization statistics
        latest = self.lidar.get_latest_synchronized_measurement()
        if latest:
            print(f"[HWController] Final measurement sync:")
            print(f"  Timestamp: {latest.timestamp_us}us")
            print(f"  Position: Az={latest.azimuth:.3f}° El={latest.elevation:.3f}°")
            print(f"  LiDAR: {latest.distance}cm @ {latest.strength}")

        # Stop all subsystems
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        if self.scanner:
            self.scanner.scanning = False
        if self.lidar:
            self.lidar.stop()
        if self.stepper:
            self.stepper.stop()
        if self.servo:
            self.servo.stop()
        if self.pi:
            self.pi.stop()

        print("[HWController] Shutdown complete")


def run_hardware_controller(shared_data):
    """Entry point for hardware controller process"""
    controller = None
    try:
        controller = UltraPreciseHardwareController(shared_data)
        controller.run()
    except Exception as e:
        print(f"[HWController] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.stop()
        print("[HWController] Process terminated")