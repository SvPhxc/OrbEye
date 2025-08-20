#!/usr/bin/env python3
"""
Fast and Responsive Hardware Controller
Minimal overhead, direct control, no excessive printing
"""

import time
import math
import pigpio
import serial
import numpy as np
import threading
from collections import deque

# ============== HARDWARE CONFIGURATION ==============
# GPIO Pins
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6

# Motor Parameters
MICROSTEP_ANGLE = 0.05625  # degrees per microstep
STEPS_PER_REV = int(360 / MICROSTEP_ANGLE)

# Servo Configuration
SERVO_MIN_PULSE = 670
SERVO_MAX_PULSE = 1670
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 90

# Speed Limits
STEPPER_MAX_SPEED = 4000  # Reduced for better control
STEPPER_MIN_SPEED = 200
STEPPER_ACCELERATION = 8000  # steps/sec^2

# LiDAR Configuration
LIDAR_PORT = "/dev/serial0"
LIDAR_BAUD = 115200

# Scan Parameters
SCAN_SPEED = 45.0  # degrees/second
SCAN_ELEVATION_STEP = 5.0


class FastLiDAR:
    """Minimal overhead LiDAR reader"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.serial = None
        self.running = False
        self.thread = None

        try:
            self.serial = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=0.001)
            # Configure for 1000Hz
            self.serial.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            self.serial.write(bytearray([0x5A, 0x05, 0x11, 0x70]))
            time.sleep(0.05)
        except Exception as e:
            print(f"[LiDAR] Init failed: {e}")

    def start(self):
        if self.serial and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._read_loop)
            self.thread.daemon = True
            self.thread.start()

    def _read_loop(self):
        buffer = bytearray()

        while self.running:
            try:
                if self.serial.in_waiting:
                    buffer.extend(self.serial.read(self.serial.in_waiting))

                    while len(buffer) >= 9:
                        if buffer[0] == 0x59 and buffer[1] == 0x59:
                            dist = buffer[2] + (buffer[3] << 8)
                            strength = buffer[4] + (buffer[5] << 8)

                            # Get current positions immediately
                            az = self.shared_data["stepper_degrees"].value
                            el = self.shared_data["servo_degrees"].value

                            # Direct write to shared memory
                            with self.shared_data["lidar_data"].get_lock():
                                self.shared_data["lidar_data"][:] = [dist, strength, time.time()]

                            with self.shared_data["lidar_position"].get_lock():
                                self.shared_data["lidar_position"][:] = [az, el]

                            buffer = buffer[9:]
                        else:
                            buffer = buffer[1:]

                time.sleep(0.0001)

            except:
                if self.running:
                    time.sleep(0.001)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        if self.serial:
            self.serial.close()


class FastStepper:
    """Direct stepper control with minimal overhead"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.current_steps = 0
        self.target_steps = 0
        self.is_moving = False

        # GPIO setup
        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        self.pi.write(STEPPER_ENABLE_PIN, 0)
        self.pi.write(STEPPER_SLEEP_PIN, 1)
        time.sleep(0.001)

        # Step counter callback
        self.callback = self.pi.callback(STEPPER_PULSE_PIN, pigpio.RISING_EDGE, self._count_step)

    def _count_step(self, gpio, level, tick):
        """Fast step counting"""
        direction = self.pi.read(STEPPER_DIR_PIN)
        if direction:
            self.current_steps += 1
        else:
            self.current_steps -= 1

        self.current_steps = self.current_steps % STEPS_PER_REV
        degrees = self.current_steps * MICROSTEP_ANGLE

        with self.shared_data["stepper_degrees"].get_lock():
            self.shared_data["stepper_degrees"].value = degrees

    def move_to(self, target_angle):
        """Fast movement with minimal overhead"""
        if self.is_moving:
            return False

        self.is_moving = True

        # Calculate shortest path
        target_angle = target_angle % 360
        current_angle = (self.current_steps * MICROSTEP_ANGLE) % 360

        delta = target_angle - current_angle
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        steps = int(abs(delta) / MICROSTEP_ANGLE)

        if steps < 2:
            self.is_moving = False
            return True

        # Set direction
        direction = 1 if delta > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        # Simple speed profile - fewer calculations
        if steps < 100:
            max_speed = STEPPER_MIN_SPEED + steps * 10
        else:
            max_speed = min(STEPPER_MAX_SPEED, steps * 20)

        # Quick acceleration
        accel_steps = min(steps // 4, 50)

        # Execute movement
        if accel_steps > 0:
            for i in range(1, accel_steps + 1):
                speed = STEPPER_MIN_SPEED + (max_speed - STEPPER_MIN_SPEED) * i / accel_steps
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
                time.sleep(1.0 / speed)

        # Cruise
        cruise_steps = steps - 2 * accel_steps
        if cruise_steps > 0:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(max_speed), 500000)
            time.sleep(cruise_steps / max_speed)

        # Quick deceleration
        if accel_steps > 0:
            for i in range(accel_steps, 0, -1):
                speed = STEPPER_MIN_SPEED + (max_speed - STEPPER_MIN_SPEED) * i / accel_steps
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(speed), 500000)
                time.sleep(1.0 / speed)

        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.is_moving = False
        return True

    def continuous_rotation(self, deg_per_sec):
        """Start continuous rotation"""
        steps_per_sec = min(abs(deg_per_sec) / MICROSTEP_ANGLE, STEPPER_MAX_SPEED)
        direction = 1 if deg_per_sec > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(steps_per_sec), 500000)

    def stop_rotation(self):
        """Stop rotation"""
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

    def cleanup(self):
        self.stop_rotation()
        if self.callback:
            self.callback.cancel()
        self.pi.write(STEPPER_ENABLE_PIN, 1)


class FastServo:
    """Direct servo control"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.current_angle = 45.0

        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        self.set_angle(45.0)

    def set_angle(self, angle):
        """Direct angle setting"""
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))

        pulse_range = SERVO_MAX_PULSE - SERVO_MIN_PULSE
        angle_range = SERVO_MAX_ANGLE - SERVO_MIN_ANGLE
        pulse = SERVO_MIN_PULSE + (angle / angle_range) * pulse_range

        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse))
        self.current_angle = angle

        with self.shared_data["servo_degrees"].get_lock():
            self.shared_data["servo_degrees"].value = angle

    def cleanup(self):
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)


class FastScanner:
    """Efficient background scanner"""

    def __init__(self, stepper, servo, shared_data):
        self.stepper = stepper
        self.servo = servo
        self.shared_data = shared_data
        self.data = []
        self.scanning = False

    def scan(self):
        """Fast scanning with minimal overhead"""
        self.scanning = True
        self.data = []

        for elevation in range(0, 91, int(SCAN_ELEVATION_STEP)):
            if not self.scanning or self.shared_data["shutdown"].value:
                break

            # Move servo
            self.servo.set_angle(elevation)
            time.sleep(0.1)

            # Continuous rotation scan
            self.stepper.continuous_rotation(SCAN_SPEED)

            start_angle = self.shared_data["stepper_degrees"].value
            last_sample = time.time()

            while self.scanning:
                current_angle = self.shared_data["stepper_degrees"].value

                # Check if rotation complete
                if abs(current_angle - start_angle) >= 358:
                    break

                # Sample at 50Hz
                if time.time() - last_sample > 0.02:
                    with self.shared_data["lidar_data"].get_lock():
                        dist, strength, _ = self.shared_data["lidar_data"][:]

                    if dist > 0:
                        self.data.append([current_angle, elevation, dist, strength])

                    last_sample = time.time()

                time.sleep(0.001)

            self.stepper.stop_rotation()

            # Reverse direction for next scan
            SCAN_SPEED *= -1

        # Save data
        if self.data:
            try:
                filename = self.shared_data["background_path"].value.decode()
                np.save(filename, np.array(self.data))
            except:
                pass

        self.scanning = False
        self.shared_data["background_scan_active"].value = False

    def stop(self):
        self.scanning = False
        self.stepper.stop_rotation()


class HardwareController:
    """Main controller - lean and fast"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.running = False

        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("pigpio not connected")

        # Initialize components
        self.lidar = FastLiDAR(shared_data)
        self.stepper = FastStepper(self.pi, shared_data)
        self.servo = FastServo(self.pi, shared_data)
        self.scanner = FastScanner(self.stepper, self.servo, shared_data)

        self.lidar.start()

        print("[HW] Ready")

    def run(self):
        """Main loop - minimal overhead"""
        self.running = True
        last_move_time = 0
        scan_thread = None

        while self.running and not self.shared_data["shutdown"].value:
            current_time = time.time()

            # Background scan
            if self.shared_data["background_scan_active"].value and not self.scanner.scanning:
                if scan_thread is None or not scan_thread.is_alive():
                    scan_thread = threading.Thread(target=self.scanner.scan)
                    scan_thread.daemon = True
                    scan_thread.start()

            # Movement commands - rate limited to prevent oscillation
            if self.shared_data["go_to_target"].value and (current_time - last_move_time) > 0.02:
                if not self.scanner.scanning and not self.stepper.is_moving:
                    target_az = self.shared_data["target_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value

                    # Move servo immediately (fast)
                    self.servo.set_angle(target_el)

                    # Move stepper
                    if self.stepper.move_to(target_az):
                        self.shared_data["go_to_target"].value = False
                        self.shared_data["target_reached"].value = True
                        last_move_time = current_time

            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.scanner.stop()
        self.lidar.stop()
        self.stepper.cleanup()
        self.servo.cleanup()
        if self.pi:
            self.pi.stop()


def run_hardware_controller(shared_data):
    """Entry point"""
    controller = None
    try:
        controller = HardwareController(shared_data)
        controller.run()
    except Exception as e:
        print(f"[HW] Error: {e}")
    finally:
        if controller:
            controller.stop()