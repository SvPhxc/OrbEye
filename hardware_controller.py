#!/usr/bin/env python3
"""
Integrated Hardware Controller for LiDAR Scanner System
Final version with proper inter-process communication and state management
Coordinates with tracker process for seamless operation
"""

import time
import math
import pigpio
import serial
import queue
import numpy as np
from multiprocessing import Process, Manager, Value, Array, Lock
import threading
from collections import deque
from enum import Enum
import signal
import sys
import json

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Hardware pins
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625  # degrees per step


# System states
class SystemState(Enum):
    IDLE = 0
    MOVING = 1
    SCANNING = 2
    TRACKER_MOVE = 3  # Priority movement for tracker
    ERROR = 4
    SHUTDOWN = 5
    PAUSED = 6  # Scan paused for tracker


# Movement priorities
class Priority(Enum):
    NORMAL = 0
    HIGH = 1  # Tracker requests
    CRITICAL = 2  # Emergency


# Motor parameters
class MotorParams:
    STEPPER_MAX_SPEED = 8000
    STEPPER_MIN_SPEED = 100
    STEPPER_CRUISE_SPEED = 6400
    ACCEL_STEPS = 150

    SERVO_MIN_PULSE = 670
    SERVO_MAX_PULSE = 1670
    SERVO_MIN_ANGLE = 0
    SERVO_MAX_ANGLE = 90
    SERVO_ZERO_OFFSET = 0

    PAN_MIN_ANGLE = 0.0
    PAN_MAX_ANGLE = 360.0

    MOVEMENT_SETTLE_TIME = 0.05
    POSITION_TOLERANCE = 0.5  # degrees


# Scan profiles
class ScanProfiles:
    FAST = {
        "azimuth_speed": 90.0,
        "elevation_step": 5.0,
        "data_rate": 30,
        "servo_move_time": 0.5,
        "servo_settle_time": 0.1,
    }

    NORMAL = {
        "azimuth_speed": 60.0,
        "elevation_step": 3.0,
        "data_rate": 40,
        "servo_move_time": 0.8,
        "servo_settle_time": 0.2,
    }

    HIGH_QUALITY = {
        "azimuth_speed": 30.0,
        "elevation_step": 2.0,
        "data_rate": 50,
        "servo_move_time": 1.0,
        "servo_settle_time": 0.3,
    }


CURRENT_SCAN_PROFILE = ScanProfiles.NORMAL
SCAN_AZIMUTH_SPEED = CURRENT_SCAN_PROFILE["azimuth_speed"]
SCAN_ELEVATION_STEP = CURRENT_SCAN_PROFILE["elevation_step"]
SCAN_DATA_RATE = CURRENT_SCAN_PROFILE["data_rate"]
SERVO_MOVE_TIME = CURRENT_SCAN_PROFILE["servo_move_time"]
SERVO_SETTLE_TIME = CURRENT_SCAN_PROFILE["servo_settle_time"]

SCAN_TILT_MAX = 90.0
SCAN_TILT_MIN = 0.0
MAX_SCAN_TIME = 600


# ============================================================================
# LIDAR CONTROLLER
# ============================================================================

class LidarController:
    """Enhanced LiDAR controller with position-tagged data"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.ser = None
        self.lidar_queue = queue.Queue(maxsize=100)
        self.shutdown_event = threading.Event()
        self.lidar_thread = None
        self.serial_connected = False
        self.simulation_mode = False

        self._initialize_serial()

    def _initialize_serial(self):
        """Initialize serial with fallback to simulation"""
        ports = ["/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"]

        for port in ports:
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.1)
                self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
                time.sleep(0.01)
                self.ser.write(bytearray([0x5A, 0x05, 0x11, 0x70]))
                print(f"[LiDAR] Initialized on {port}")
                self.serial_connected = True
                self.lidar_thread = threading.Thread(target=self._lidar_reader_thread)
                self.lidar_thread.daemon = True
                self.lidar_thread.start()
                return
            except Exception as e:
                if self.ser:
                    try: self.ser.close()
                    except: pass
                self.ser = None

        print("[LiDAR] No serial port found, using simulation mode")
        self.simulation_mode = True
        self.lidar_thread = threading.Thread(target=self._simulation_thread)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()

    def _simulation_thread(self):
        """Generate simulated LiDAR data for testing"""
        while not self.shutdown_event.is_set():
            try:
                dist = 150 + np.random.randint(-10, 10)
                strength = 180 + np.random.randint(-20, 20)
                self.lidar_queue.put_nowait((dist, strength, time.time()))
                time.sleep(0.001)
            except queue.Full:
                pass

    def _lidar_reader_thread(self):
        """Read actual LiDAR data"""
        consecutive_errors = 0
        while not self.shutdown_event.is_set():
            try:
                if not self.ser or not self.ser.is_open:
                    time.sleep(0.1)
                    continue
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    dist = frame[0] + (frame[1] << 8)
                    strength = frame[2] + (frame[3] << 8)
                    timestamp = time.time()
                    try:
                        self.lidar_queue.put_nowait((dist, strength, timestamp))
                        consecutive_errors = 0
                    except queue.Full:
                        try:
                            self.lidar_queue.get_nowait()
                            self.lidar_queue.put_nowait((dist, strength, timestamp))
                        except: pass
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    print(f"[LiDAR] Multiple errors, attempting reconnect")
                    self._attempt_reconnect()
                    consecutive_errors = 0

    def _attempt_reconnect(self):
        """Try to reconnect to serial port"""
        if self.ser:
            try: self.ser.close()
            except: pass
        time.sleep(1)
        self._initialize_serial()

    def update_lidar_data(self):
        """Update shared LiDAR data with position tagging"""
        latest = None
        count = 0
        while not self.lidar_queue.empty() and count < 10:
            try:
                latest = self.lidar_queue.get_nowait()
                count += 1
            except queue.Empty:
                break

        if latest:
            dist, strength, timestamp = latest
            current_az = self.shared_data["stepper_degrees"].value
            current_el = self.shared_data["servo_degrees"].value

            with self.shared_data["lidar_lock"]:
                # --- THIS IS THE FIX ---
                # You cannot use slice assignment with a manager.Array.
                # You must assign each element by its index.
                self.shared_data["lidar_data"][0] = dist
                self.shared_data["lidar_data"][1] = strength
                self.shared_data["lidar_data"][2] = timestamp

                self.shared_data["lidar_position"][0] = current_az
                self.shared_data["lidar_position"][1] = current_el

                self.shared_data["lidar_valid"].value = True

            self.shared_data["lidar_reads"].value += 1
            return dist, strength, timestamp

        return None, None, None

    def stop(self):
        """Shutdown LiDAR controller"""
        self.shutdown_event.set()
        if self.lidar_thread:
            self.lidar_thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            try: self.ser.close()
            except: pass
        print("[LiDAR] Stopped")


# ============================================================================
# STEPPER CONTROLLER
# ============================================================================

class StepperController:
    """Thread-safe stepper controller with request handling"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.step_count = 0
        self.current_direction = 1
        self.step_lock = threading.Lock()
        self.movement_active = False

        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        self.pi.write(STEPPER_ENABLE_PIN, 0)
        self.pi.write(STEPPER_SLEEP_PIN, 1)
        time.sleep(0.001)

        self.step_callback = self.pi.callback(
            STEPPER_PULSE_PIN, pigpio.RISING_EDGE, self._step_counter_callback
        )
        print("[Stepper] Initialized")

    def _step_counter_callback(self, gpio, level, tick):
        """Count steps and update position"""
        if level == 1:
            with self.step_lock:
                if self.current_direction:
                    self.step_count += 1
                else:
                    self.step_count -= 1
                steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
                self.step_count = self.step_count % steps_per_rotation
                degrees = self.step_count * MICROSTEP_ANGLE
                self.shared_data["stepper_degrees"].value = degrees

    def move_to_angle(self, target_angle, priority=Priority.NORMAL):
        """Execute movement with priority handling"""
        if self.movement_active:
            if priority.value <= Priority.NORMAL.value:
                return False

        self.movement_active = True
        try:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            self.pi.wave_tx_stop()
            self.pi.wave_clear()

            target_angle = np.clip(target_angle, MotorParams.PAN_MIN_ANGLE, MotorParams.PAN_MAX_ANGLE)
            current_pos_steps = self.step_count
            target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
            error_steps = target_pos_steps - current_pos_steps

            steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
            if abs(error_steps) > (steps_per_rotation / 2):
                if error_steps > 0: error_steps -= steps_per_rotation
                else: error_steps += steps_per_rotation

            total_steps = abs(error_steps)
            if total_steps < 1: return True

            direction = 1 if error_steps > 0 else 0
            with self.step_lock:
                self.current_direction = direction
                self.pi.write(STEPPER_DIR_PIN, direction)

            return self._execute_movement(total_steps)
        finally:
            self.movement_active = False

    def _execute_movement(self, total_steps):
        """Execute movement with waveforms"""
        if total_steps <= MotorParams.ACCEL_STEPS * 2:
            accel_steps = (total_steps + 1) // 2
            decel_steps = total_steps - accel_steps
        else:
            accel_steps = MotorParams.ACCEL_STEPS
            decel_steps = MotorParams.ACCEL_STEPS

        pulses = []
        for i in range(1, accel_steps + 1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / accel_steps)
            delay_us = int(500000 / speed)
            pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us)])

        cruise_steps = total_steps - (accel_steps + decel_steps)
        if cruise_steps > 0:
            delay_us = int(500000 / MotorParams.STEPPER_MAX_SPEED)
            for _ in range(cruise_steps):
                pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us)])

        for i in range(decel_steps, 0, -1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / decel_steps)
            delay_us = int(500000 / speed)
            pulses.extend([pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us)])

        MAX_PULSES = 4000
        start_idx = 0
        while start_idx < len(pulses):
            if self.shared_data["shutdown"].value: return False
            end_idx = min(start_idx + MAX_PULSES, len(pulses))
            chunk = pulses[start_idx:end_idx]
            try:
                self.pi.wave_clear()
                self.pi.wave_add_generic(chunk)
                wave_id = self.pi.wave_create()
                if wave_id >= 0:
                    self.pi.wave_send_once(wave_id)
                    timeout = time.time() + 10.0
                    while self.pi.wave_tx_busy():
                        if time.time() > timeout:
                            self.pi.wave_tx_stop()
                            return False
                        time.sleep(0.01)
                    self.pi.wave_delete(wave_id)
                else: return False
            except Exception as e:
                print(f"[Stepper] Movement error: {e}")
                return False
            start_idx = end_idx
        return True

    def continuous_movement(self, start_az, end_az, speed_dps):
        """Continuous movement for scanning"""
        self.movement_active = True
        try:
            total_distance = abs(end_az - start_az)
            movement_time = total_distance / speed_dps
            steps_per_second = speed_dps / MICROSTEP_ANGLE
            direction = 1 if end_az > start_az else 0
            with self.step_lock:
                self.current_direction = direction
                self.pi.write(STEPPER_DIR_PIN, direction)

            target_freq = min(int(steps_per_second), MotorParams.STEPPER_MAX_SPEED)
            for i in range(1, 11):
                freq = int((target_freq / 10) * i)
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)
                time.sleep(0.002)

            start_time = time.time()
            while (time.time() - start_time) < movement_time:
                if self.shared_data["shutdown"].value: break
                time.sleep(0.001)

            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            return True
        finally:
            self.movement_active = False

    def stop(self):
        """Shutdown stepper"""
        self.running = False
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        if self.step_callback:
            self.step_callback.cancel()
        self.pi.write(STEPPER_ENABLE_PIN, 1)
        print("[Stepper] Stopped")


# ============================================================================
# SERVO CONTROLLER
# ============================================================================

class ServoController:
    """Servo controller with event-based operation"""

    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.shutdown_event = threading.Event()
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        self.move_to_angle(45.0)
        print("[Servo] Initialized")

    def move_to_angle(self, angle):
        """Move servo to angle"""
        angle = np.clip(angle, MotorParams.SERVO_MIN_ANGLE, MotorParams.SERVO_MAX_ANGLE)
        physical_angle = angle - MotorParams.SERVO_ZERO_OFFSET
        physical_angle = np.clip(physical_angle, MotorParams.SERVO_MIN_ANGLE, MotorParams.SERVO_MAX_ANGLE)
        pulse_range = MotorParams.SERVO_MAX_PULSE - MotorParams.SERVO_MIN_PULSE
        angle_range = MotorParams.SERVO_MAX_ANGLE - MotorParams.SERVO_MIN_ANGLE
        pulse_width = MotorParams.SERVO_MIN_PULSE + (physical_angle / angle_range) * pulse_range
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))
        self.shared_data["servo_degrees"].value = angle

    def control_loop(self):
        """Main servo control loop"""
        while not self.shutdown_event.is_set():
            target = self.shared_data["target_elevation"].value
            current = self.shared_data["servo_degrees"].value
            if abs(target - current) > 0.5:
                self.move_to_angle(target)
            self.shutdown_event.wait(0.001)

    def stop(self):
        """Stop servo"""
        self.shutdown_event.set()
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
        print("[Servo] Stopped")


# ============================================================================
# BACKGROUND SCANNER
# ============================================================================

class BackgroundScanner:
    """Pausable background scanner"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.current_elevation = SCAN_TILT_MAX
        self.scan_direction = 1
        self.data_buffer = []
        self.scan_state = None
        self.paused = False

    def start_scan(self, lidar, stepper, servo):
        """Execute scan with pause capability"""
        if not self.shared_data["background_scan_active"].value: return False
        if self.shared_data["background_scan_paused"].value: return True

        self.shared_data["target_elevation"].value = self.current_elevation
        time.sleep(SERVO_MOVE_TIME)

        if self.scan_direction == 1: start_az, end_az = 0, 360
        else: start_az, end_az = 360, 0
        stepper.move_to_angle(start_az)
        time.sleep(0.05)

        movement_active = threading.Event()
        movement_active.set()
        def movement_thread():
            stepper.continuous_movement(start_az, end_az, SCAN_AZIMUTH_SPEED)
            movement_active.clear()
        thread = threading.Thread(target=movement_thread)
        thread.daemon = True
        thread.start()

        self._collect_data(lidar, movement_active)
        thread.join(timeout=15)

        progress = ((SCAN_TILT_MAX - self.current_elevation) / (SCAN_TILT_MAX - SCAN_TILT_MIN)) * 100
        self.shared_data["scan_progress"].value = progress

        self.current_elevation -= SCAN_ELEVATION_STEP
        if self.current_elevation < SCAN_TILT_MIN:
            self._save_data()
            self._reset()
            return False

        self.scan_direction *= -1
        return True

    def pause(self):
        """Pause scan and save state"""
        self.paused = True
        self.scan_state = {'elevation': self.current_elevation, 'direction': self.scan_direction, 'data_count': len(self.data_buffer)}
        print(f"[Scanner] Paused at elevation {self.current_elevation}°")

    def resume(self):
        """Resume from saved state"""
        if self.scan_state:
            self.current_elevation = self.scan_state['elevation']
            self.scan_direction = self.scan_state['direction']
            print(f"[Scanner] Resumed at elevation {self.current_elevation}°")
        self.paused = False

    def _collect_data(self, lidar, movement_active):
        """Collect data during movement"""
        interval = 1.0 / SCAN_DATA_RATE
        next_time = time.time()
        while movement_active.is_set():
            if time.time() >= next_time:
                dist, strength, _ = lidar.update_lidar_data()
                if dist:
                    az = self.shared_data["stepper_degrees"].value
                    el = self.shared_data["servo_degrees"].value
                    self.data_buffer.append([az, el, dist, strength])
                next_time += interval
            time.sleep(0.0005)

    def _save_data(self):
        """Save scan data"""
        if self.data_buffer:
            try:
                # Correctly decode the ctypes string buffer
                path = self.shared_data["background_path"].value.decode('utf-8').rstrip('\x00')
                np.save(path, np.array(self.data_buffer))
                print(f"[Scanner] Saved {len(self.data_buffer)} points to {path}")
            except Exception as e:
                print(f"[Scanner] Save error: {e}")

    def _reset(self):
        """Reset scanner"""
        self.current_elevation = SCAN_TILT_MAX
        self.scan_direction = 1
        self.data_buffer = []
        self.shared_data["background_scan_active"].value = False
        self.shared_data["scan_progress"].value = 0.0
        print("[Scanner] Reset")


# ============================================================================
# MOVEMENT COORDINATOR
# ============================================================================

class MovementCoordinator:
    """Coordinates movements between scanner and tracker"""

    def __init__(self, shared_data, stepper, servo):
        self.shared_data = shared_data
        self.stepper = stepper
        self.servo = servo
        self.current_request_id = 0

    def handle_tracker_request(self):
        """Process high-priority tracker movement"""
        if not self.shared_data["go_to_target"].value: return False

        request_id = self.shared_data["movement_request_id"].value
        if request_id <= self.current_request_id: return False

        target_az = self.shared_data["target_azimuth"].value
        target_el = self.shared_data["target_elevation"].value
        priority = Priority(self.shared_data["movement_priority"].value)
        print(f"[Coordinator] Processing movement request {request_id} to ({target_az:.1f}°, {target_el:.1f}°)")

        with self.shared_data["state_lock"]:
            self.shared_data["system_state"].value = SystemState.TRACKER_MOVE.value

        self.servo.move_to_angle(target_el)
        success = self.stepper.move_to_angle(target_az, priority)
        time.sleep(MotorParams.MOVEMENT_SETTLE_TIME)

        self.shared_data["target_reached"].value = success
        self.shared_data["movement_complete_id"].value = request_id
        self.shared_data["go_to_target"].value = False

        if success: self.shared_data["total_movements"].value += 1
        else: self.shared_data["failed_movements"].value += 1

        self.current_request_id = request_id
        with self.shared_data["state_lock"]:
            self.shared_data["system_state"].value = SystemState.IDLE.value
        return success


# ============================================================================
# MAIN CONTROLLER
# ============================================================================

def hardware_controller_process(shared_data):
    """Main hardware controller process"""
    pi = None
    stepper = None
    servo = None
    lidar = None
    scanner = None
    coordinator = None
    servo_thread = None

    def signal_handler(sig, frame):
        print("\n[Controller] Shutdown signal received")
        shared_data["shutdown"].value = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        pi = pigpio.pi()
        if not pi.connected:
            print("[Controller] Failed to connect to pigpio daemon. Run: sudo pigpiod")
            return

        print("[Controller] Initializing subsystems...")
        stepper = StepperController(pi, shared_data)
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = BackgroundScanner(shared_data)
        coordinator = MovementCoordinator(shared_data, stepper, servo)

        servo_thread = threading.Thread(target=servo.control_loop)
        servo_thread.daemon = True
        servo_thread.start()

        print("[Controller] System ready")
        last_state = SystemState.IDLE

        while not shared_data["shutdown"].value:
            lidar.update_lidar_data()

            with shared_data["state_lock"]:
                current_state = SystemState(shared_data["system_state"].value)

            if current_state != last_state:
                print(f"[Controller] State: {last_state.name} -> {current_state.name}")
                shared_data["last_state_change"].value = time.time()
                last_state = current_state

            if shared_data["go_to_target"].value:
                if current_state == SystemState.SCANNING:
                    scanner.pause()
                    shared_data["background_scan_paused"].value = True

                coordinator.handle_tracker_request()

                if shared_data["background_scan_paused"].value:
                    scanner.resume()
                    shared_data["background_scan_paused"].value = False

            elif shared_data["background_scan_active"].value:
                if current_state == SystemState.IDLE:
                    with shared_data["state_lock"]:
                        shared_data["system_state"].value = SystemState.SCANNING.value

                if current_state == SystemState.SCANNING:
                    if not scanner.start_scan(lidar, stepper, servo):
                        with shared_data["state_lock"]:
                            shared_data["system_state"].value = SystemState.IDLE.value

            time.sleep(0.001)

    except Exception as e:
        print(f"[Controller] Error: {e}")
        import traceback
        traceback.print_exc()
        if "state_lock" in shared_data:
            with shared_data["state_lock"]:
                shared_data["system_state"].value = SystemState.ERROR.value

    finally:
        print("[Controller] Shutting down...")
        if "state_lock" in shared_data:
            with shared_data["state_lock"]:
                shared_data["system_state"].value = SystemState.SHUTDOWN.value

        if stepper: stepper.stop()
        if servo: servo.stop()
        if servo_thread and servo_thread.is_alive(): servo_thread.join(timeout=2)
        if lidar: lidar.stop()
        if pi and pi.connected: pi.stop()

        print("[Controller] Shutdown complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_hardware_controller(shared_data=None):
    """Run the hardware controller"""
    if shared_data is None:
        manager = Manager()
        # This is just for standalone testing, not used when called from main.py
        shared_data = {}

    print("=" * 60)
    print("Hardware Controller - Integrated Version")
    print(f"Scan Profile: {SCAN_AZIMUTH_SPEED}°/s")
    print(f"Position Tolerance: {MotorParams.POSITION_TOLERANCE}°")
    print("=" * 60)

    try:
        hardware_controller_process(shared_data)
    except KeyboardInterrupt:
        print("\n[Controller] Interrupted")
    except Exception as e:
        print(f"[Controller] Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_hardware_controller()