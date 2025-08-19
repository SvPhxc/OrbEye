#!/usr/bin/env python3
"""
Hardware Controller for Stepper Motor (Pan) and Servo (Tilt) Control
- Uses multiprocessing with shared data for safe communication.
- Open-loop stepper control with trapezoidal motion profiling for precise,
  no-overshoot movement using pigpio hardware-timed waveforms.
- Live, real-time position feedback via a step-counting callback.
- Continuous background scanning with unified movement logic to prevent
  hardware conflicts and memory-related crashes.
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
    # Stepper Parameters for Motion Profiling
    STEPPER_MAX_SPEED = 8000  # Steps per second (at cruise)
    STEPPER_MIN_SPEED = 400   # Steps per second (start/end speed)
    # How many steps to use for acceleration and deceleration ramp
    ACCEL_STEPS = 800

    # Servo Parameters
    SERVO_MIN_PULSE = 670
    SERVO_MAX_PULSE = 1670
    SERVO_MIN_ANGLE = 0
    SERVO_MAX_ANGLE = 90

    # Servo mounting offset - adjust this to set your zero point
    SERVO_ZERO_OFFSET = 0

    # Pan angle limits
    PAN_MIN_ANGLE = 0.0
    PAN_MAX_ANGLE = 360.0


# Continuous Scan Parameters
class ScanProfiles:
    FAST = {
        "azimuth_speed": 90.0,
        "elevation_step": 5.0,
        "data_rate": 30,
        "servo_move_time": 0.4,
        "servo_settle_time": 0.1,
    }
    NORMAL = {
        "azimuth_speed": 60.0,
        "elevation_step": 3.0,
        "data_rate": 40,
        "servo_move_time": 0.8,
        "servo_settle_time": 0.2,
    }

# Select scan profile here
CURRENT_SCAN_PROFILE = ScanProfiles.FAST

# Apply selected profile
SCAN_AZIMUTH_SPEED = CURRENT_SCAN_PROFILE["azimuth_speed"]
SCAN_ELEVATION_STEP = CURRENT_SCAN_PROFILE["elevation_step"]
SCAN_DATA_RATE = CURRENT_SCAN_PROFILE["data_rate"]
SERVO_MOVE_TIME = CURRENT_SCAN_PROFILE["servo_move_time"]
SERVO_SETTLE_TIME = CURRENT_SCAN_PROFILE["servo_settle_time"]

# Fixed parameters
SCAN_TILT_MAX = 90.0
SCAN_TILT_MIN = 0.0


class LidarController:
    """Controls LiDAR sensor and data collection"""
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.ser = None
        self.lidar_queue = queue.Queue(maxsize=100)
        self.shutdown_event = threading.Event()
        self.lidar_thread = None

        try:
            port = "/dev/serial0"
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            self.ser.write(bytearray([0x5A, 0x05, 0x11, 0x70]))
            print(f"[HWCtrl-LIDAR] LiDAR initialized on port {port}")
            self.lidar_thread = threading.Thread(target=self._lidar_reader_thread)
            self.lidar_thread.daemon = True
            self.lidar_thread.start()
        except Exception as e:
            print(f"[HWCtrl-LIDAR] Failed to initialize LiDAR: {e}")

    def _lidar_reader_thread(self):
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
        return None, None, None

    def stop(self):
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
        self.scan_direction = 1
        self.scan_start_time = None
        self._estimate_scan_time()

    def start_continuous_scan(self, lidar_controller, stepper_controller, servo_controller):
        if not self.shared_data["background_scan_active"].value:
            return False

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        print(f"[HWCtrl] Starting continuous scan at elevation {self.current_elevation:.1f}°")
        self.shared_data["target_elevation"].value = self.current_elevation
        time.sleep(SERVO_MOVE_TIME + SERVO_SETTLE_TIME)

        self._perform_continuous_azimuth_sweep(lidar_controller, stepper_controller)
        self.current_elevation -= SCAN_ELEVATION_STEP

        if self.current_elevation < SCAN_TILT_MIN:
            print("[HWCtrl] CONTINUOUS BACKGROUND SCAN completed.")
            total_time = time.time() - self.scan_start_time
            print(f"[HWCtrl] Total scan time: {total_time:.1f} seconds")
            self._save_scan_data()
            self._reset_scan()
            return False
        else:
            self.scan_direction *= -1
            return True

    def _perform_continuous_azimuth_sweep(self, lidar_controller, stepper_controller):
        """
        Perform a continuous azimuth sweep using the main move_to_angle method.
        This ensures consistent, safe movement and prevents hardware conflicts.
        """
        if self.scan_direction == 1:
            start_az, end_az = 0.0, 360.0
            print("[HWCtrl] Forward sweep: 0° -> 360°")
        else:
            start_az, end_az = 360.0, 0.0
            print("[HWCtrl] Backward sweep: 360° -> 0°")

        print(f"[HWCtrl] Positioning for sweep start at {start_az:.1f}°...")
        stepper_controller.move_to_angle(start_az)
        time.sleep(0.1)

        print(f"[HWCtrl] Starting continuous sweep movement to {end_az:.1f}°...")
        movement_thread = threading.Thread(target=stepper_controller.move_to_angle, args=(end_az,))
        movement_thread.daemon = True
        movement_thread.start()

        self._collect_data_during_movement(lidar_controller, movement_thread)
        movement_thread.join(timeout=15.0)

    def _collect_data_during_movement(self, lidar_controller, movement_thread):
        """Collect LiDAR data as long as the movement thread is active."""
        collection_interval = 1.0 / SCAN_DATA_RATE
        last_sample_time = time.time()
        sample_count = 0
        print(f"[HWCtrl] Data collection started at {SCAN_DATA_RATE} Hz...")

        while movement_thread.is_alive():
            current_time = time.time()
            if (current_time - last_sample_time) >= collection_interval:
                last_sample_time = current_time
                current_az = self.shared_data["stepper_degrees"].value
                current_el = self.shared_data["servo_degrees"].value
                dist, strength, timestamp = lidar_controller.get_lidar_data()
                if dist is not None and dist > 0:
                    self.background_data_buffer.append([current_az, current_el, dist, strength])
                    sample_count += 1
            else:
                time.sleep(0.001)
        print(f"[HWCtrl] Data collection completed. Total samples: {sample_count}")

    def _save_scan_data(self):
        if self.background_data_buffer:
            try:
                path = self.shared_data["background_path"].value
                data_array = np.array(self.background_data_buffer)
                np.save(path, data_array)
                print(f"[HWCtrl] Saved {len(self.background_data_buffer)} scan points to {path}")
            except Exception as e:
                print(f"[HWCtrl] Error saving scan data: {e}")

    def _estimate_scan_time(self):
        num_rings = int((SCAN_TILT_MAX - SCAN_TILT_MIN) / SCAN_ELEVATION_STEP) + 1
        time_per_rotation = 360.0 / SCAN_AZIMUTH_SPEED
        time_per_servo = SERVO_MOVE_TIME + SERVO_SETTLE_TIME
        total_scan_time = (num_rings * time_per_rotation) + ((num_rings - 1) * time_per_servo)
        print(f"[HWCtrl] Expected scan time: {total_scan_time:.1f}s")

    def _reset_scan(self):
        self.current_elevation = SCAN_TILT_MAX
        self.scan_direction = 1
        self.background_data_buffer = []
        self.shared_data["background_scan_active"].value = False
        self.scan_start_time = None
        print("[HWCtrl] Scan parameters reset.")


class PWMStepperController:
    """
    Open-loop stepper motor controller using pre-calculated motion profiles
    and pigpio waveforms for precise, hardware-timed execution.
    This version includes a step-counting callback to provide LIVE angle updates.
    """
    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.step_count = 0

        self.pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)

        self.pi.write(STEPPER_ENABLE_PIN, 0)
        self.pi.write(STEPPER_SLEEP_PIN, 1)
        time.sleep(0.001)

        self.step_callback = self.pi.callback(STEPPER_PULSE_PIN, pigpio.RISING_EDGE, self._step_counter_callback)
        print("[HWCtrl] Open-Loop Stepper controller with LIVE feedback initialized.")

    def _step_counter_callback(self, gpio, level, tick):
        if level == 1:
            direction = self.pi.read(STEPPER_DIR_PIN)
            if direction:
                self.step_count += 1
            else:
                self.step_count -= 1

            steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
            self.step_count = self.step_count % steps_per_rotation

            degrees = self.step_count * MICROSTEP_ANGLE
            with self.shared_data["stepper_degrees"].get_lock():
                self.shared_data["stepper_degrees"].value = degrees

    def move_to_angle(self, target_angle):
        self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 500000)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()

        target_angle = max(MotorParams.PAN_MIN_ANGLE, min(MotorParams.PAN_MAX_ANGLE, target_angle))

        current_pos_steps = self.step_count
        target_pos_steps = int(target_angle / MICROSTEP_ANGLE)
        error_steps = target_pos_steps - current_pos_steps

        steps_per_rotation = int(360.0 / MICROSTEP_ANGLE)
        if abs(error_steps) > (steps_per_rotation / 2):
            error_steps = error_steps - steps_per_rotation if error_steps > 0 else error_steps + steps_per_rotation

        total_steps = abs(error_steps)
        if total_steps < 1:
            return

        direction = 1 if error_steps > 0 else 0
        self.pi.write(STEPPER_DIR_PIN, direction)

        if total_steps <= MotorParams.ACCEL_STEPS * 2:
            accel_steps_actual = total_steps // 2
            decel_steps_actual = total_steps - accel_steps_actual
        else:
            accel_steps_actual = MotorParams.ACCEL_STEPS
            decel_steps_actual = MotorParams.ACCEL_STEPS

        pulses = []
        for i in range(1, accel_steps_actual + 1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / accel_steps_actual)
            delay_us = int(500000 / speed)
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        cruise_steps = total_steps - (accel_steps_actual + decel_steps_actual)
        if cruise_steps > 0:
            delay_us = int(500000 / MotorParams.STEPPER_MAX_SPEED)
            for _ in range(cruise_steps):
                pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
                pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        for i in range(decel_steps_actual, 0, -1):
            speed = MotorParams.STEPPER_MIN_SPEED + (MotorParams.STEPPER_MAX_SPEED - MotorParams.STEPPER_MIN_SPEED) * (i / decel_steps_actual)
            delay_us = int(500000 / speed)
            pulses.append(pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, delay_us))
            pulses.append(pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, delay_us))

        self.pi.wave_add_generic(pulses)
        wave_id = self.pi.wave_create()

        if wave_id >= 0:
            self.pi.wave_send_once(wave_id)
            while self.pi.wave_tx_busy():
                time.sleep(0.01)
            self.pi.wave_delete(wave_id)

        final_pos_deg = self.shared_data["stepper_degrees"].value
        print(f"[HWCtrl-Stepper] Movement complete. Final position: {final_pos_deg:.3f}°")

    def move_to_target(self):
        target_pos = self.shared_data["target_azimuth"].value
        self.move_to_angle(target_pos)
        self.shared_data["target_reached"].value = True

    def stop(self):
        self.running = False
        print("[HWCtrl] Stopping stepper controller...")
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        if self.step_callback:
            self.step_callback.cancel()
            self.step_callback = None
        self.pi.write(STEPPER_PULSE_PIN, 0)
        self.pi.write(STEPPER_ENABLE_PIN, 1)
        print("[HWCtrl] Stepper controller stopped.")


class ServoController:
    """Controls servo motor for tilt movement"""
    def __init__(self, pi, shared_data):
        self.pi = pi
        self.shared_data = shared_data
        self.running = True
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        self.move_to_angle(45.0)
        print(f"[HWCtrl] Servo controller initialized (zero offset: {MotorParams.SERVO_ZERO_OFFSET}°)")

    def angle_to_pulse_width(self, angle):
        user_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, angle))
        physical_angle = user_angle - MotorParams.SERVO_ZERO_OFFSET
        physical_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, physical_angle))
        pulse_range = MotorParams.SERVO_MAX_PULSE - MotorParams.SERVO_MIN_PULSE
        angle_range = MotorParams.SERVO_MAX_ANGLE - MotorParams.SERVO_MIN_ANGLE
        pulse_width = MotorParams.SERVO_MIN_PULSE + ((physical_angle - MotorParams.SERVO_MIN_ANGLE) / angle_range * pulse_range)
        return int(pulse_width)

    def move_to_angle(self, target_angle):
        target_angle = max(MotorParams.SERVO_MIN_ANGLE, min(MotorParams.SERVO_MAX_ANGLE, target_angle))
        pulse_width = self.angle_to_pulse_width(target_angle)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
        with self.shared_data["servo_degrees"].get_lock():
            self.shared_data["servo_degrees"].value = target_angle

    def control_servo(self):
        while self.running:
            target_elevation = self.shared_data["target_elevation"].value
            current_servo = self.shared_data["servo_degrees"].value
            if abs(target_elevation - current_servo) > 0.5:
                self.move_to_angle(target_elevation)
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
        print("[HWCtrl] Servo controller stopped")


def combined_controller_process(shared_data):
    """Main process for controlling all hardware components."""
    pi, stepper, servo, lidar, servo_thread = None, None, None, None, None
    try:
        pi = pigpio.pi()
        if not pi.connected:
            print("[HWCtrl] Failed to connect to pigpio daemon")
            return

        stepper = PWMStepperController(pi, shared_data)
        servo = ServoController(pi, shared_data)
        lidar = LidarController(shared_data)
        scanner = ContinuousBackgroundScanner(shared_data)

        servo_thread = threading.Thread(target=servo.control_servo)
        servo_thread.daemon = True
        servo_thread.start()

        print("[HWCtrl] Combined controller initialized.")
        while not shared_data["shutdown"].value:
            lidar.get_lidar_data()
            if shared_data["background_scan_active"].value:
                if not scanner.start_continuous_scan(lidar, stepper, servo):
                    time.sleep(0.1)
            elif shared_data["go_to_target"].value:
                shared_data["target_reached"].value = False
                stepper.move_to_target()
                shared_data["go_to_target"].value = False
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"[HWCtrl] Combined controller error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[HWCtrl] Shutting down...")
        if stepper: stepper.stop()
        if servo: servo.stop()
        if servo_thread: servo_thread.join(timeout=1.0)
        if lidar: lidar.stop()
        if pi and pi.connected: pi.stop()
        print("[HWCtrl] Shutdown complete.")


if __name__ == '__main__':
    """
    Example of how to run the controller process.
    This would typically be launched from a main application.
    """
    print("Starting hardware controller process simulation.")
    try:
        with Manager() as manager:
            shared_data = {
                "shutdown": manager.Value('b', False),
                "go_to_target": manager.Value('b', False),
                "target_reached": manager.Value('b', True),
                "target_azimuth": manager.Value('f', 0.0),
                "target_elevation": manager.Value('f', 45.0),
                "stepper_degrees": manager.Value('f', 0.0),
                "servo_degrees": manager.Value('f', 45.0),
                "background_scan_active": manager.Value('b', False),
                "background_path": manager.Value('c', b'scan_data.npy'),
                "lidar_data": manager.Array('f', [0.0, 0.0, 0.0]),
            }

            controller_process = Process(target=combined_controller_process, args=(shared_data,))
            controller_process.start()

            # --- Example Command Sequence ---
            print("\n--- SIMULATING USER COMMANDS ---")

            # Command 1: Go to 90 degrees
            print("\nCOMMAND: Go to 90 degrees")
            shared_data["target_azimuth"].value = 90.0
            shared_data["go_to_target"].value = True
            while shared_data["go_to_target"].value:
                time.sleep(0.1)
            print("Move to 90 complete.")

            time.sleep(2)

            # Command 2: Start a background scan
            print("\nCOMMAND: Start background scan")
            shared_data["background_scan_active"].value = True
            while shared_data["background_scan_active"].value:
                time.sleep(0.5)
            print("Background scan complete.")

            time.sleep(2)

            # Command 3: Go back to 0 degrees
            print("\nCOMMAND: Go to 0 degrees")
            shared_data["target_azimuth"].value = 0.0
            shared_data["go_to_target"].value = True
            while shared_data["go_to_target"].value:
                time.sleep(0.1)
            print("Move to 0 complete.")

            # --- End of Simulation ---
            print("\n--- SIMULATION COMPLETE ---")
            shared_data["shutdown"].value = True
            controller_process.join(timeout=5)
            if controller_process.is_alive():
                print("Controller process did not shut down cleanly, terminating.")
                controller_process.terminate()

    except KeyboardInterrupt:
        print("\nMain process interrupted by user.")
    finally:
        print("Hardware controller process stopped.")