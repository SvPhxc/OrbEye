#!/usr/bin/env python3
"""
Improved Hardware Controller with Time-Based Position Tracking
Fixes position lag during high-speed movements by calculating position
based on time and commanded speed rather than counting individual steps.
"""

import time
import math
import pigpio
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
    STEPPER_MAX_SPEED = 8000  # max steps per second
    STEPPER_MIN_SPEED = 100  # min steps per second
    STEPPER_CRUISE_SPEED = 6400  # cruise speed for scans
    ACCEL_STEPS = 150  # acceleration/deceleration steps

    # Speed threshold for switching between counting and time-based tracking
    COUNTING_SPEED_THRESHOLD = 2000  # Below this, use step counting; above, use time-based

    # PID Parameters
    KP = 0.2
    KI = 0.0
    KD = 0.0

    # Servo Parameters
    SERVO_MIN_PULSE = 670
    SERVO_MAX_PULSE = 1670
    SERVO_MIN_ANGLE = 0
    SERVO_MAX_ANGLE = 90
    SERVO_ZERO_OFFSET = 0

    # Pan angle limits
    PAN_MIN_ANGLE = 0.0
    PAN_MAX_ANGLE = 360.0


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