import time
import serial
import pigpio
import threading
import queue
import numpy as np
import traceback
from collections import deque

# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6

# --- System & Microstepping Configuration ---
MICROSTEP_ANGLE = 0.05625  # Angle of a single microstep (1.8 deg / 32x microstepping)
MAX_PULSE_FREQ = 250000  # Max pulse frequency for stepper

# --- Define the boundaries and speed for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0  # This now defines the vertical step between sweep rows
SCAN_PAN_SPEED_DPS = 200.0  # Desired speed of the horizontal sweep in degrees per second

# This is a "braking zone" at the edges of the scan to allow the PID to smoothly reverse direction.
SCAN_TURNAROUND_DEG = 5.0

# --- PID & Movement Tuning ---
TARGET_REACHED_THRESHOLD_DEG = 0.5  # For GOTO commands, not used in scan

MAX_PAN_SPEED_DPS = 1000.0  # Absolute maximum speed the PID can command
PAN_KP, PAN_KI, PAN_KD = 15.0, 0.0, 0.05  # Tighter PID gains for responsive tracking

MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 10.0, 0.0, 0.01


class PIDController:
    """A Proportional-Integral-Derivative (PID) controller."""

    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-150, 150), wrap_range=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.wrap_range = wrap_range
        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()
        self._last_output = 0

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 1e-6:
            return self._last_output

        error = self.setpoint - current_value

        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            if abs(error) > range_width / 2:
                if error > 0:
                    error -= range_width
                else:
                    error += range_width

        self._integral += error * dt
        self._integral = max(-1.0, min(1.0, self._integral))
        derivative = (error - self._last_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)

        self._last_error = error
        self._last_time = time.monotonic()
        self._last_output = max(self.output_limits[0], min(self.output_limits[1], output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()


class HardwareController:
    """Manages all hardware interactions with latency compensation."""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=500)
        self.background_data_buffer = []

        # Buffer to store recent history of (timestamp, pan_pos, tilt_pos)
        # A deque is a highly efficient list for adding/removing from the ends.
        self.position_history = deque(maxlen=200)

        self.internal_pan_pos = shared_data["stepper_degrees"].value
        self.internal_tilt_pos = shared_data["servo_degrees"].value

        self.pan_pid = PIDController(
            PAN_KP, PAN_KI, PAN_KD,
            output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS),
            wrap_range=(0, 360)
        )
        self.tilt_pid = PIDController(
            TILT_KP, TILT_KI, TILT_KD,
            output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS)
        )

        # State flags for scanning logic
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.scan_target_az = SCAN_PAN_MIN

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    dist = frame[0] + (frame[1] << 8)
                    strength = frame[2] + (frame[3] << 8)
                    # Timestamp is captured HERE, as early as possible.
                    ts = time.monotonic()
                    if not self.lidar_queue.full():
                        self.lidar_queue.put_nowait((dist, strength, ts))
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error.")
                break
            except queue.Full:
                pass
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(SCAN_TILT_MIN, min(SCAN_TILT_MAX, self.internal_tilt_pos + tilt_velocity_dps * dt))

        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            pulse_freq = int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, MAX_PULSE_FREQ))
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, pulse_freq, 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        pulse_width = int(500 + (self.internal_tilt_pos / 90.0) * 2000)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def _handle_state_machine(self, current_state):
        if self.shared_data["background_scan_active"].value:
            next_state = "BACKGROUND_SCAN"
        elif self.shared_data["go_to_target"].value:
            next_state = "GOTO_POSITION"
        else:
            next_state = "IDLE"

        if next_state != current_state:
            print(f"[HWCtrl] State change: {current_state} -> {next_state}")
            self.pan_pid.reset();
            self.tilt_pid.reset()
            if next_state == "BACKGROUND_SCAN":
                self.current_scan_el = SCAN_TILT_MAX
                self.scan_pan_direction = 1
                self.scan_target_az = self.internal_pan_pos  # Start sweep from current pos
        return next_state

    def _run_goto_position_state(self):
        # This function is for simple "go to target and stop"
        target_az = self.shared_data["target_azimuth"].value
        target_el = self.shared_data["target_elevation"].value
        self.pan_pid.set_setpoint(target_az)
        self.tilt_pid.set_setpoint(target_el)
        pan_vel = self.pan_pid.update(self.internal_pan_pos)
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)
        return pan_vel, tilt_vel

    def _run_background_scan_state(self, dt):
        """Handle the continuous sweeping scan."""
        # Determine the target Azimuth for this timestep
        if self.scan_pan_direction == 1:  # Sweeping forward
            self.scan_target_az += SCAN_PAN_SPEED_DPS * dt
            if self.scan_target_az >= SCAN_PAN_MAX:
                self.scan_target_az = SCAN_PAN_MAX
                self.current_scan_el -= SCAN_STEP_DEG
                self.scan_pan_direction = -1
        else:  # Sweeping backward
            self.scan_target_az -= SCAN_PAN_SPEED_DPS * dt
            if self.scan_target_az <= SCAN_PAN_MIN:
                self.scan_target_az = SCAN_PAN_MIN
                self.current_scan_el -= SCAN_STEP_DEG
                self.scan_pan_direction = 1

        # Check if scan is complete
        if self.current_scan_el < SCAN_TILT_MIN:
            print("[HWCtrl] BACKGROUND_SCAN finished.")
            self._save_background_data()
            self.shared_data["background_scan_active"].value = False
            return 0, 0

        # The PID controller's job is to make the motor follow the moving scan_target_az
        self.pan_pid.set_setpoint(self.scan_target_az)
        self.tilt_pid.set_setpoint(self.current_scan_el)
        pan_vel = self.pan_pid.update(self.internal_pan_pos)
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        return pan_vel, tilt_vel

    def _get_interpolated_position(self, target_time):
        """Finds the motor position at a specific time using interpolation."""
        if not self.position_history: return None, None

        # Find the two history points that bracket the target_time
        p1, p2 = None, None
        for i in range(len(self.position_history) - 1, -1, -1):
            if self.position_history[i][0] <= target_time:
                p1 = self.position_history[i]
                if i + 1 < len(self.position_history):
                    p2 = self.position_history[i + 1]
                break

        if p1 is None or p2 is None:
            # Not enough history to interpolate, return the oldest known position
            return self.position_history[0][1], self.position_history[0][2]

        t1, pan1, tilt1 = p1
        t2, pan2, tilt2 = p2

        time_diff = t2 - t1
        if time_diff <= 1e-9:  # Avoid division by zero
            return pan1, tilt1

        # Calculate the interpolation factor
        fraction = (target_time - t1) / time_diff

        # Handle pan wrap-around (e.g., interpolating from 359 to 1 degree)
        pan_diff = pan2 - pan1
        if pan_diff > 180:
            pan_diff -= 360
        elif pan_diff < -180:
            pan_diff += 360

        interpolated_pan = (pan1 + fraction * pan_diff) % 360
        interpolated_tilt = tilt1 + fraction * (tilt2 - tilt1)

        return interpolated_pan, interpolated_tilt

    def _save_background_data(self):
        if self.background_data_buffer:
            print(f"[HWCtrl] Saving {len(self.background_data_buffer)} background scan points...")
            try:
                np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer))
                print(f"[HWCtrl] Data saved to {self.shared_data['background_path'].value}")
            except Exception as e:
                print(f"[HWCtrl] ERROR saving background data: {e}")

    def run(self):
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")

            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")

            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))

            lidar_thread = threading.Thread(target=self._lidar_reader_thread, daemon=True)
            lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time = time.monotonic()
            current_state = "IDLE"

            while not self.shared_data["shutdown"].value:
                loop_time = time.monotonic()
                dt = loop_time - last_loop_time
                if dt <= 0.001:  # Loop at ~1kHz
                    time.sleep(0.0005)
                    continue
                last_loop_time = loop_time

                current_state = self._handle_state_machine(current_state)

                pan_vel, tilt_vel = 0, 0
                if current_state == "GOTO_POSITION":
                    pan_vel, tilt_vel = self._run_goto_position_state()
                elif current_state == "BACKGROUND_SCAN":
                    pan_vel, tilt_vel = self._run_background_scan_state(dt)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Append current state to our history buffer for interpolation
                self.position_history.append((loop_time, self.internal_pan_pos, self.internal_tilt_pos))

                # Process LiDAR data with interpolation
                try:
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()

                        if current_state == "BACKGROUND_SCAN":
                            pan_pos, tilt_pos = self._get_interpolated_position(ts)
                            if pan_pos is not None:
                                self.background_data_buffer.append([pan_pos, tilt_pos, dist, strength])
                        else:  # Update live data if not scanning
                            with self.shared_data["lidar_data"].get_lock():
                                self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

        except Exception as e:
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join(timeout=1)
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop();
                print("[HWCtrl] pigpio resources released.")
            if self.ser and self.ser.is_open: self.ser.close(); print("[HWCtrl] Serial port closed.")


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()