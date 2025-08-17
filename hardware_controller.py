import time
import serial
import pigpio
import threading
import queue
import numpy as np
import traceback

# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6

# --- System & Microstepping Configuration ---
# This is the angle of a single microstep. (e.g., 1.8 deg/step motor with 32x microstepping = 1.8/32 = 0.05625)
MICROSTEP_ANGLE = 0.05625
# Maximum pulse frequency pigpio can generate reliably. Limits top speed.
MAX_PULSE_FREQ = 250000

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
# The angular distance between each measurement point.
SCAN_STEP_DEG = 1.0
# How long to pause at each point to let motors settle before taking a reading.
SCAN_SETTLE_TIME_S = 0.02  # 20 milliseconds

# --- PID & Movement Tuning ---
TARGET_REACHED_THRESHOLD_DEG = 0.5  # How close to the target we need to be.

# PID gains control how the motors move to their targets.
# Higher KP = faster, more aggressive movement.
MAX_PAN_SPEED_DPS = 1000.0
PAN_KP, PAN_KI, PAN_KD = 10.0, 0.0, 0.01

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

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 0:
            # Avoid division by zero or negative time travel
            return self.output_limits[0] if self.setpoint < current_value else self.output_limits[1]

        error = self.setpoint - current_value

        # Handle wrap-around for circular systems like a 360-degree pan
        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            if abs(error) > range_width / 2:
                if error > 0:
                    error -= range_width
                else:
                    error += range_width

        self._integral += error * dt
        # Simple anti-windup
        self._integral = max(-1.0, min(1.0, self._integral))

        derivative = (error - self._last_error) / dt

        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)

        self._last_error = error
        self._last_time = time.monotonic()

        return max(self.output_limits[0], min(self.output_limits[1], output))

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def get_setpoint(self):
        return self.setpoint

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()


class HardwareController:
    """Manages all hardware interactions including motors and LiDAR."""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=100)  # Increased queue size
        self.background_data_buffer = []

        # Internal state tracking for motor positions
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

        # State flags for the new "Step-and-Scan" logic
        self.current_scan_el = SCAN_TILT_MAX
        self.current_scan_az = SCAN_PAN_MIN
        self.scan_pan_direction = 1  # 1 for forward (MIN to MAX), -1 for reverse
        self.scan_state = "MOVING"  # Can be "MOVING" or "SAMPLING"
        self.scan_settle_start_time = 0

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                # Find the start of a frame
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    dist = frame[0] + (frame[1] << 8)
                    strength = frame[2] + (frame[3] << 8)
                    # Put the latest data in the queue
                    if not self.lidar_queue.full():
                        self.lidar_queue.put_nowait((dist, strength, time.time()))
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set():
                    print("[HWCtrl-LIDAR] Serial error.")
                break
            except queue.Full:
                # If queue is full, we are not processing fast enough.
                # It's better to drop old data.
                pass
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        """Send commands to the stepper and servo motors."""
        # Update internal position based on velocity command
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(SCAN_TILT_MIN,
                                     min(SCAN_TILT_MAX, self.internal_tilt_pos + (tilt_velocity_dps * dt)))

        # Stepper motor control
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            pulse_freq = int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, MAX_PULSE_FREQ))
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, pulse_freq, 500000)  # 50% duty cycle
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)  # Stop motor

        # Servo motor control (pulse width range for 0-90 degrees may need tuning)
        pulse_width = int(500 + (self.internal_tilt_pos / 90.0) * 2000)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def _handle_state_machine(self, current_state):
        if self.shared_data["background_scan_active"].value:
            next_state = "BACKGROUND_SCAN"
        elif self.shared_data["go_to_target"].value:
            next_state = "GOTO_POSITION"
        else:
            # Default to IDLE if no other state is active
            next_state = "IDLE"

        if next_state != current_state:
            print(f"[HWCtrl] State change: {current_state} -> {next_state}")
            self.pan_pid.reset()
            self.tilt_pid.reset()
            # Initialize scan parameters when starting a new background scan
            if next_state == "BACKGROUND_SCAN":
                self.current_scan_el = SCAN_TILT_MAX
                self.current_scan_az = SCAN_PAN_MIN
                self.scan_pan_direction = 1
                self.scan_state = "MOVING"
                self.background_data_buffer = []  # Clear old data
        return next_state

    def _run_goto_position_state(self):
        target_az = self.shared_data["target_azimuth"].value
        target_el = self.shared_data["target_elevation"].value

        self.pan_pid.set_setpoint(target_az)
        self.tilt_pid.set_setpoint(target_el)

        pan_vel = self.pan_pid.update(self.internal_pan_pos)
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        pan_error = abs(self.pan_pid.get_setpoint() - self.internal_pan_pos)
        tilt_error = abs(self.tilt_pid.get_setpoint() - self.internal_tilt_pos)

        # Handle pan wrap-around for error checking
        if pan_error > 180:
            pan_error = 360 - pan_error

        target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG
        self.shared_data["target_reached"].value = target_reached

        return pan_vel, tilt_vel

    def _run_background_scan_state(self):
        """Handles the 'Step-and-Scan' logic."""
        pan_vel, tilt_vel = 0, 0

        # Set the target for the PID controllers
        self.pan_pid.set_setpoint(self.current_scan_az)
        self.tilt_pid.set_setpoint(self.current_scan_el)

        # Calculate error to check if we've reached the target
        pan_error = abs(self.pan_pid.get_setpoint() - self.internal_pan_pos)
        if pan_error > 180: pan_error = 360 - pan_error  # Handle wrap around
        tilt_error = abs(self.tilt_pid.get_setpoint() - self.internal_tilt_pos)
        is_target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG

        if self.scan_state == "MOVING":
            if is_target_reached:
                # We've arrived. Stop the motors and switch to SAMPLING state.
                self.scan_state = "SAMPLING"
                self.scan_settle_start_time = time.monotonic()
                pan_vel, tilt_vel = 0, 0
            else:
                # Still moving towards the target point.
                pan_vel = self.pan_pid.update(self.internal_pan_pos)
                tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        elif self.scan_state == "SAMPLING":
            # Wait for the settling time to pass
            if time.monotonic() - self.scan_settle_start_time >= SCAN_SETTLE_TIME_S:
                # --- DATA RECORDING ---
                # Clear old readings from the queue and get the most recent one
                last_reading = None
                while not self.lidar_queue.empty():
                    last_reading = self.lidar_queue.get_nowait()

                if last_reading:
                    dist, strength, _ = last_reading
                    # Log the data with the TARGET position for perfect accuracy
                    self.background_data_buffer.append(
                        [self.current_scan_az, self.current_scan_el, dist, strength]
                    )

                # --- MOVE TO NEXT POINT ---
                # Calculate the next azimuth position
                self.current_scan_az += SCAN_STEP_DEG * self.scan_pan_direction

                # Check if we reached the end of a row
                if self.scan_pan_direction == 1 and self.current_scan_az > SCAN_PAN_MAX:
                    # End of forward row, move down and reverse
                    self.current_scan_el -= SCAN_STEP_DEG
                    self.scan_pan_direction = -1
                    self.current_scan_az = SCAN_PAN_MAX  # Set target for next row
                elif self.scan_pan_direction == -1 and self.current_scan_az < SCAN_PAN_MIN:
                    # End of backward row, move down and go forward
                    self.current_scan_el -= SCAN_STEP_DEG
                    self.scan_pan_direction = 1
                    self.current_scan_az = SCAN_PAN_MIN  # Set target for next row

                # Check if scan is complete
                if self.current_scan_el < SCAN_TILT_MIN:
                    print("[HWCtrl] BACKGROUND_SCAN finished.")
                    self._save_background_data()
                    self.shared_data["background_scan_active"].value = False
                else:
                    # Switch back to MOVING state for the next point
                    self.scan_state = "MOVING"

        return pan_vel, tilt_vel

    def _save_background_data(self):
        """Saves the collected background scan data to a file."""
        if self.background_data_buffer:
            print(f"[HWCtrl] Saving {len(self.background_data_buffer)} background scan points...")
            try:
                np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer))
                print(f"[HWCtrl] Data saved to {self.shared_data['background_path'].value}")
            except Exception as e:
                print(f"[HWCtrl] ERROR saving background data: {e}")
        else:
            print("[HWCtrl] No background data to save.")

    def run(self):
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpio connection failed.")

            # Setup GPIO pins
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")

            # Setup serial for LiDAR
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            # This command might be specific to your LiDAR model for setting sample rate/mode
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))

            lidar_thread = threading.Thread(target=self._lidar_reader_thread, daemon=True)
            lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time = time.monotonic()
            current_state = "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001:  # Loop at roughly 1kHz max
                    time.sleep(0.0005)
                    continue
                last_loop_time = time.monotonic()

                current_state = self._handle_state_machine(current_state)

                pan_vel, tilt_vel = 0, 0
                if current_state == "GOTO_POSITION":
                    pan_vel, tilt_vel = self._run_goto_position_state()
                elif current_state == "BACKGROUND_SCAN":
                    pan_vel, tilt_vel = self._run_background_scan_state()
                # In IDLE state, velocities remain 0

                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Process any remaining LiDAR data for live feed (not used for background scan)
                try:
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()
                        with self.shared_data["lidar_data"].get_lock():
                            self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass

                # Update shared position data
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

        except Exception as e:
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive():
                lidar_thread.join(timeout=1)

            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0)
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop()
                print("[HWCtrl] pigpio resources released.")

            if self.ser and self.ser.is_open:
                self.ser.close()
                print("[HWCtrl] Serial port closed.")


def run_hardware_controller(shared_data):
    """Entry point function to run the hardware controller."""
    controller = HardwareController(shared_data)
    controller.run()