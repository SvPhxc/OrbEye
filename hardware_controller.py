import time
import serial
import pigpio
import threading
import queue
import numpy as np

# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 0.01  # Wider threshold for settling in the turnaround zone
SCAN_PAN_SPEED_DPS = 1000.0  # Can be set aggressively now
SCAN_TURNAROUND_DEG = 0.05  # Defines a "braking zone" at scan edges for PID settling.

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 2.0

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 1000.0
PAN_KP, PAN_KI, PAN_KD = 10.0, 0.001, 0.002
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 10.0, 0.000, 0.000


class PIDController:
    """A Proportional-Integral-Derivative (PID) controller."""

    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-150, 150), anti_windup_limit=1, wrap_range=None):
        """
        Initialize the PID controller.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): The target value.
            output_limits (tuple): A tuple (min, max) for the output value.
            anti_windup_limit (float): The limit for the integral term.
            wrap_range (tuple): A tuple (min, max) for handling wrap-around values (e.g., 0-360 degrees).
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup_limit = anti_windup_limit
        self.wrap_range = wrap_range

        self._integral = 0
        self._last_error = 0
        self._last_output = 0
        self._last_time = time.monotonic()

    def update(self, current_value, pan_avoid_wrap=False):
        """
        Calculate the PID output value for a given measurement.

        Args:
            current_value (float): The current measured value.
            pan_avoid_wrap (bool): Flag to handle angle wrapping logic specifically.

        Returns:
            float: The calculated output.
        """
        dt = time.monotonic() - self._last_time
        if dt <= 0:
            return self._last_output

        error = self.setpoint - current_value

        # Handle wrap-around for circular systems like a 360-degree pan
        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            if pan_avoid_wrap and abs(error) > range_width / 2:
                if error > 0:
                    error -= range_width
                else:
                    error += range_width
            else:
                error = (error + range_width / 2) % range_width - range_width / 2

        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))

        derivative = (error - self._last_error) / dt

        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)

        self._last_error = error
        self._last_time = time.monotonic()
        self._last_output = max(self.output_limits[0], min(self.output_limits[1], output))

        return self._last_output

    def set_setpoint(self, new_setpoint):
        """Update the setpoint of the controller."""
        self.setpoint = new_setpoint

    def get_setpoint(self):
        """Return the current setpoint."""
        return self.setpoint

    def reset(self):
        """Reset the PID controller's integral and error terms."""
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
        self.lidar_queue = queue.Queue(maxsize=20)
        self.background_data_buffer = []

        self.internal_pan_pos = shared_data["stepper_degrees"].value
        self.internal_tilt_pos = shared_data["servo_degrees"].value

        self.pan_pid = PIDController(
            PAN_KP, PAN_KI, PAN_KD,
            output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS),
            anti_windup_limit=1,
            wrap_range=(0, 360)
        )
        self.tilt_pid = PIDController(
            TILT_KP, TILT_KI, TILT_KD,
            output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS)
        )

        # State flags for scanning logic
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.scan_target_az = 0.0
        self.scan_is_turning = False
        self.pan_avoid_wrap = False

    def _calculate_pan_error(self, setpoint, current_value, avoid_wrap=False):
        """Helper to calculate panoramic angle error with wrap-around logic."""
        error = setpoint - current_value
        range_width = 360
        if avoid_wrap and abs(error) > range_width / 2:
            if error > 0:
                error -= range_width
            else:
                error += range_width
        else:
            error = (error + range_width / 2) % range_width - range_width / 2
        return error

    def _lidar_reader_thread(self):
        """Dedicated thread to continuously read data from the LiDAR sensor."""
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    try:
                        dist = frame[0] + (frame[1] << 8)
                        strength = frame[2] + (frame[3] << 8)
                        self.lidar_queue.put_nowait((dist, strength, time.time()))
                    except queue.Full:
                        pass  # Ignore if the queue is full
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set():
                    print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        """Send commands to the stepper and servo motors."""
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))

        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            pulse_freq = int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, 250000))
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, pulse_freq, 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        pulse_width = int(500 + (self.internal_tilt_pos / 0.09) + (36 / 0.09))
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def _handle_state_machine(self, current_state):
        """Determine the current operational state of the controller."""
        if self.shared_data["background_scan_active"].value:
            next_state = "BACKGROUND_SCAN"
        elif self.shared_data["go_to_target"].value:
            next_state = "GOTO_POSITION"
        elif self.shared_data["lidar_track_mode_active"].value:
            next_state = "HF_TRACKING"
        else:
            next_state = "IDLE"

        if next_state != current_state:
            print(f"[HWCtrl] State change: {current_state} -> {next_state}")
            self.pan_pid.reset()
            self.tilt_pid.reset()
            if next_state == "BACKGROUND_SCAN":
                self.current_scan_el = SCAN_TILT_MAX
                self.scan_pan_direction = 1
                self.scan_is_turning = False
                self.scan_target_az = self.internal_pan_pos
        return next_state

    def _run_goto_or_tracking_state(self, current_state):
        """Handle GOTO_POSITION and HF_TRACKING states."""
        if current_state == "GOTO_POSITION":
            target_az = self.shared_data["target_azimuth"].value
            target_el = self.shared_data["target_elevation"].value
        else:  # HF_TRACKING
            target_az = self.shared_data["predicted_azimuth"].value
            target_el = self.shared_data["predicted_elevation"].value

        if abs(target_az - self.internal_pan_pos) > 180:
            self.pan_avoid_wrap = True

        self.pan_pid.set_setpoint(target_az)
        self.tilt_pid.set_setpoint(target_el)

        pan_vel = self.pan_pid.update(self.internal_pan_pos, self.pan_avoid_wrap)
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        pan_error = abs(self._calculate_pan_error(target_az, self.internal_pan_pos, self.pan_avoid_wrap))
        tilt_error = abs(target_el - self.internal_tilt_pos)
        target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG

        if current_state == "GOTO_POSITION":
            self.shared_data["target_reached"].value = target_reached
            if target_reached:
                self.pan_avoid_wrap = False

        return pan_vel, tilt_vel

    def _run_background_scan_state(self, dt):
        """Handle the BACKGROUND_SCAN state."""
        if self.current_scan_el < SCAN_TILT_MIN:
            print("[HWCtrl] BACKGROUND_SCAN finished.")
            self._save_background_data()
            self.shared_data["background_scan_active"].value = False
        elif self.scan_is_turning:
            pan_error = abs(self._calculate_pan_error(self.pan_pid.get_setpoint(), self.internal_pan_pos))
            if pan_error < TARGET_REACHED_THRESHOLD_DEG:
                self.pan_pid.reset()
                self.scan_pan_direction *= -1
                self.current_scan_el -= SCAN_STEP_DEG
                self.scan_is_turning = False
                self.scan_target_az = SCAN_PAN_MIN if self.scan_pan_direction == 1 else SCAN_PAN_MAX
                self.pan_avoid_wrap = True
                print(
                    f"[HWCtrl-SCAN] Row finished. New elevation: {self.current_scan_el:.1f} deg, Direction: {self.scan_pan_direction}")
        else:  # Sweeping
            self.scan_target_az += SCAN_PAN_SPEED_DPS * self.scan_pan_direction * dt
            if self.scan_pan_direction == 1 and self.scan_target_az >= SCAN_PAN_MAX - SCAN_TURNAROUND_DEG:
                self.scan_target_az = SCAN_PAN_MAX
                self.scan_is_turning = True
                self.pan_avoid_wrap = False
            elif self.scan_pan_direction == -1 and self.scan_target_az <= SCAN_PAN_MIN + SCAN_TURNAROUND_DEG:
                self.scan_target_az = SCAN_PAN_MIN
                self.scan_is_turning = True
                self.pan_avoid_wrap = False

        self.pan_pid.set_setpoint(self.scan_target_az)
        pan_vel = self.pan_pid.update(self.internal_pan_pos, self.pan_avoid_wrap)
        self.tilt_pid.set_setpoint(self.current_scan_el)
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        return pan_vel, tilt_vel

    def _save_background_data(self):
        """Saves the collected background scan data to a file."""
        if self.background_data_buffer:
            print(f"[HWCtrl] Auto-saving {len(self.background_data_buffer)} background scan points...")
            try:
                np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer))
                print(f"[HWCtrl] Data saved to {self.shared_data['background_path'].value}")
                self.background_data_buffer = []  # Clear buffer after saving
            except Exception as e:
                print(f"[HWCtrl] ERROR saving background data: {e}")

    def run(self):
        """Main loop for the hardware controller."""
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpio connection failed.")

            # Setup GPIO pins
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0)  # Enable driver
            self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake driver
            print("[HWCtrl] Stepper driver enabled.")

            # Setup serial for LiDAR
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4]))

            # Start LiDAR reader thread
            lidar_thread = threading.Thread(target=self._lidar_reader_thread, daemon=True)
            lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time = time.monotonic()
            current_state = "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.0005:
                    continue
                last_loop_time = time.monotonic()

                current_state = self._handle_state_machine(current_state)

                pan_vel, tilt_vel = 0, 0
                if current_state in ["GOTO_POSITION", "HF_TRACKING"]:
                    pan_vel, tilt_vel = self._run_goto_or_tracking_state(current_state)
                elif current_state == "BACKGROUND_SCAN":
                    pan_vel, tilt_vel = self._run_background_scan_state(dt)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Process LiDAR data from queue
                try:
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()
                        with self.shared_data["lidar_data"].get_lock():
                            self.shared_data["lidar_data"][:] = [dist, strength, ts]
                        if current_state == "BACKGROUND_SCAN":
                            self.background_data_buffer.append(
                                [self.internal_pan_pos, self.internal_tilt_pos, dist, strength])
                except queue.Empty:
                    pass

                # Update shared position data
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

                time.sleep(0.0001)

        except Exception as e:
            import traceback
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive():
                lidar_thread.join(timeout=1)

            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                self.pi.write(STEPPER_ENABLE_PIN, 1)  # Disable driver
                self.pi.write(STEPPER_SLEEP_PIN, 0)  # Sleep driver
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop()
                print("[HWCtrl] pigpio resources released.")

            if self.ser and self.ser.is_open:
                self.ser.close()


def run_hardware_controller(shared_data):
    """Entry point function to run the hardware controller."""
    controller = HardwareController(shared_data)
    controller.run()