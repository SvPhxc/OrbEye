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
TARGET_REACHED_THRESHOLD_DEG = 0.4

# --- IMPROVED SCAN PARAMETERS ---
# Constant velocity scanning with precise timing
SCAN_VELOCITY_DPS = 720.0  # Constant scan velocity in degrees per second
SCAN_SAMPLE_RATE_HZ = 1000  # LiDAR sampling rate (samples per second)
SCAN_DEGREES_PER_SAMPLE = SCAN_VELOCITY_DPS / SCAN_SAMPLE_RATE_HZ  # 12 degrees per sample at 120dps/10Hz

# Define the boundaries and resolution for scanning
# IMPORTANT: Limited range to prevent cable damage - no full 360° rotation!
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360  # Safe range with cable protection margins
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1

# Scan calibration offset
SCAN_PAN_CALIBRATION_OFFSET_DEG = 0

# PID Constants (unchanged)
MAX_PAN_SPEED_DPS = 720.0
PAN_KP, PAN_KI, PAN_KD = 6.5, 0.0001, 0.0005
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 6.5, 0.000, 0.000


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-90, 90), anti_windup_limit=10, wrap_range=None):
        self.Kp, self.Ki, self.Kd, self.setpoint, self.output_limits, self.anti_windup_limit, self.wrap_range = Kp, Ki, Kd, setpoint, output_limits, anti_windup_limit, wrap_range
        self._integral, self._last_error, self._last_output, self._last_time = 0, 0, 0, time.monotonic()

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            error = (error + range_width / 2) % range_width - range_width / 2
        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * (error - self._last_error) / dt)
        self._last_error, self._last_time, self._last_output = error, time.monotonic(), max(self.output_limits[0],
                                                                                            min(self.output_limits[1],
                                                                                                output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def get_setpoint(self):
        return self.setpoint

    def reset(self):
        self._integral, self._last_error = 0, 0
        self._last_time = time.monotonic()


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi, self.ser = None, None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=10)
        self.internal_pan_pos = shared_data["stepper_degrees"].value
        self.internal_tilt_pos = shared_data["servo_degrees"].value
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS),
                                     anti_windup_limit=40, wrap_range=(0, 360))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))

        # IMPROVED BACKGROUND SCAN STATE VARIABLES
        self.background_data_buffer = []
        self.scan_state = "IDLE"  # IDLE, POSITIONING, SCANNING, MOVING_TO_NEXT_ROW, FINISHED
        self.current_scan_elevation = SCAN_TILT_MAX
        self.scan_direction = 1  # 1 for forward (0->360), -1 for backward (360->0)
        self.scan_start_time = 0
        self.scan_start_position = 0
        self.expected_scan_position = 0
        self.last_sample_time = 0
        self.row_start_time = 0
        self.samples_this_row = 0

        # Pre-calculate scan parameters for accuracy
        self.scan_range_deg = SCAN_PAN_MAX - SCAN_PAN_MIN  # Total scan range per row
        self.scan_duration_per_row = self.scan_range_deg / SCAN_VELOCITY_DPS  # Time to scan one row
        self.expected_samples_per_row = int(self.scan_duration_per_row * SCAN_SAMPLE_RATE_HZ)

        print(f"[HWCtrl] Scan parameters: {SCAN_VELOCITY_DPS} dps, {SCAN_SAMPLE_RATE_HZ} Hz")
        print(f"[HWCtrl] Scan range: {SCAN_PAN_MIN}° to {SCAN_PAN_MAX}° ({self.scan_range_deg}°)")
        print(
            f"[HWCtrl] Expected: {self.scan_duration_per_row:.2f}s per row, {self.expected_samples_per_row} samples per row")

    def _get_shortest_pan_error(self, setpoint, current_value):
        error = setpoint - current_value
        return (error + 180) % 360 - 180

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

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        # Update internal position tracking
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))

        # Execute pan motor commands
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            pulse_freq = int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, 250000))
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, pulse_freq, 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Execute tilt motor commands
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(500 + (self.internal_tilt_pos / 0.09) + (36 / 0.09)))

    def _get_expected_scan_position(self, current_time):
        """Calculate where the scanner should be based on constant velocity motion"""
        if self.scan_state != "SCANNING":
            return self.internal_pan_pos

        elapsed_time = current_time - self.scan_start_time
        expected_displacement = SCAN_VELOCITY_DPS * elapsed_time * self.scan_direction
        expected_pos = self.scan_start_position + expected_displacement

        # Clamp to safe range instead of wrapping
        expected_pos = max(SCAN_PAN_MIN, min(SCAN_PAN_MAX, expected_pos))
        return expected_pos

    def _should_sample_lidar(self, current_time):
        """Determine if we should take a LiDAR sample based on timing"""
        if self.scan_state != "SCANNING":
            return False

        # Sample based on time intervals for consistent spacing
        time_since_last_sample = current_time - self.last_sample_time
        return time_since_last_sample >= (1.0 / SCAN_SAMPLE_RATE_HZ)

    def _handle_background_scan_state_machine(self, current_time, dt):
        """Improved background scan state machine with constant velocity and precise positioning"""
        pan_vel, tilt_vel = 0, 0

        if self.scan_state == "IDLE":
            # Initialize scan - move to starting position
            self.current_scan_elevation = SCAN_TILT_MAX
            self.scan_direction = 1
            self.scan_state = "POSITIONING"
            self.pan_pid.set_setpoint(SCAN_PAN_MIN)
            self.tilt_pid.set_setpoint(self.current_scan_elevation)
            print(
                f"[HWCtrl-SCAN] Starting background scan. Moving to initial position: az={SCAN_PAN_MIN}, el={self.current_scan_elevation}")

        elif self.scan_state == "POSITIONING":
            # Move to scan start position
            self.pan_pid.set_setpoint(SCAN_PAN_MIN)
            self.tilt_pid.set_setpoint(self.current_scan_elevation)

            pan_vel = self.pan_pid.update(self.internal_pan_pos)
            tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

            # Check if we've reached the starting position
            pan_error = abs(self._get_shortest_pan_error(SCAN_PAN_MIN, self.internal_pan_pos))
            tilt_error = abs(self.current_scan_elevation - self.internal_tilt_pos)

            if pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG:
                # Start the scanning row
                self.scan_state = "SCANNING"
                self.scan_start_time = current_time
                self.scan_start_position = self.internal_pan_pos
                self.last_sample_time = current_time
                self.samples_this_row = 0
                self.row_start_time = current_time
                print(
                    f"[HWCtrl-SCAN] Starting scan row at elevation {self.current_scan_elevation:.1f}°, direction {self.scan_direction}")

        elif self.scan_state == "SCANNING":
            # Constant velocity scanning
            target_pan = self._get_expected_scan_position(current_time)

            # Use direct velocity control instead of PID for constant motion
            pan_vel = SCAN_VELOCITY_DPS * self.scan_direction

            # Keep tilt position steady
            self.tilt_pid.set_setpoint(self.current_scan_elevation)
            tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

            # Check if we've completed this row
            scan_progress = (current_time - self.scan_start_time) * SCAN_VELOCITY_DPS

            if self.scan_direction == 1 and (self.scan_start_position + scan_progress) >= SCAN_PAN_MAX:
                # Completed forward scan
                self.scan_state = "MOVING_TO_NEXT_ROW"
                self.scan_direction = -1
                print(f"[HWCtrl-SCAN] Row completed (forward). Samples taken: {self.samples_this_row}")

            elif self.scan_direction == -1 and (self.scan_start_position + scan_progress) <= SCAN_PAN_MIN:
                # Completed backward scan
                self.scan_state = "MOVING_TO_NEXT_ROW"
                self.scan_direction = 1
                print(f"[HWCtrl-SCAN] Row completed (backward). Samples taken: {self.samples_this_row}")

        if self.scan_state == "IDLE":
            # Initialize scan - move to starting position
            self.current_scan_elevation = SCAN_TILT_MAX
            self.scan_direction = 1  # Always start scanning forward
            self.scan_state = "POSITIONING"
            start_pos = SCAN_PAN_MIN  # Always start from minimum position
            self.pan_pid.set_setpoint(start_pos)
            self.tilt_pid.set_setpoint(self.current_scan_elevation)
            print(
                f"[HWCtrl-SCAN] Starting background scan. Moving to initial position: az={start_pos}, el={self.current_scan_elevation}")

        elif self.scan_state == "POSITIONING":
            # Move to scan start position for current row
            target_start = SCAN_PAN_MIN if self.scan_direction == 1 else SCAN_PAN_MAX
            self.pan_pid.set_setpoint(target_start)
            self.tilt_pid.set_setpoint(self.current_scan_elevation)

            pan_vel = self.pan_pid.update(self.internal_pan_pos)
            tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

            # Check if we've reached the starting position
            pan_error = abs(self._get_shortest_pan_error(target_start, self.internal_pan_pos))
            tilt_error = abs(self.current_scan_elevation - self.internal_tilt_pos)

            if pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG:
                # Start the scanning row
                self.scan_state = "SCANNING"
                self.scan_start_time = current_time
                self.scan_start_position = self.internal_pan_pos
                self.last_sample_time = current_time
                self.samples_this_row = 0
                self.row_start_time = current_time
                scan_end = SCAN_PAN_MAX if self.scan_direction == 1 else SCAN_PAN_MIN
                print(f"[HWCtrl-SCAN] Starting scan row at elevation {self.current_scan_elevation:.1f}°")
                print(f"[HWCtrl-SCAN] Direction: {self.scan_direction} (from {target_start}° to {scan_end}°)")

        elif self.scan_state == "SCANNING":
            # Constant velocity scanning within safe range
            target_pan = self._get_expected_scan_position(current_time)

            # Use direct velocity control for constant motion
            pan_vel = SCAN_VELOCITY_DPS * self.scan_direction

            # Keep tilt position steady
            self.tilt_pid.set_setpoint(self.current_scan_elevation)
            tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

            # Check if we've reached the end of the safe scan range
            if self.scan_direction == 1 and self.internal_pan_pos >= SCAN_PAN_MAX:
                # Reached maximum safe position
                self.scan_state = "MOVING_TO_NEXT_ROW"
                self.scan_direction = -1  # Next row will scan backward
                print(
                    f"[HWCtrl-SCAN] Row completed (forward to {SCAN_PAN_MAX}°). Samples taken: {self.samples_this_row}")

            elif self.scan_direction == -1 and self.internal_pan_pos <= SCAN_PAN_MIN:
                # Reached minimum safe position
                self.scan_state = "MOVING_TO_NEXT_ROW"
                self.scan_direction = 1  # Next row will scan forward
                print(
                    f"[HWCtrl-SCAN] Row completed (backward to {SCAN_PAN_MIN}°). Samples taken: {self.samples_this_row}")

        elif self.scan_state == "MOVING_TO_NEXT_ROW":
            # Move to next elevation and prepare for next row
            self.current_scan_elevation -= SCAN_STEP_DEG

            if self.current_scan_elevation < SCAN_TILT_MIN:
                # Scan completely finished
                self.scan_state = "FINISHED"
                print("[HWCtrl-SCAN] Background scan completed!")
                print(
                    f"[HWCtrl-SCAN] Total scan coverage: {SCAN_PAN_MIN}° to {SCAN_PAN_MAX}° ({self.scan_range_deg}°) at each elevation")

                # Auto-save data
                if self.background_data_buffer:
                    print(f"[HWCtrl] Auto-saving {len(self.background_data_buffer)} background scan points...")
                    try:
                        np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer))
                        print(f"[HWCtrl] Data saved to {self.shared_data['background_path'].value}")
                        self.background_data_buffer = []
                    except Exception as e:
                        print(f"[HWCtrl] ERROR saving background data: {e}")

                self.shared_data["background_scan_active"].value = False
            else:
                # Prepare for next row - direction is already set from previous row completion
                # Move to the appropriate starting position for the next scan direction
                target_start = SCAN_PAN_MIN if self.scan_direction == 1 else SCAN_PAN_MAX
                self.pan_pid.set_setpoint(target_start)
                self.tilt_pid.set_setpoint(self.current_scan_elevation)
                self.scan_state = "POSITIONING"
                print(f"[HWCtrl-SCAN] Moving to next row: elevation {self.current_scan_elevation:.1f}°")
                print(f"[HWCtrl-SCAN] Next row will scan {'forward' if self.scan_direction == 1 else 'backward'}")

            # Use PID to move to position during transitions
            pan_vel = self.pan_pid.update(self.internal_pan_pos)
            tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

        elif self.scan_state == "FINISHED":
            # Scan is done, stop all motion
            pan_vel, tilt_vel = 0, 0

        return pan_vel, tilt_vel

    def _process_lidar_data_during_scan(self, current_time):
        """Process LiDAR data with improved sampling strategy"""
        try:
            while not self.lidar_queue.empty():
                dist, strength, ts = self.lidar_queue.get_nowait()

                # Update shared data
                with self.shared_data["lidar_data"].get_lock():
                    self.shared_data["lidar_data"][:] = [dist, strength, ts]

                # Only log data during active scanning (constant velocity motion)
                if self.scan_state == "SCANNING" and self._should_sample_lidar(current_time):
                    # Calculate precise position based on time and velocity
                    elapsed_scan_time = current_time - self.scan_start_time
                    precise_pan_displacement = SCAN_VELOCITY_DPS * elapsed_scan_time * self.scan_direction
                    precise_pan_pos = self.scan_start_position + precise_pan_displacement

                    # Clamp to safe range instead of wrapping
                    precise_pan_pos = max(SCAN_PAN_MIN, min(SCAN_PAN_MAX, precise_pan_pos))

                    # Apply calibration offset
                    corrected_pan_pos = precise_pan_pos + (self.scan_direction * SCAN_PAN_CALIBRATION_OFFSET_DEG)
                    # Keep corrected position within safe bounds too
                    corrected_pan_pos = max(SCAN_PAN_MIN, min(SCAN_PAN_MAX, corrected_pan_pos))

                    # Log the data point
                    self.background_data_buffer.append([
                        corrected_pan_pos,
                        self.current_scan_elevation,
                        dist,
                        strength,
                        current_time  # Add timestamp for debugging
                    ])

                    self.last_sample_time = current_time
                    self.samples_this_row += 1

                    # Debug output every 50 samples
                    if self.samples_this_row % 50 == 0:
                        print(f"[HWCtrl-SCAN] Row progress: {self.samples_this_row} samples, "
                              f"pos: {corrected_pan_pos:.1f}°, el: {self.current_scan_elevation:.1f}°")

        except queue.Empty:
            pass

    def run(self):
        try:
            # Initialize hardware
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpio connection failed.")

            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")

            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time = time.monotonic()
            current_mode = "IDLE"

            while not self.shared_data["shutdown"].value:
                current_time = time.monotonic()
                dt = current_time - last_loop_time
                if dt <= 0.001:
                    continue
                last_loop_time = current_time

                # Determine operating mode
                if self.shared_data["background_scan_active"].value:
                    next_mode = "BACKGROUND_SCAN"
                elif self.shared_data["go_to_target"].value:
                    next_mode = "GOTO_POSITION"
                elif self.shared_data["lidar_track_mode_active"].value:
                    next_mode = "HF_TRACKING"
                else:
                    next_mode = "IDLE"

                # Handle mode transitions
                if next_mode != current_mode:
                    print(f"[HWCtrl] Mode change: {current_mode} -> {next_mode}")
                    self.pan_pid.reset()
                    self.tilt_pid.reset()

                    if next_mode == "BACKGROUND_SCAN":
                        self.scan_state = "IDLE"  # Reset scan state machine

                    current_mode = next_mode

                # Execute mode-specific logic
                pan_vel, tilt_vel = 0, 0

                if current_mode == "IDLE":
                    pass

                elif current_mode == "GOTO_POSITION" or current_mode == "HF_TRACKING":
                    target_az = (self.shared_data["target_azimuth"].value if current_mode == "GOTO_POSITION"
                                 else self.shared_data["predicted_azimuth"].value)
                    target_el = (self.shared_data["target_elevation"].value if current_mode == "GOTO_POSITION"
                                 else self.shared_data["predicted_elevation"].value)

                    self.pan_pid.set_setpoint(target_az)
                    self.tilt_pid.set_setpoint(target_el)

                    pan_vel = self.pan_pid.update(self.internal_pan_pos)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                    # Check if target reached (for GOTO mode)
                    if current_mode == "GOTO_POSITION":
                        pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                        tilt_error = abs(target_el - self.internal_tilt_pos)
                        self.shared_data["target_reached"].value = (pan_error < TARGET_REACHED_THRESHOLD_DEG and
                                                                    tilt_error < TARGET_REACHED_THRESHOLD_DEG)

                elif current_mode == "BACKGROUND_SCAN":
                    # Execute improved background scan state machine
                    pan_vel, tilt_vel = self._handle_background_scan_state_machine(current_time, dt)

                # Execute motor commands
                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Process LiDAR data
                self._process_lidar_data_during_scan(current_time)

                # Update shared position data
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

                time.sleep(0.001)  # 1kHz control loop

        except Exception as e:
            import traceback
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()

            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0)
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop()
                print("[HWCtrl] pigpio resources released.")

            if self.ser and self.ser.is_open:
                self.ser.close()


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()