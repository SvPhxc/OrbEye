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
SCAN_PAN_SPEED_DPS = 360.0

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN = 0.0
SCAN_PAN_MAX = 360.0 - MICROSTEP_ANGLE
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1

# ==============================================================================
# --- SCAN CALIBRATION ---
SCAN_PAN_CALIBRATION_OFFSET_DEG = 0
# ==============================================================================


# ==============================================================================
# --- PD CONTROL TUNING GAINS ---
MAX_PAN_SPEED_DPS = 360.0
PAN_KP, PAN_KI, PAN_KD = 6.5, 0.0, 0.0005
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 6.5, 0.0, 0.0


# ==============================================================================


# ==============================================================================
# PID CONTROLLER CLASS
# ==============================================================================
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
        derivative = (error - self._last_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        self._last_error, self._last_time = error, time.monotonic()
        self._last_output = max(self.output_limits[0], min(self.output_limits[1], output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

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
                                     wrap_range=(0, 360))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.background_data_buffer = []

        # --- FIX: Initialize the attribute here ---
        self.scan_is_turning = False

        self.scan_phase = None  # Can be "homing", "scanning"

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
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        self.internal_pan_pos += (pan_velocity_dps * dt)
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            frequency = abs(pan_velocity_dps) / MICROSTEP_ANGLE
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(min(frequency, 250000)), 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        pulse_width = int(500 + (self.internal_tilt_pos / 0.09) + (36 / 0.09))
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def run(self):
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
            print("[HWCtrl] Hardware Controller process is running.")
            last_loop_time, current_state = time.monotonic(), "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: continue
                last_loop_time = time.monotonic()

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
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    if next_state == "BACKGROUND_SCAN":
                        print("[HWCtrl-SCAN] Initializing scan. Homing to 0 degrees...")
                        self.scan_phase = "homing"
                        self.scan_is_turning = False  # Reset turning state
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                if current_state == "IDLE":
                    pass

                elif current_state == "GOTO_POSITION" or current_state == "HF_TRACKING":
                    target_az = self.shared_data["target_azimuth"].value if current_state == "GOTO_POSITION" else \
                    self.shared_data["predicted_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value if current_state == "GOTO_POSITION" else \
                    self.shared_data["predicted_elevation"].value
                    current_wrapped_pos = self.internal_pan_pos % 360.0
                    self.pan_pid.set_setpoint(target_az)
                    self.tilt_pid.set_setpoint(target_el)
                    pan_vel = self.pan_pid.update(current_wrapped_pos)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)
                    pan_error = abs(self.pan_pid._last_error)
                    tilt_error = abs(self.tilt_pid._last_error)
                    target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG
                    if current_state == "GOTO_POSITION":
                        self.shared_data["target_reached"].value = target_reached

                elif current_state == "BACKGROUND_SCAN":
                    if self.scan_phase == "homing":
                        current_wrapped_pos = self.internal_pan_pos % 360.0
                        self.pan_pid.set_setpoint(0.0)
                        pan_vel = self.pan_pid.update(current_wrapped_pos)

                        pan_error = abs(self.pan_pid._last_error)
                        if pan_error < TARGET_REACHED_THRESHOLD_DEG and abs(pan_vel) < 1.0:
                            print("[HWCtrl-SCAN] Homing complete. Position zeroed.")
                            self.internal_pan_pos = 0.0
                            self.current_scan_el = SCAN_TILT_MAX
                            self.scan_pan_direction = 1
                            self.scan_phase = "scanning"

                    elif self.scan_phase == "scanning":
                        if not self.scan_is_turning:
                            pan_vel = SCAN_PAN_SPEED_DPS * self.scan_pan_direction
                            is_past_max = self.scan_pan_direction == 1 and self.internal_pan_pos >= SCAN_PAN_MAX
                            is_past_min = self.scan_pan_direction == -1 and self.internal_pan_pos <= SCAN_PAN_MIN
                            if is_past_max or is_past_min:
                                self.scan_is_turning = True
                                self.internal_pan_pos = SCAN_PAN_MAX if is_past_max else SCAN_PAN_MIN
                        if self.scan_is_turning:
                            pan_vel = 0
                            self.scan_pan_direction *= -1
                            self.current_scan_el -= SCAN_STEP_DEG
                            print(
                                f"[HWCtrl-SCAN] Row finished. New elevation: {self.current_scan_el:.1f} deg, Direction: {self.scan_pan_direction}")
                            if self.current_scan_el < SCAN_TILT_MIN:
                                print("[HWCtrl] BACKGROUND_SCAN finished.")
                                if self.background_data_buffer:
                                    np.save(self.shared_data["background_path"].value,
                                            np.array(self.background_data_buffer))
                                    self.background_data_buffer = []
                                self.shared_data["background_scan_active"].value = False
                            else:
                                self.scan_is_turning = False

                    self.tilt_pid.set_setpoint(self.current_scan_el)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                try:
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()
                        with self.shared_data["lidar_data"].get_lock():
                            self.shared_data["lidar_data"][:] = [dist, strength, ts]
                        if current_state == "BACKGROUND_SCAN" and self.scan_phase == "scanning" and not self.scan_is_turning:
                            current_log_pos = self.internal_pan_pos % 360.0
                            corrected_pan_pos = (current_log_pos + (
                                        self.scan_pan_direction * SCAN_PAN_CALIBRATION_OFFSET_DEG)) % 360.0
                            self.background_data_buffer.append(
                                [corrected_pan_pos, self.internal_tilt_pos, dist, strength])
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value = self.internal_pan_pos % 360.0
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos
                time.sleep(0.001)

        except Exception as e:
            import traceback
            print(f"[HWCtrl] CRITICAL ERROR: {e}");
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop();
                print("[HWCtrl] pigpio resources released.")
            if self.ser and self.ser.is_open:
                self.ser.close();
                print("[HWCtrl] Serial port closed.")


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()