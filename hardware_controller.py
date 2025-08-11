# ==============================================================================
# hardware_controller.py (Modulo Bug Fixed Version)
# ==============================================================================

import time
import serial
import pigpio
import threading
import queue
import numpy as np


# --- PIDController Class (self-contained) ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd;
        self.setpoint = setpoint;
        self.output_limits = output_limits;
        self.anti_windup_limit = anti_windup_limit
        self._integral = 0;
        self._last_error = 0;
        self._last_time = time.monotonic();
        self._last_output = 0

    def update(self, current_value):
        current_time = time.monotonic();
        dt = current_time - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value;
        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        derivative = (error - self._last_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._last_error, self._last_time, self._last_output = error, current_time, output
        return output

    def set_setpoint(self, new_setpoint): self.setpoint = new_setpoint

    def reset(self): self._integral, self._last_error, self._last_time = 0, 0, time.monotonic()


# --- Hardware & Scan Constants ---
SERVO_PIN = 13;
STEPPER_PULSE_PIN = 19;
STEPPER_DIR_PIN = 3;
STEPPER_ENABLE_PIN = 4;
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625;
TARGET_REACHED_THRESHOLD_DEG = 0.1
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360;
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90;
SCAN_TILT_STEP_DEG = 1.0

# --- PID & Speed Tuning ---
MAX_PAN_SPEED_DPS = 450.0
PAN_KP, PAN_KI, PAN_KD = 6.0, 0.05, 0.15
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 6.2, 0.1, 0.2


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data;
        self.pi = None;
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=1)
        self.internal_pan_pos = self.shared_data["stepper_degrees"].value
        self.internal_tilt_pos = self.shared_data["servo_degrees"].value
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))
        self.current_scan_az = SCAN_PAN_MIN;
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1;
        self.background_data_buffer = []
        self.scan_sub_state = "IDLE"

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59');
                frame = self.ser.read(7)
                if len(frame) == 7:
                    distance, strength = frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8)
                    try:
                        self.lidar_queue.put_nowait((distance, strength, time.time()))
                    except queue.Full:
                        pass
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        # Pan Stepper
        pan_deg_change = pan_velocity_dps * dt
        if abs(pan_deg_change) > 0.001:
            pan_steps_to_move = round(abs(pan_deg_change) / MICROSTEP_ANGLE)
            if pan_steps_to_move > 0:
                self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
                frequency = int(min(pan_steps_to_move / dt, 300000))
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, frequency, 500000)
            else:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

            # --- FIX #1: Remove the modulo operator from the internal position ---
            self.internal_pan_pos += pan_deg_change
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Tilt Servo
        tilt_deg_change = tilt_velocity_dps * dt
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + tilt_deg_change))
        pulse_width = 500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))

    def _update_scan_for_next_row(self):
        """Prepares the state for the next row sweep."""
        self.current_scan_el -= SCAN_TILT_STEP_DEG
        self.scan_pan_direction *= -1
        return self.current_scan_el >= SCAN_TILT_MIN

    def run(self):
        try:
            self.pi = pigpio.pi();
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")
            self.ser = serial.Serial(self.shared_data["lidar_port"], 115200, timeout=0.1)
            set_rate_command = bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E])
            ser.write(set_rate_command)
            lidar_thread = threading.Thread(target=self._lidar_reader_thread);
            lidar_thread.daemon = True;
            lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")
            last_loop_time = time.monotonic();
            current_state = "IDLE"
            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: time.sleep(0.001); continue
                last_loop_time = time.monotonic()
                if self.shared_data["lidar_track_mode_active"].value:
                    next_state = "HF_TRACKING"
                elif self.shared_data["go_to_target"].value:
                    next_state = "GOTO_POSITION"
                elif self.shared_data["search_mode_active"].value:
                    next_state = "SEARCHING"
                elif self.shared_data["background_scan_active"].value:
                    next_state = "BACKGROUND_SCAN"
                else:
                    next_state = "IDLE"
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    if next_state == "BACKGROUND_SCAN":
                        self.current_scan_az = self.internal_pan_pos  # Start sweep from current position
                        self.current_scan_el = SCAN_TILT_MAX
                        self.scan_pan_direction = 1;
                        self.scan_sub_state = "MOVING_TO_ROW_START"
                    elif next_state == "IDLE":
                        self.scan_sub_state = "IDLE"
                    current_state = next_state
                pan_vel, tilt_vel = 0, 0
                if current_state == "HF_TRACKING":
                    self.pan_pid.set_setpoint(self.shared_data["predicted_azimuth"].value);
                    self.tilt_pid.set_setpoint(self.shared_data["predicted_elevation"].value)
                    pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                        self.internal_tilt_pos)
                elif current_state == "GOTO_POSITION":
                    self.pan_pid.set_setpoint(self.shared_data["target_azimuth"].value);
                    self.tilt_pid.set_setpoint(self.shared_data["target_elevation"].value)
                    target_reached = abs(
                        self.pan_pid.setpoint - self.internal_pan_pos) < TARGET_REACHED_THRESHOLD_DEG and abs(
                        self.tilt_pid.setpoint - self.internal_tilt_pos) < TARGET_REACHED_THRESHOLD_DEG
                    self.shared_data["target_reached"].value = target_reached
                    if not target_reached:
                        pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                            self.internal_tilt_pos)
                    else:
                        self.shared_data["go_to_target"].value = False
                elif current_state == "BACKGROUND_SCAN":
                    if self.scan_sub_state == "MOVING_TO_ROW_START":
                        self.pan_pid.set_setpoint(self.internal_pan_pos);
                        self.tilt_pid.set_setpoint(self.current_scan_el)
                        target_reached = abs(
                            self.tilt_pid.setpoint - self.internal_tilt_pos) < TARGET_REACHED_THRESHOLD_DEG
                        if target_reached:
                            print(f"[HWCtrl-Scan] Reached start of row El={self.current_scan_el:.1f}. Starting sweep.")
                            self.scan_sub_state = "SWEEPING_ROW"
                        else:
                            pan_vel, tilt_vel = 0, self.tilt_pid.update(self.internal_tilt_pos)
                    elif self.scan_sub_state == "SWEEPING_ROW":
                        pan_vel = MAX_PAN_SPEED_DPS * self.scan_pan_direction;
                        tilt_vel = 0
                        try:
                            dist, strength, _ = self.lidar_queue.get_nowait()
                            self.background_data_buffer.append(
                                [self.internal_pan_pos % 360, self.internal_tilt_pos, dist, strength])
                        except queue.Empty:
                            pass
                        pan_finished = (self.scan_pan_direction == 1 and self.internal_pan_pos >= SCAN_PAN_MAX) or \
                                       (self.scan_pan_direction == -1 and self.internal_pan_pos <= SCAN_PAN_MIN)
                        if pan_finished:
                            print(f"[HWCtrl-Scan] Finished sweep at El={self.current_scan_el:.1f}.")
                            pan_vel = 0
                            if self._update_scan_for_next_row():
                                self.scan_sub_state = "MOVING_TO_ROW_START"
                            else:
                                print("[HWCtrl] Background scan finished."); self.shared_data[
                                    "background_scan_active"].value = False
                self._execute_motor_commands(pan_vel, tilt_vel, dt)
                try:
                    dist, strength, ts = self.lidar_queue.get_nowait()
                    with self.shared_data["lidar_data"].get_lock():
                        self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass

                # --- FIX #2: Apply modulo only when reporting to shared memory ---
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos % 360
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

                if self.shared_data["save_background_trigger"].value:
                    if self.background_data_buffer:
                        print(f"[HWCtrl] Saving {len(self.background_data_buffer)} points...")
                        np.save(self.shared_data["background_path"], np.array(self.background_data_buffer));
                        self.background_data_buffer = []
                    self.shared_data["save_background_trigger"].value = False
                time.sleep(0.001)
        except Exception as e:
            import traceback; print(f"[HWCtrl] CRITICAL ERROR: {e}"); traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join(timeout=1)
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1);
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0);
                self.pi.stop()
                print("[HWCtrl] pigpio resources released.")
            if self.ser and self.ser.is_open: self.ser.close()


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()