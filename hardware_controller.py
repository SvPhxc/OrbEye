# hardware_controller.py

import time
import serial
import pigpio
import threading
import queue
import numpy as np
import traceback


def _shortest_angular_delta(target, current):
    """Calculates the shortest angle between two points, handling wrap-around."""
    return ((target - current + 540.0) % 360.0) - 180.0


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint, self.output_limits, self.anti_windup_limit = setpoint, output_limits, anti_windup_limit
        self._integral, self._last_error, self._last_time, self._last_output = 0, 0, time.monotonic(), 0

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if abs(error) > 180: error = _shortest_angular_delta(self.setpoint, current_value)
        self._integral += error * dt;
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        derivative = (error - self._last_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._last_error, self._last_time, self._last_output = error, time.monotonic(), output
        return output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral, self._last_error, self._last_time = 0, 0, time.monotonic()


# --- Hardware & Motor Constants ---
SERVO_PIN = 13;
STEPPER_PULSE_PIN = 19;
STEPPER_DIR_PIN = 3;
STEPPER_ENABLE_PIN = 4;
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625  # The angle of a single pulse
TARGET_REACHED_THRESHOLD_DEG = 0.1
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90

# These PID values are tuned to prevent commanding accelerations that the motor cannot handle,
# which is the primary cause of skipped steps.
MAX_PAN_SPEED_DPS = 350.0
PAN_KP, PAN_KI, PAN_KD = 3.5, 0.1, 1.2

MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 6.0, 0.0, 0.0


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None;
        self.ser = None;
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=1)
        self.internal_pan_pos = self.shared_data["stepper_degrees"].value
        self.internal_tilt_pos = self.shared_data["servo_degrees"].value
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))

    def _lidar_reader_thread(self):
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59');
                frame = self.ser.read(7)
                if len(frame) == 7: self.lidar_queue.put_nowait(
                    (frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8), time.time()))
            except (serial.SerialException, OSError, queue.Full):
                pass

    def _execute_motor_commands(self, pan_vel_dps, tilt_vel_dps, dt):
        if abs(pan_vel_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_vel_dps < 0 else 1)
            freq = int(abs(pan_vel_dps) / MICROSTEP_ANGLE)
            safe_freq = min(freq, 300000)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, safe_freq, 500000)

            # --- CRITICAL FIX: Calculate position based on actual pulses commanded ---
            pulses_in_dt = safe_freq * dt
            angle_change_deg = pulses_in_dt * MICROSTEP_ANGLE

            if pan_vel_dps > 0:
                self.internal_pan_pos = (self.internal_pan_pos + angle_change_deg) % 360.0
            else:
                self.internal_pan_pos = (self.internal_pan_pos - angle_change_deg + 360.0) % 360.0
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        if abs(tilt_vel_dps) > 0.1:
            self.internal_tilt_pos += tilt_vel_dps * dt
            self.internal_tilt_pos = max(SCAN_TILT_MIN, min(SCAN_TILT_MAX, self.internal_tilt_pos))
            pulse_width = 500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)
            self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))

    def run(self):
        try:
            self.pi = pigpio.pi();
            assert self.pi.connected
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            self.ser = serial.Serial(self.shared_data["lidar_port"], 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            lidar_thread = threading.Thread(target=self._lidar_reader_thread);
            lidar_thread.daemon = True;
            lidar_thread.start()
            print("[HWCtrl] Process running.");
            last_loop_time = time.monotonic();
            current_state = "IDLE"
            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time;
                last_loop_time = time.monotonic()
                if dt <= 0.001: time.sleep(0.001); continue

                is_acquiring = self.shared_data["acquirer_status"].value == 1
                is_tracking = self.shared_data["lidar_track_mode_active"].value
                is_moving = self.shared_data["go_to_target"].value

                next_state = "GOTO_POSITION" if is_moving or is_acquiring or is_tracking else "IDLE"

                if next_state != current_state:
                    self.pan_pid.reset();
                    self.tilt_pid.reset();
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0
                if current_state == "GOTO_POSITION":
                    if is_tracking and self.shared_data["ekf_initialized"].value:
                        target_az = self.shared_data["predicted_azimuth"].value
                        target_el = self.shared_data["predicted_elevation"].value
                    else:
                        target_az = self.shared_data["target_azimuth"].value
                        target_el = self.shared_data["target_elevation"].value

                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    pan_err = _shortest_angular_delta(target_az, self.internal_pan_pos)
                    tilt_err = target_el - self.internal_tilt_pos

                    if abs(pan_err) < TARGET_REACHED_THRESHOLD_DEG and abs(tilt_err) < TARGET_REACHED_THRESHOLD_DEG:
                        self.shared_data["target_reached"].value = True
                        if not is_tracking and not is_acquiring: self.shared_data["go_to_target"].value = False
                    else:
                        self.shared_data["target_reached"].value = False
                        pan_vel = self.pan_pid.update(self.internal_pan_pos)
                        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)
                try:
                    d, s, ts = self.lidar_queue.get_nowait()
                    with self.shared_data["lidar_data"].get_lock():
                        self.shared_data["lidar_data"][:] = [d, s, ts]
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos
        except Exception as e:
            print(f"[HWCtrl] CRITICAL ERROR: {e}");
            traceback.print_exc()
        finally:
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join(timeout=1)
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0);
                self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()
            print("[HWCtrl] Shut down.")


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data);
    controller.run()