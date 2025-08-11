# File: hardware_handler.py

import time
import serial
import pigpio
import threading
import queue
import numpy as np
import traceback

from acquisition import (
    generate_spiral_search_path,
    generate_refinement_path,
    check_lidar_for_target,
    populate_points_buffer,
    MIN_POINT_SEPARATION_S
)


# --- PIDController Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup_limit = anti_windup_limit
        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()
        self._last_output = 0

    def update(self, current_value):
        current_time = time.monotonic()
        dt = current_time - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if abs(error) > 180:  # Handle angle wrapping for pan
            error -= np.sign(error) * 360
        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        derivative = (error - self._last_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._last_error, self._last_time, self._last_output = error, current_time, output
        return output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral, self._last_error, self._last_time = 0, 0, time.monotonic()


# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 0.5
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_TILT_STEP_DEG = 1.5
MAX_PAN_SPEED_DPS = 450.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.1, 0.05
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 12.0, 0.1, 0.05
MANUAL_JOG_SPEED_DPS = 15.0


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=5)
        self.internal_pan_pos = self.shared_data["stepper_degrees"].value
        self.internal_tilt_pos = self.shared_data["servo_degrees"].value
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))
        self.search_path_generator = None
        self.acquired_points = []
        self.best_point_in_refine = None
        self.acquisition_start_time = 0
        self.scan_sub_state = "IDLE"
        self.bg_index = {}
        self.background_data_buffer = []
        self.scan_pan_direction = 1
        self.current_scan_el = SCAN_TILT_MAX

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    distance, strength = frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8)
                    if not self.lidar_queue.full():
                        self.lidar_queue.put_nowait((distance, strength, time.time()))
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set():
                    print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        pan_deg_change = pan_velocity_dps * dt
        if abs(pan_deg_change) > 0.001:
            frequency = abs(pan_velocity_dps) / MICROSTEP_ANGLE
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(frequency), 500000)
            self.internal_pan_pos += pan_deg_change
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        tilt_deg_change = tilt_velocity_dps * dt
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + tilt_deg_change))
        pulse_width = 500 + (self.internal_tilt_pos / 0.09)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))

    def _reset_acquisition_state(self):
        self.search_path_generator = None
        self.acquired_points = []
        self.best_point_in_refine = None
        self.acquisition_start_time = 0
        self.scan_sub_state = "IDLE"
        self.shared_data['points_count'].value = 0

    def run(self):
        try:
            self.pi = pigpio.pi()
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT);
            self.pi.set_servo_pulsewidth(SERVO_PIN, 1500)
            self.ser = serial.Serial(self.shared_data.get("lidar_port", "/dev/serial0"), 115200, timeout=0.1)
            lidar_thread = threading.Thread(target=self._lidar_reader_thread);
            lidar_thread.daemon = True;
            lidar_thread.start()
            print("[HWCtrl] Main loop running.");
            last_loop_time = time.monotonic();
            current_state = "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.01: time.sleep(0.005); continue
                last_loop_time = time.monotonic()

                if any([self.shared_data[k].value for k in ['tilt_up', 'tilt_down', 'pan_left', 'pan_right']]):
                    next_state = "MANUAL_JOG"
                elif self.shared_data["go_to_target"].value:
                    next_state = "GOTO_POSITION"
                elif self.shared_data["ekf_running"].value:
                    next_state = "HF_TRACKING"
                elif self.shared_data["acquire_points"].value:
                    next_state = "SEARCHING"
                elif self.shared_data["background_scan_active"].value:
                    next_state = "BACKGROUND_SCAN"
                else:
                    next_state = "IDLE"

                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    if next_state == "SEARCHING":
                        self._reset_acquisition_state();
                        seed_az, seed_el = self.internal_pan_pos % 360, self.internal_tilt_pos
                        self.search_path_generator = generate_spiral_search_path(seed_az, seed_el)
                        self.scan_sub_state = "COARSE_SEARCH";
                        self.acquisition_start_time = time.monotonic()
                        print(f"[HWCtrl-ACQ] Starting Coarse Search around Az={seed_az:.1f}, El={seed_el:.1f}")
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0
                if current_state == "MANUAL_JOG":
                    if self.shared_data['tilt_up'].value: tilt_vel = MANUAL_JOG_SPEED_DPS
                    if self.shared_data['tilt_down'].value: tilt_vel = -MANUAL_JOG_SPEED_DPS
                    if self.shared_data['pan_left'].value: pan_vel = -MANUAL_JOG_SPEED_DPS
                    if self.shared_data['pan_right'].value: pan_vel = MANUAL_JOG_SPEED_DPS

                elif current_state == "GOTO_POSITION":
                    target_az = self.shared_data["target_azimuth"].value;
                    target_el = self.shared_data["target_elevation"].value
                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    current_pan_norm = self.internal_pan_pos % 360
                    pan_error = ((target_az - current_pan_norm + 180) % 360) - 180;
                    tilt_error = target_el - self.internal_tilt_pos
                    if abs(pan_error) < TARGET_REACHED_THRESHOLD_DEG and abs(tilt_error) < TARGET_REACHED_THRESHOLD_DEG:
                        self.shared_data["go_to_target"].value = False;
                        pan_vel, tilt_vel = 0, 0
                        print(f"[HWCtrl] GoTo target ({target_az:.1f}, {target_el:.1f}) reached.")
                    else:
                        pan_vel = self.pan_pid.update(current_pan_norm);
                        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                elif current_state == "SEARCHING":
                    if time.monotonic() - self.acquisition_start_time > 45.0:
                        print("[HWCtrl-ACQ] Acquisition timed out.");
                        self.shared_data['acquire_points'].value = False;
                        continue

                    target_az, target_el = self.pan_pid.setpoint, self.tilt_pid.setpoint
                    pan_error = ((target_az - (self.internal_pan_pos % 360) + 180) % 360) - 180
                    target_reached = abs(pan_error) < TARGET_REACHED_THRESHOLD_DEG and abs(
                        target_el - self.internal_tilt_pos) < TARGET_REACHED_THRESHOLD_DEG

                    if target_reached:
                        time.sleep(0.04)  # Dwell
                        try:
                            dist, strength, ts = self.lidar_queue.get_nowait()
                            is_target = check_lidar_for_target(dist, strength, self.shared_data)
                            if self.scan_sub_state == "COARSE_SEARCH" and is_target:
                                print(f"[HWCtrl-ACQ] Coarse hit! Str: {strength}. Refining for P1.");
                                self.scan_sub_state = "REFINE_P1"
                                center_az, center_el = self.internal_pan_pos % 360, self.internal_tilt_pos
                                self.search_path_generator = generate_refinement_path(center_az, center_el);
                                self.best_point_in_refine = None
                            elif "REFINE" in self.scan_sub_state and is_target:
                                current_point = {'az': self.internal_pan_pos % 360, 'el': self.internal_tilt_pos,
                                                 'distance_m': dist / 100.0, 'strength': strength, 'timestamp': ts}
                                if self.best_point_in_refine is None or strength > self.best_point_in_refine[
                                    'strength']: self.best_point_in_refine = current_point
                        except queue.Empty:
                            pass

                        try:
                            next_az, next_el = next(self.search_path_generator)
                            self.pan_pid.set_setpoint(next_az);
                            self.tilt_pid.set_setpoint(next_el)
                        except StopIteration:
                            if "REFINE" in self.scan_sub_state and self.best_point_in_refine:
                                self.acquired_points.append(self.best_point_in_refine);
                                print(
                                    f"[HWCtrl-ACQ] {self.scan_sub_state} success. Best Str: {self.best_point_in_refine['strength']:.0f}")
                                if self.scan_sub_state == "REFINE_P1":
                                    time.sleep(MIN_POINT_SEPARATION_S);
                                    self.scan_sub_state = "REFINE_P2"
                                    self.search_path_generator = generate_refinement_path(
                                        self.best_point_in_refine['az'], self.best_point_in_refine['el']);
                                    self.best_point_in_refine = None
                                elif self.scan_sub_state == "REFINE_P2":
                                    time.sleep(MIN_POINT_SEPARATION_S);
                                    self.scan_sub_state = "REFINE_P3";
                                    p1, p2 = self.acquired_points
                                    dt_acq = p2['timestamp'] - p1['timestamp'];
                                    vel_az = (
                                        ((p2['az'] - p1['az'] + 180) % 360 - 180) / dt_acq if dt_acq > 0.01 else 0);
                                    vel_el = (p2['el'] - p1['el']) / dt_acq if dt_acq > 0.01 else 0
                                    pred_az, pred_el = (p2['az'] + vel_az * dt_acq) % 360, max(0, min(90, p2[
                                        'el'] + vel_el * dt_acq))
                                    self.search_path_generator = generate_refinement_path(pred_az, pred_el);
                                    self.best_point_in_refine = None
                                elif self.scan_sub_state == "REFINE_P3":
                                    print("[HWCtrl-ACQ] Complete! 3 points found.");
                                    populate_points_buffer(self.shared_data, self.acquired_points)
                                    self.shared_data['acquire_points'].value = False;
                                    self.shared_data['ekf_start'].value = True
                            else:
                                print("[HWCtrl-ACQ] Search/refine failed."); self.shared_data[
                                    'acquire_points'].value = False

                    pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos % 360), self.tilt_pid.update(
                        self.internal_tilt_pos)

                elif current_state == "HF_TRACKING":
                    self.pan_pid.set_setpoint(self.shared_data["predicted_azimuth"].value);
                    self.tilt_pid.set_setpoint(self.shared_data["predicted_elevation"].value)
                    pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos % 360), self.tilt_pid.update(
                        self.internal_tilt_pos)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)
                with self.shared_data["lidar_data"].get_lock():
                    self.shared_data["lidar_data"][:] = [0, 0, 0]  # clear old data
                try:
                    dist, strength, ts = self.lidar_queue.get_nowait()
                    with self.shared_data["lidar_data"].get_lock():
                        self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value = self.internal_pan_pos % 360
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

        except Exception:
            print(f"[HWCtrl] CRITICAL ERROR: {traceback.format_exc()}")
        finally:
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join(timeout=1)
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0);
                self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()
            print("[HWCtrl] Shutdown complete.")


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()