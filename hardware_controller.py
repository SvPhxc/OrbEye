# hardware_controller.py

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
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 1.5
SCAN_PAN_SPEED_DPS = 600.0
SCAN_TURNAROUND_DEG = 1.0

# --- Boundary and Resolution Constants ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 720.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.01, 0.05
MAX_TILT_SPEED_DPS = 720.0
TILT_KP, TILT_KI, TILT_KD = 8.0, 0.02, 0.05

# --- Acquisition Scan Constants ---
ACQUIRE_PAN_MIN, ACQUIRE_PAN_MAX = 45, 315
ACQUIRE_TILT_MIN, ACQUIRE_TILT_MAX = 10, 80
ACQUIRE_PAN_SPEED_DPS = 30.0
ACQUIRE_TILT_STEP_DEG = 7.0
ACQUIRE_MIN_SEPARATION_DEG = 15.0
ACQUIRE_POINTS_NEEDED = 2


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20, wrap_range=None):
        self.Kp, self.Ki, self.Kd, self.setpoint, self.output_limits, self.anti_windup_limit, self.wrap_range = Kp, Ki, Kd, setpoint, output_limits, anti_windup_limit, wrap_range
        self._integral, self._last_error, self._last_output, self._last_time = 0, 0, 0, time.monotonic()

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0];
            error = (error + range_width / 2) % range_width - range_width / 2
        self._integral += error * dt;
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * (error - self._last_error) / dt)
        self._last_error, self.last_time, self._last_output = error, time.monotonic(), max(self.output_limits[0],
                                                                                           min(self.output_limits[1],
                                                                                               output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral, self._last_error, self.last_time = 0, 0, time.monotonic()

    def get_setpoint(self):
        return self.setpoint


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi, self.ser = None, None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=20)
        self.internal_pan_pos = shared_data["stepper_degrees"].value
        self.internal_tilt_pos = shared_data["servo_degrees"].value
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS),
                                     wrap_range=(0, 360))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))

        self.hf_tracking_substate = "IDLE";
        self.grid_scan_index = 0;
        self.grid_scan_points = [];
        self.grid_scan_center_pos = (0, 0);
        self.GRID_STEP_DEG = 1.5;
        self.GRID_DWELL_TIME = 0.025;
        self.GRID_OFFSETS = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)];
        self.grid_point_arrival_time = 0;
        self.ARRIVAL_THRESHOLD_DEG = 1.0;

        self.acquire_scan_el = ACQUIRE_TILT_MAX
        self.acquire_pan_direction = 1
        self.last_acquired_point_azel = None

    def _get_shortest_pan_error(self, setpoint, current_value):
        error = setpoint - current_value
        return (error + 180) % 360 - 180

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59');
                frame = self.ser.read(7)
                if len(frame) == 7:
                    try:
                        self.lidar_queue.put_nowait(
                            (frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8), time.time()))
                    except queue.Full:
                        pass
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error."); break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_vel, tilt_vel, dt):
        self.internal_pan_pos = (self.internal_pan_pos + (pan_vel * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_vel * dt)))
        if abs(pan_vel) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_vel < 0 else 1)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(min(abs(pan_vel) / MICROSTEP_ANGLE, 250000)), 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)))

    def _handle_acquisition(self, dt):
        target_az = self.pan_pid.get_setpoint() + (ACQUIRE_PAN_SPEED_DPS * self.acquire_pan_direction * dt)
        if target_az >= ACQUIRE_PAN_MAX and self.acquire_pan_direction == 1:
            self.acquire_pan_direction = -1; self.acquire_scan_el -= ACQUIRE_TILT_STEP_DEG
        elif target_az <= ACQUIRE_PAN_MIN and self.acquire_pan_direction == -1:
            self.acquire_pan_direction = 1; self.acquire_scan_el -= ACQUIRE_TILT_STEP_DEG
        if self.acquire_scan_el < ACQUIRE_TILT_MIN: self.shared_data["acquirer_status"].value = 3; self.shared_data[
            "acquire_points"].value = False; return 0, 0
        self.pan_pid.set_setpoint(target_az);
        self.tilt_pid.set_setpoint(self.acquire_scan_el)
        try:
            dist_cm, strength, ts = self.lidar_queue.get_nowait();
            min_m, max_m = self.shared_data["lidar_acceptance_range"][:]
            if (min_m <= dist_cm / 100.0 <= max_m) and strength > 100:
                current_azel = (self.internal_pan_pos, self.internal_tilt_pos)
                if self.last_acquired_point_azel is None or np.sqrt(
                        self._get_shortest_pan_error(current_azel[0], self.last_acquired_point_azel[0]) ** 2 + (
                                current_azel[1] - self.last_acquired_point_azel[1]) ** 2) > ACQUIRE_MIN_SEPARATION_DEG:
                    k = self.shared_data["points_count"].value;
                    print(f"[HWCtrl-Acquire] Found point {k + 1}/{ACQUIRE_POINTS_NEEDED}")
                    self.shared_data["points_buffer"][k * 5: k * 5 + 5] = [current_azel[0], current_azel[1],
                                                                           dist_cm / 100.0, strength, ts]
                    self.shared_data["points_count"].value = k + 1;
                    self.last_acquired_point_azel = current_azel
                    if self.shared_data["points_count"].value >= ACQUIRE_POINTS_NEEDED: self.shared_data[
                        "acquirer_status"].value = 2; self.shared_data["acquire_points"].value = False
        except queue.Empty:
            pass
        return self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)

    def _handle_hf_tracking(self, dt):
        pan_vel, tilt_vel = 0, 0
        if self.hf_tracking_substate == "IDLE":
            if self.shared_data["new_prediction_available"].value: self.shared_data[
                "new_prediction_available"].value = False; self.hf_tracking_substate = "MOVING_TO_PREDICTION"
        elif self.hf_tracking_substate == "MOVING_TO_PREDICTION":
            target_az, target_el = self.shared_data["predicted_azimuth"].value, self.shared_data[
                "predicted_elevation"].value
            self.pan_pid.set_setpoint(target_az);
            self.tilt_pid.set_setpoint(target_el)
            if abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos)) < self.ARRIVAL_THRESHOLD_DEG and abs(
                    target_el - self.internal_tilt_pos) < self.ARRIVAL_THRESHOLD_DEG:
                self.hf_tracking_substate = "SCANNING_GRID";
                self.grid_scan_index = 0;
                self.grid_scan_points = [];
                self.grid_scan_center_pos = (target_az, target_el);
                self.grid_point_arrival_time = time.monotonic()
            else:
                pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                    self.internal_tilt_pos)
        elif self.hf_tracking_substate == "SCANNING_GRID":
            if time.monotonic() > self.grid_point_arrival_time + self.GRID_DWELL_TIME:
                try:
                    while not self.lidar_queue.empty(): self.grid_scan_points.append(
                        [self.pan_pid.get_setpoint(), self.tilt_pid.get_setpoint()] + list(
                            self.lidar_queue.get_nowait()))
                except queue.Empty:
                    pass
                if self.grid_scan_index >= len(self.GRID_OFFSETS):
                    if self.grid_scan_points:
                        best_point = min(self.grid_scan_points, key=lambda p: p[2]); refined = [best_point[0],
                                                                                                best_point[1],
                                                                                                best_point[2] / 100.0,
                                                                                                best_point[3],
                                                                                                best_point[4]]
                    else:
                        refined = [self.grid_scan_center_pos[0], self.grid_scan_center_pos[1], 9999.0, 0, time.time()]
                    self.shared_data["refined_measurement"][:] = refined;
                    self.shared_data["refined_measurement_updated"].value = True;
                    self.hf_tracking_substate = "IDLE"
                else:
                    pan_off, tilt_off = self.GRID_OFFSETS[self.grid_scan_index]
                    self.pan_pid.set_setpoint((self.grid_scan_center_pos[0] + pan_off * self.GRID_STEP_DEG) % 360)
                    self.tilt_pid.set_setpoint(
                        max(0, min(90, self.grid_scan_center_pos[1] + tilt_off * self.GRID_STEP_DEG)))
                    self.grid_scan_index += 1;
                    self.grid_point_arrival_time = time.monotonic()
            pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)
        return pan_vel, tilt_vel

    def run(self):
        try:
            self.pi = pigpio.pi();
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
            print("[HWCtrl] Hardware Controller process is running.")
            last_loop_time, current_state = time.monotonic(), "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: time.sleep(0.001); continue
                last_loop_time = time.monotonic()
                if self.shared_data["acquire_points"].value:
                    next_state = "ACQUIRING"
                elif self.shared_data["lidar_track_mode_active"].value:
                    next_state = "HF_TRACKING"
                elif self.shared_data["go_to_target"].value:
                    next_state = "GOTO_POSITION"
                else:
                    next_state = "IDLE"
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    if next_state == "ACQUIRING":
                        self.acquire_scan_el = ACQUIRE_TILT_MAX;
                        self.acquire_pan_direction = 1;
                        self.last_acquired_point_azel = None
                        self.shared_data["points_count"].value = 0;
                        self.shared_data["acquirer_status"].value = 1
                        self.pan_pid.set_setpoint(self.internal_pan_pos)
                    elif next_state == "HF_TRACKING":
                        self.hf_tracking_substate = "IDLE"
                    current_state = next_state
                pan_vel, tilt_vel = 0, 0
                if current_state == "ACQUIRING":
                    pan_vel, tilt_vel = self._handle_acquisition(dt)
                elif current_state == "HF_TRACKING":
                    pan_vel, tilt_vel = self._handle_hf_tracking(dt)
                elif current_state == "GOTO_POSITION":
                    target_az, target_el = self.shared_data["target_azimuth"].value, self.shared_data[
                        "target_elevation"].value
                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    if abs(self._get_shortest_pan_error(target_az,
                                                        self.internal_pan_pos)) < TARGET_REACHED_THRESHOLD_DEG and abs(
                        target_el - self.internal_tilt_pos) < TARGET_REACHED_THRESHOLD_DEG:
                        self.shared_data["target_reached"].value = True; self.shared_data["go_to_target"].value = False
                    else:
                        pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                            self.internal_tilt_pos)
                self._execute_motor_commands(pan_vel, tilt_vel, dt)
                try:
                    dist, strength, ts = self.lidar_queue.get_nowait()
                    self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass
                self.shared_data["stepper_degrees"].value, self.shared_data[
                    "servo_degrees"].value = self.internal_pan_pos, self.internal_tilt_pos
        except Exception as e:
            print(f"[HWCtrl] CRITICAL ERROR: {e}"); traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...");
            self.shutdown_event.set()
            if self.pi and self.pi.connected: self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0); self.pi.write(
                STEPPER_ENABLE_PIN, 1); self.pi.write(STEPPER_SLEEP_PIN, 0); self.pi.set_servo_pulsewidth(SERVO_PIN,
                                                                                                          0); self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()