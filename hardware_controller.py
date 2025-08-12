import time
import serial
import pigpio
import threading
import queue
import numpy as np

# --- Constants ---
# (Keep all your hardware constants: SERVO_PIN, STEPPER_PINS, etc.)
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625
SCAN_PAN_SPEED_DPS = 600.0
SCAN_TURNAROUND_DEG = 1.0
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0
MAX_PAN_SPEED_DPS = 600.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.01, 0.05
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 8.0, 0.02, 0.05

# --- NEW: Grid Scan Constants ---
GRID_STEP_DEGREES = 1.0  # Must match the value in tracker_logic.py
GRID_SETTLE_THRESHOLD_DEG = 0.5  # How close to be before taking a reading
GRID_POINT_TIMEOUT_S = 0.5  # Timeout for moving to a single grid point


# --- PID CONTROLLER CLASS ---
# (Your PIDController class remains unchanged)
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20, wrap_range=None):
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

    def reset(self):
        self._integral, self._last_error = 0, 0; self._last_time = time.monotonic()


# --- HARDWARE CONTROLLER CLASS ---
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
        # (Other init variables from your code)
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.background_data_buffer = []
        self.scan_target_az = 0.0
        self.scan_is_turning = False

    def _get_shortest_pan_error(self, setpoint, current_value):
        error = setpoint - current_value
        return (error + 180) % 360 - 180

    def _lidar_reader_thread(self):
        # (This thread remains unchanged)
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
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        # (This function remains unchanged)
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, 250000)), 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)))

    def run(self):
        try:
            # --- Hardware Initialization ---
            self.pi = pigpio.pi();
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0);
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")
            self.ser = serial.Serial("/dev/serial0", 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))
            threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time, current_state = time.monotonic(), "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: continue
                last_loop_time = time.monotonic()

                # --- STATE MACHINE LOGIC ---
                if self.shared_data["grid_scan_request"].value:  # <-- NEW STATE TRIGGER
                    next_state = "GRID_SCAN"
                elif self.shared_data["background_scan_active"].value:
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
                        self.current_scan_el, self.scan_pan_direction, self.scan_is_turning = SCAN_TILT_MAX, 1, False
                        self.scan_target_az = self.internal_pan_pos
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                # --- STATE IMPLEMENTATIONS ---
                if current_state == "IDLE":
                    pass  # Do nothing

                elif current_state == "GOTO_POSITION":
                    target_az = self.shared_data["target_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value
                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                    el_error = abs(target_el - self.internal_tilt_pos)
                    target_reached = pan_error < 1.0 and el_error < 1.0
                    if not target_reached:
                        pan_vel = self.pan_pid.update(self.internal_pan_pos)
                        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)
                    self.shared_data["target_reached"].value = target_reached

                elif current_state == "HF_TRACKING":  # Now just follows the prediction
                    target_az = self.shared_data["predicted_azimuth"].value
                    target_el = self.shared_data["predicted_elevation"].value
                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    pan_vel = self.pan_pid.update(self.internal_pan_pos)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                elif current_state == "BACKGROUND_SCAN":
                    # (Your background scan logic remains unchanged)
                    pass

                elif current_state == "GRID_SCAN":  # <-- NEW STATE IMPLEMENTATION
                    center_az = self.shared_data["grid_scan_target_az"].value
                    center_el = self.shared_data["grid_scan_target_el"].value
                    results = [0.0] * 9

                    for i in range(9):
                        row, col = divmod(i, 3)
                        pan_offset = (col - 1) * GRID_STEP_DEGREES
                        tilt_offset = (row - 1) * GRID_STEP_DEGREES
                        target_az = (center_az + pan_offset) % 360
                        target_el = max(0, min(90, center_el + tilt_offset))

                        self.pan_pid.set_setpoint(target_az)
                        self.tilt_pid.set_setpoint(target_el)

                        point_start_time = time.monotonic()
                        while time.monotonic() - point_start_time < GRID_POINT_TIMEOUT_S:
                            dt_inner = time.monotonic() - last_loop_time
                            last_loop_time = time.monotonic()
                            p_vel = self.pan_pid.update(self.internal_pan_pos)
                            t_vel = self.tilt_pid.update(self.internal_tilt_pos)
                            self._execute_motor_commands(p_vel, t_vel, dt_inner)

                            pan_err = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                            tilt_err = abs(target_el - self.internal_tilt_pos)

                            if pan_err < GRID_SETTLE_THRESHOLD_DEG and tilt_err < GRID_SETTLE_THRESHOLD_DEG:
                                break  # Settled at point
                            time.sleep(0.002)

                        # Once settled (or timed out), take measurement
                        time.sleep(0.01)  # Final settle
                        # Clear queue and get the latest reading
                        while not self.lidar_queue.empty(): self.lidar_queue.get()
                        try:
                            dist, strength, ts = self.lidar_queue.get(timeout=0.05)
                            results[i] = dist
                        except queue.Empty:
                            results[i] = 0.0  # No reading

                    # Publish results and reset flags
                    self.shared_data["grid_scan_results"][:] = results
                    self.shared_data["grid_scan_request"].value = False
                    self.shared_data["grid_scan_complete"].value = True
                    print("[HWCtrl] Grid scan complete.")
                    # State will automatically change on the next loop iteration

                # --- Motor Execution & Data Logging ---
                if current_state != "GRID_SCAN":  # Grid scan has its own motor loop
                    self._execute_motor_commands(pan_vel, tilt_vel, dt)

                try:  # Independent data logging
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()
                        with self.shared_data["lidar_data"].get_lock():
                            self.shared_data["lidar_data"][:] = [dist, strength, ts]
                        if current_state == "BACKGROUND_SCAN": self.background_data_buffer.append(
                            [self.internal_pan_pos, self.internal_tilt_pos, dist, strength])
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos
                time.sleep(0.002)

        except Exception as e:
            import traceback;
            traceback.print_exc()
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
        finally:
            # (Your shutdown cleanup logic remains unchanged)
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1);
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0);
                self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()


