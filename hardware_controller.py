# hardware_controller.py (Corrected for Physical Boundary Detection and Target Resync)

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
TARGET_REACHED_THRESHOLD_DEG = 1.0
SCAN_PAN_SPEED_DPS = 150.0

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 600.0
PAN_KP, PAN_KI, PAN_KD = 15.0, 0.08, 0.3
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 8.0, 0.02, 0.05


# ==============================================================================
# PID CONTROLLER CLASS
# ==============================================================================
class PIDController:
    # ... (PID Controller class remains unchanged) ...
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20, wrap_range=None):
        self.Kp, self.Ki, self.Kd, self.setpoint, self.output_limits, self.anti_windup_limit, self.wrap_range = Kp, Ki, Kd, setpoint, output_limits, anti_windup_limit, wrap_range
        self._integral, self._last_error, self._last_output, self._last_time = 0, 0, 0, time.monotonic()

    def update(self, current_value):
        current_time = time.monotonic()
        dt = current_time - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if self.wrap_range:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            error = (error + range_width / 2) % range_width - range_width / 2
        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * (error - self._last_error) / dt)
        self._last_error, self._last_time = error, current_time
        self._last_output = max(self.output_limits[0], min(self.output_limits[1], output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral, self._last_error = 0, 0; self._last_time = time.monotonic()


# ==============================================================================

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
        self.scan_target_az = 0.0

    def _get_shortest_pan_error(self, setpoint, current_value):
        error = setpoint - current_value
        return (error + 180) % 360 - 180

    def _lidar_reader_thread(self):
        # ... (lidar reader thread remains unchanged) ...
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
        # ... (motor execution logic remains unchanged) ...
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
            self.pi = pigpio.pi();
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
            last_physical_pan_pos = self.internal_pan_pos  # Initialize last known position

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
                        self.current_scan_el, self.scan_pan_direction = SCAN_TILT_MAX, 1
                        self.scan_target_az = self.internal_pan_pos
                        last_physical_pan_pos = self.internal_pan_pos
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                if current_state == "IDLE":
                    pass
                elif current_state == "GOTO_POSITION" or current_state == "HF_TRACKING":
                    # ... (Point-to-point logic is unchanged) ...
                    target_az = self.shared_data["target_azimuth"].value if current_state == "GOTO_POSITION" else \
                    self.shared_data["predicted_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value if current_state == "GOTO_POSITION" else \
                    self.shared_data["predicted_elevation"].value
                    self.pan_pid.set_setpoint(target_az);
                    self.tilt_pid.set_setpoint(target_el)
                    pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                    target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG
                    if not target_reached: pan_vel, tilt_vel = self.pan_pid.update(
                        self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)
                    if current_state == "GOTO_POSITION": self.shared_data["target_reached"].value = target_reached

                elif current_state == "BACKGROUND_SCAN":
                    if self.current_scan_el < SCAN_TILT_MIN:
                        print("[HWCtrl] BACKGROUND_SCAN finished.");
                        self.shared_data["background_scan_active"].value = False
                    else:
                        # 1. The virtual target moves continuously to "pull" the motor along.
                        self.scan_target_az += SCAN_PAN_SPEED_DPS * self.scan_pan_direction * dt
                        # We don't need to clamp the target; the PID's wrap logic handles it.

                        # 2. Tell the PID to chase the moving target.
                        self.pan_pid.set_setpoint(self.scan_target_az)
                        pan_vel = self.pan_pid.update(self.internal_pan_pos)

                        # 3. Use the tilt PID to hold its elevation steady.
                        self.tilt_pid.set_setpoint(self.current_scan_el)
                        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                # --- CODE REPAIRED HERE ---
                # Motor commands and data logging are now followed by the reversal check.

                # Store position BEFORE executing commands to compare against the new position
                last_physical_pan_pos = self.internal_pan_pos
                self._execute_motor_commands(pan_vel, tilt_vel, dt)  # This updates internal_pan_pos

                # Check for reversal only if we are in the scan state
                if current_state == "BACKGROUND_SCAN":
                    row_finished = False
                    # Detect a physical wrap-around from ~360 to ~0
                    if self.scan_pan_direction == 1 and self.internal_pan_pos < 90 and last_physical_pan_pos > 270:
                        row_finished = True
                    # Detect a physical wrap-around from ~0 to ~360
                    elif self.scan_pan_direction == -1 and self.internal_pan_pos > 270 and last_physical_pan_pos < 90:
                        row_finished = True

                    if row_finished:
                        self.current_scan_el -= SCAN_STEP_DEG
                        self.scan_pan_direction *= -1
                        # CRITICAL: Re-sync the virtual target to the physical position
                        # to prevent the PID from seeing a huge error and "skipping".
                        self.scan_target_az = self.internal_pan_pos
                        print(
                            f"[HWCtrl-SCAN] Physical boundary crossed. Reversing. New elevation: {self.current_scan_el:.1f} deg")

                # Independent data logging
                try:
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
                if self.shared_data["save_background_trigger"].value:
                    if self.background_data_buffer:
                        print(f"[HWCtrl] Saving {len(self.background_data_buffer)} points...")
                        np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer));
                        self.background_data_buffer = []
                    self.shared_data["save_background_trigger"].value = False

                time.sleep(0.002)
        except Exception as e:
            import traceback;
            print(f"[HWCtrl] CRITICAL ERROR: {e}");
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and locals()['lidar_thread'].is_alive(): locals()['lidar_thread'].join(
                timeout=1)
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