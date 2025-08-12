# hardware_controller.py (Corrected for Rotational Control)

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
STEPPER_ENABLE_PIN = 4  # Active LOW to enable driver
STEPPER_SLEEP_PIN = 6  # Set HIGH for operation
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 1.0

# --- Define the boundaries and resolution for scanning and searching ---

SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0

# --- PID Tuning Gains (CRITICAL!) ---

MAX_PAN_SPEED_DPS = 360.0
PAN_KP, PAN_KI, PAN_KD = 1.0, 0.05, 0.15
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 1.2, 0.1, 0.2

#==============================================================================
# PID CONTROLLER CLASS (UPGRADED FOR ROTATIONAL SYSTEMS)
#==============================================================================

class PIDController:
    """
    A generic PID controller class, now with support for wrapped ranges
    (e.g., for 360-degree rotational systems).
    """

    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20, wrap_range=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup_limit = anti_windup_limit
        # --- CODE REPAIRED HERE ---
        # If wrap_range is set (e.g., to (0, 360)), the controller will handle error
        # calculation for a continuous, wrapping system.
        self.wrap_range = wrap_range

        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()
        self._last_output = 0

    def update(self, current_value):
        current_time = time.monotonic()
        dt = current_time - self._last_time
        if dt <= 0: return self._last_output

        error = self.setpoint - current_value

        # --- CODE REPAIRED HERE ---
        # If wrapping is enabled, calculate the shortest path.
        # For example, going from 359deg to 1deg is an error of +2, not -358.
        if self.wrap_range is not None:
            range_width = self.wrap_range[1] - self.wrap_range[0]
            # This formula correctly finds the shortest angular distance
            error = (error + range_width / 2) % range_width - range_width / 2

        P_out = self.Kp * error
        self._integral += error * dt
        self._integral = max(-self.anti_windup_limit, min(self.anti_windup_limit, self._integral))
        I_out = self.Ki * self._integral
        derivative = (error - self._last_error) / dt
        D_out = self.Kd * derivative
        output = P_out + I_out + D_out
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self._last_error = error
        self._last_time = current_time
        self._last_output = output
        return output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.monotonic()
#==============================================================================

class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=1)
        self.internal_pan_pos = self.shared_data["stepper_degrees"].value
        self.internal_tilt_pos = self.shared_data["servo_degrees"].value

        # --- CODE REPAIRED HERE ---
        # Initialize the pan PID with wrapping enabled for the 0-360 degree range.
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD,
                                     output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS),
                                     wrap_range=(0, 360))

        # The tilt PID does not wrap, so it's initialized normally.
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))

        self.current_scan_az = SCAN_PAN_MIN
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.background_data_buffer = []

    def _lidar_reader_thread(self):
        """Dedicated thread to continuously read from the TF-Mini S."""
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                self.ser.read_until(b'\x59\x59')
                frame = self.ser.read(7)
                if len(frame) == 7:
                    distance, strength = frame[0] + (frame[1] << 8), frame[2] + (frame[3] << 8)
                    try:
                        self.lidar_queue.put_nowait((distance, strength, time.time()))
                    except queue.Full:
                        pass
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set():
                    print("[HWCtrl-LIDAR] Serial error. Thread stopping.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        """Translates desired velocities into hardware commands."""
        # Pan Stepper
        pan_deg_change = pan_velocity_dps * dt
        if abs(pan_deg_change) > 0.001:
            pan_steps_to_move = round(abs(pan_deg_change) / MICROSTEP_ANGLE)
            if pan_steps_to_move > 0:
                self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
                frequency = min(pan_steps_to_move / dt, 250000)
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(frequency), 500000)
            else:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            # The modulo operator correctly handles wrapping the internal position.
            self.internal_pan_pos = (self.internal_pan_pos + pan_deg_change) % 360
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Tilt Servo
        tilt_deg_change = tilt_velocity_dps * dt
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + tilt_deg_change))
        pulse_width = 500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))

    # ... (The rest of the file remains the same) ...

    def _update_scan_pattern(self):
        """Calculates the next point in the raster scan pattern."""
        self.current_scan_az += SCAN_STEP_DEG * self.scan_pan_direction
        if self.scan_pan_direction == 1 and self.current_scan_az > SCAN_PAN_MAX:
            self.scan_pan_direction = -1
            self.current_scan_az = SCAN_PAN_MAX
            self.current_scan_el -= SCAN_STEP_DEG
        elif self.scan_pan_direction == -1 and self.current_scan_az < SCAN_PAN_MIN:
            self.scan_pan_direction = 1
            self.current_scan_az = SCAN_PAN_MIN
            self.current_scan_el -= SCAN_STEP_DEG
        return self.current_scan_el >= SCAN_TILT_MIN

    def run(self):
        """Main entry point and control loop for the hardware controller."""
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0)
            self.pi.write(STEPPER_SLEEP_PIN, 1)
            print("[HWCtrl] Stepper driver enabled.")
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            set_rate_command = bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E])
            self.ser.write(set_rate_command)
            lidar_thread = threading.Thread(target=self._lidar_reader_thread); lidar_thread.daemon = True; lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")
            last_loop_time = time.monotonic()
            current_state = "IDLE"
            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001:
                    time.sleep(0.001)
                    continue
                last_loop_time = time.monotonic()
                if self.shared_data["lidar_track_mode_active"].value: next_state = "HF_TRACKING"
                elif self.shared_data["go_to_target"].value: next_state = "GOTO_POSITION"
                elif self.shared_data["background_scan_active"].value: next_state = "BACKGROUND_SCAN"
                else: next_state = "IDLE"
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset(); self.tilt_pid.reset()
                    if next_state in ["BACKGROUND_SCAN", "SEARCHING"]: self.current_scan_az, self.current_scan_el = SCAN_PAN_MIN, SCAN_TILT_MAX; self.scan_pan_direction = 1
                    current_state = next_state
                pan_vel, tilt_vel = 0, 0
                target_reached = abs(self.pan_pid.setpoint - self.internal_pan_pos) < TARGET_REACHED_THRESHOLD_DEG and abs(self.tilt_pid.setpoint - self.internal_tilt_pos) < TARGET_REACHED_THRESHOLD_DEG
                if current_state == "HF_TRACKING":
                    self.pan_pid.set_setpoint(self.shared_data["predicted_azimuth"].value); self.tilt_pid.set_setpoint(self.shared_data["predicted_elevation"].value)
                    pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)
                elif current_state == "GOTO_POSITION":
                    self.pan_pid.set_setpoint(self.shared_data["target_azimuth"].value); self.tilt_pid.set_setpoint(self.shared_data["target_elevation"].value)
                    self.shared_data["target_reached"].value = target_reached
                    if not target_reached: pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)
                elif current_state in ["BACKGROUND_SCAN", "SEARCHING"]:
                    self.pan_pid.set_setpoint(self.current_scan_az); self.tilt_pid.set_setpoint(self.current_scan_el)
                    if target_reached:
                        try:
                            dist, strength, _ = self.lidar_queue.get_nowait()
                            if current_state == "BACKGROUND_SCAN":
                                self.background_data_buffer.append([self.internal_pan_pos, self.internal_tilt_pos, dist, strength])
                            elif current_state == "SEARCHING":
                                min_r, max_r = self.shared_data["lidar_acceptance_range"];
                                if (min_r * 100) < dist < (max_r * 100):
                                    print(f"[HWCtrl-SEARCH] Target FOUND at Az:{self.internal_pan_pos:.1f}, El:{self.internal_tilt_pos:.1f}, Dist:{dist}cm")
                                    with self.shared_data["satellite_points"].get_lock(): self.shared_data["satellite_points"][:] = [self.internal_pan_pos, self.internal_tilt_pos, dist, strength]
                        except queue.Empty: pass
                        if not self._update_scan_pattern():
                            print(f"[HWCtrl] {current_state} finished.")
                            if current_state == "BACKGROUND_SCAN": self.shared_data["background_scan_active"].value = False
                    else:
                        pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(self.internal_tilt_pos)
                self._execute_motor_commands(pan_vel, tilt_vel, dt)
                try:
                    dist, strength, ts = self.lidar_queue.get_nowait()
                    with self.shared_data["lidar_data"].get_lock():
                        self.shared_data["lidar_data"][:] = [dist, strength, ts]
                except queue.Empty:
                    pass
                self.shared_data["stepper_degrees"].value, self.shared_data["servo_degrees"].value = self.internal_pan_pos, self.internal_tilt_pos
                if self.shared_data["save_background_trigger"].value:
                    if self.background_data_buffer:
                        print(f"[HWCtrl] Saving {len(self.background_data_buffer)} points...")
                        np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer)); self.background_data_buffer = []
                    self.shared_data["save_background_trigger"].value = False
                time.sleep(0.002)
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