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
TARGET_REACHED_THRESHOLD_DEG = 0.4  # Wider threshold for settling in the turnaround zone
SCAN_PAN_SPEED_DPS = 1200.0  # Can be set aggressively now

# --- CODE REPAIRED HERE ---
# This is the key to preventing skips. It defines a "braking zone" at the
# edges of the scan, giving the PID time to settle before reversing.
SCAN_TURNAROUND_DEG = 0.5

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 1200.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.000, 0.000  # Gains can be more aggressive with this new logic
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 5.0, 0.00, 0.0000


# ==============================================================================
# PID CONTROLLER CLASS
# ==============================================================================
class PIDController:
    # ... (PID Controller class remains unchanged) ...
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
        self._last_error, self.last_time, self._last_output = error, time.monotonic(), max(self.output_limits[0],
                                                                                           min(self.output_limits[1],
                                                                                               output))
        return self._last_output

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint

    def reset(self):
        self._integral, self._last_error = 0, 0;
        self._last_time = time.monotonic()


# ==============================================================================

class HardwareController:
    def __init__(self, shared_data):
        # ... (init remains mostly the same) ...
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
        # --- CODE REPAIRED HERE ---
        # A new state flag to manage the sweep vs. the turnaround.
        self.scan_is_turning = False

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
                        self.current_scan_el, self.scan_pan_direction, self.scan_is_turning = SCAN_TILT_MAX, 1, False
                        self.scan_target_az = self.internal_pan_pos
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                if current_state == "IDLE":
                    pass
                elif current_state == "GOTO_POSITION" or current_state == "HF_TRACKING":
                    # Determine the target based on the state
                    target_az = self.shared_data["target_azimuth"].value if current_state == "GOTO_POSITION" else \
                        self.shared_data["predicted_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value if current_state == "GOTO_POSITION" else \
                        self.shared_data["predicted_elevation"].value

                    # Set the PID setpoints on every loop
                    self.pan_pid.set_setpoint(target_az)
                    self.tilt_pid.set_setpoint(target_el)

                    # --- FIX IS HERE ---
                    # ALWAYS update the PID controller. Let it decide the velocity.
                    # It will naturally output 0 velocity when the error is 0.
                    pan_vel = self.pan_pid.update(self.internal_pan_pos)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                    # Now, separately, calculate the status flag without affecting the motors.
                    pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                    tilt_error = abs(target_el - self.internal_tilt_pos)
                    target_reached = pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG

                    # Update the shared flag for other processes to read
                    if current_state == "GOTO_POSITION":
                        self.shared_data["target_reached"].value = target_reached
                elif current_state == "BACKGROUND_SCAN":
                    # --- CODE MODIFIED HERE FOR AUTO-SAVE ---
                    if self.current_scan_el < SCAN_TILT_MIN:
                        print("[HWCtrl] BACKGROUND_SCAN finished.")

                        # Automatically save the data upon completion.
                        if self.background_data_buffer:
                            print(f"[HWCtrl] Auto-saving {len(self.background_data_buffer)} background scan points...")
                            try:
                                np.save(self.shared_data["background_path"].value,
                                        np.array(self.background_data_buffer))
                                print(f"[HWCtrl] Data saved to {self.shared_data['background_path'].value}")
                                self.background_data_buffer = []  # Clear buffer after saving
                            except Exception as e:
                                print(f"[HWCtrl] ERROR saving background data: {e}")

                        # Now, signal that the scan is no longer active.
                        self.shared_data["background_scan_active"].value = False

                    elif self.scan_is_turning:
                        # We are in a turnaround. Wait for the motor to settle at the edge.
                        pan_error = abs(
                            self._get_shortest_pan_error(self.pan_pid.get_setpoint(), self.internal_pan_pos))
                        if pan_error < TARGET_REACHED_THRESHOLD_DEG:
                            # Safely at the edge, now execute the turn.
                            self.scan_pan_direction *= -1
                            self.current_scan_el -= SCAN_STEP_DEG
                            self.scan_is_turning = False
                            print(
                                f"[HWCtrl-SCAN] Row finished. New elevation: {self.current_scan_el:.1f} deg, Direction: {self.scan_pan_direction}")
                    else:
                        # We are sweeping. Move the virtual target.
                        self.scan_target_az += SCAN_PAN_SPEED_DPS * self.scan_pan_direction * dt

                        # Check if we've entered a turnaround zone.
                        if self.scan_pan_direction == 1 and self.scan_target_az >= SCAN_PAN_MAX - SCAN_TURNAROUND_DEG:
                            self.scan_target_az = SCAN_PAN_MAX  # Lock target to the edge
                            self.scan_is_turning = True
                        elif self.scan_pan_direction == -1 and self.scan_target_az <= SCAN_PAN_MIN + SCAN_TURNAROUND_DEG:
                            self.scan_target_az = SCAN_PAN_MIN  # Lock target to the edge
                            self.scan_is_turning = True

                    # In all cases (sweeping or turning), tell the PID to chase the target.
                    self.pan_pid.set_setpoint(self.scan_target_az)
                    pan_vel = self.pan_pid.update(self.internal_pan_pos)
                    self.tilt_pid.set_setpoint(self.current_scan_el)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                else: #HF_TRACKING
                    # In high-frequency tracking mode, we use the predicted values directly.
                    pan_vel = self.pan_pid.update(self.shared_data["predicted_azimuth"].value
                        if self.shared_data["predicted_azimuth"].value is not None else 0)
                    tilt_vel = self.tilt_pid.update(self.shared_data["predicted_elevation"].value
                        if self.shared_data["predicted_elevation"].value is not None else 0)
                # Execute the motor commands
                    pan_vel = max(-MAX_PAN_SPEED_DPS, min(MAX_PAN_SPEED_DPS, pan_vel))
                    tilt_vel = max(-MAX_TILT_SPEED_DPS, min(MAX_TILT_SPEED_DPS, tilt_vel))




                self._execute_motor_commands(pan_vel, tilt_vel, dt)

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

                self.shared_data["stepper_degrees"].value, self.shared_data[
                    "servo_degrees"].value = self.internal_pan_pos, self.internal_tilt_pos

                # --- CODE MODIFIED HERE ---
                # The manual save trigger has been removed from here.

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


# Helper to get setpoint from PID, missing in the original class
def get_setpoint(self):
    return self.setpoint


PIDController.get_setpoint = get_setpoint


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()