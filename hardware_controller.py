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
# Tighter threshold for GOTO, wider for background scan turnaround
GOTO_TARGET_REACHED_THRESHOLD_DEG = 0.5
SCAN_TURNAROUND_REACHED_THRESHOLD_DEG = 1.0
SCAN_PAN_SPEED_DPS = 600.0

# --- CODE REPAIRED HERE ---
# This is the key to preventing skips. It defines a "braking zone" at the
# edges of the scan, giving the PID time to settle before reversing.
SCAN_TURNAROUND_DEG = 1.0

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 600.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.01, 0.05
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
        self.scan_is_turning = False

        # --- NEW: Variables for 9-Point Tracking Logic ---
        # The sub-state for the HF_TRACKING mode
        self.hf_tracking_sub_state = "IDLE"  # Can be IDLE, INITIALIZING, MOVING, DWELLING, ANALYZING
        self.grid_scan_index = 0
        self.grid_scan_results = []
        self.grid_scan_dwell_start = 0
        # Defines the 3x3, 1-degree-apart grid offsets
        self.GRID_SCAN_OFFSETS = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
        self.GRID_SCAN_DWELL_TIME_S = 0.005  # 50ms dwell time to get a stable reading
        self.grid_scan_target_points = []
        self.latest_lidar_reading = (0, 0, 0)  # (dist, str, ts) for immediate access

    def _get_shortest_pan_error(self, setpoint, current_value):
        error = setpoint - current_value
        return (error + 180) % 360 - 180

    def _lidar_reader_thread(self):
        print("[HWCtrl-LIDAR] LiDAR reader thread started.")
        while not self.shutdown_event.is_set():
            try:
                # Wait for and read a full frame
                self.ser.read_until(b'\x59\x59');
                frame = self.ser.read(7)
                if len(frame) == 7:
                    dist_cm = frame[0] + (frame[1] << 8)
                    strength = frame[2] + (frame[3] << 8)
                    timestamp = time.time()

                    # --- NEW: Update latest_lidar_reading directly for the tracker ---
                    self.latest_lidar_reading = (dist_cm, strength, timestamp)

                    try:
                        self.lidar_queue.put_nowait((dist_cm, strength, timestamp))
                    except queue.Full:
                        pass  # Don't block if the queue is full
            except (serial.SerialException, OSError):
                if not self.shutdown_event.is_set(): print("[HWCtrl-LIDAR] Serial error.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        # Update internal position based on velocity command
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))

        # Pan motor (stepper) control
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)  # Set direction
            # Set frequency for PWM based on speed, capping at 250kHz
            freq = int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, 250000))
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, freq, 500000)  # 50% duty cycle
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)  # Stop motor

        # Tilt motor (servo) control
        pulse_width = int(500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09))  # Calibration may be needed
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def run(self):
        try:
            # --- Initialization ---
            self.pi = pigpio.pi();
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            # Setup GPIO pins for stepper
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT);
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0);  # Enable driver
            self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake driver
            print("[HWCtrl] Stepper driver enabled.")
            # Setup LiDAR serial connection
            self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
            self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))  # Set LiDAR sample rate
            # Start the thread that continuously reads LiDAR data
            threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
            print("[HWCtrl] Hardware Controller process is running.")
            last_loop_time, current_state = time.monotonic(), "IDLE"

            # --- Main Control Loop ---
            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: continue
                last_loop_time = time.monotonic()

                # --- State Machine Logic ---
                if self.shared_data["background_scan_active"].value:
                    next_state = "BACKGROUND_SCAN"
                elif self.shared_data["lidar_track_mode_active"].value:
                    next_state = "HF_TRACKING"
                elif self.shared_data["go_to_target"].value:
                    next_state = "GOTO_POSITION"
                else:
                    next_state = "IDLE"

                # Handle state transitions
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    if next_state == "BACKGROUND_SCAN":
                        self.current_scan_el, self.scan_pan_direction, self.scan_is_turning = SCAN_TILT_MAX, 1, False
                        self.scan_target_az = self.internal_pan_pos
                    # --- NEW: Initialize the tracker state machine on entry ---
                    if next_state == "HF_TRACKING":
                        self.hf_tracking_sub_state = "INITIALIZING"
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                # --- State Implementations ---
                if current_state == "IDLE":
                    pass

                elif current_state == "GOTO_POSITION":
                    target_az = self.shared_data["target_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value

                    self.pan_pid.set_setpoint(target_az)
                    self.tilt_pid.set_setpoint(target_el)

                    pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                    tilt_error = abs(target_el - self.internal_tilt_pos)
                    target_reached = (pan_error < GOTO_TARGET_REACHED_THRESHOLD_DEG and
                                      tilt_error < GOTO_TARGET_REACHED_THRESHOLD_DEG)

                    if not target_reached:
                        pan_vel = self.pan_pid.update(self.internal_pan_pos)
                        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                    self.shared_data["target_reached"].value = target_reached

                # --- NEW: 9-Point Grid Scan and Track Logic ---
                elif current_state == "HF_TRACKING":
                    if self.hf_tracking_sub_state == "INITIALIZING":
                        print("[HWCtrl-TRACK] Initializing 9-point scan.")
                        center_az = self.shared_data["predicted_azimuth"].value
                        center_el = self.shared_data["predicted_elevation"].value

                        self.grid_scan_target_points = [
                            ((center_az + off_az) % 360, max(0, min(90, center_el + off_el)))
                            for off_az, off_el in self.GRID_SCAN_OFFSETS
                        ]

                        self.grid_scan_index = 0
                        self.grid_scan_results = []
                        self.hf_tracking_sub_state = "MOVING"
                        continue  # Restart loop to begin moving immediately

                    elif self.hf_tracking_sub_state == "MOVING":
                        # Check if all 9 points have been scanned
                        if self.grid_scan_index >= len(self.grid_scan_target_points):
                            self.hf_tracking_sub_state = "ANALYZING"
                            continue

                        target_az, target_el = self.grid_scan_target_points[self.grid_scan_index]
                        self.pan_pid.set_setpoint(target_az)
                        self.tilt_pid.set_setpoint(target_el)

                        pan_error = abs(self._get_shortest_pan_error(target_az, self.internal_pan_pos))
                        tilt_error = abs(target_el - self.internal_tilt_pos)
                        target_reached = (pan_error < GOTO_TARGET_REACHED_THRESHOLD_DEG and
                                          tilt_error < GOTO_TARGET_REACHED_THRESHOLD_DEG)

                        if target_reached:
                            self.hf_tracking_sub_state = "DWELLING"
                            self.grid_scan_dwell_start = time.monotonic()
                        else:
                            pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                                self.internal_tilt_pos)

                    elif self.hf_tracking_sub_state == "DWELLING":
                        # Keep motors stopped while dwelling
                        if time.monotonic() - self.grid_scan_dwell_start > self.GRID_SCAN_DWELL_TIME_S:
                            dist, strength, _ = self.latest_lidar_reading
                            current_az, current_el = self.internal_pan_pos, self.internal_tilt_pos
                            self.grid_scan_results.append((current_az, current_el, dist, strength))

                            self.grid_scan_index += 1
                            self.hf_tracking_sub_state = "MOVING"

                    elif self.hf_tracking_sub_state == "ANALYZING":
                        print("[HWCtrl-TRACK] Analyzing scan results...")
                        if not self.grid_scan_results:
                            print("[HWCtrl-TRACK] No results to analyze. Rescanning same area.")
                        else:
                            min_m, max_m = self.shared_data["lidar_acceptance_range"][:]

                            valid_points = [res for res in self.grid_scan_results if min_m <= (res[2] / 100.0) <= max_m]

                            if not valid_points:
                                print("[HWCtrl-TRACK] No valid target found in scan. Rescanning same area.")
                            else:
                                # Find the best point by sorting by signal strength (highest first)
                                best_point = sorted(valid_points, key=lambda x: x[3], reverse=True)[0]
                                best_az, best_el = best_point[0], best_point[1]
                                print(
                                    f"[HWCtrl-TRACK] New target predicted at Az: {best_az:.2f}, El: {best_el:.2f}, Str: {best_point[3]}")
                                # Update the shared prediction for the next cycle
                                self.shared_data["predicted_azimuth"].value = best_az
                                self.shared_data["predicted_elevation"].value = best_el

                        self.hf_tracking_sub_state = "INITIALIZING"  # Loop back to start the next scan
                        continue
                # --- End of New Tracking Logic ---

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

                        self.shared_data["background_scan_active"].value = False

                    elif self.scan_is_turning:
                        pan_error = abs(self._get_shortest_pan_error(self.pan_pid.setpoint, self.internal_pan_pos))
                        if pan_error < SCAN_TURNAROUND_REACHED_THRESHOLD_DEG:
                            self.scan_pan_direction *= -1
                            self.current_scan_el -= SCAN_STEP_DEG
                            self.scan_is_turning = False
                            print(
                                f"[HWCtrl-SCAN] Row finished. New elevation: {self.current_scan_el:.1f} deg, Direction: {self.scan_pan_direction}")
                    else:
                        self.scan_target_az += SCAN_PAN_SPEED_DPS * self.scan_pan_direction * dt

                        if self.scan_pan_direction == 1 and self.scan_target_az >= SCAN_PAN_MAX - SCAN_TURNAROUND_DEG:
                            self.scan_target_az, self.scan_is_turning = SCAN_PAN_MAX, True
                        elif self.scan_pan_direction == -1 and self.scan_target_az <= SCAN_PAN_MIN + SCAN_TURNAROUND_DEG:
                            self.scan_target_az, self.scan_is_turning = SCAN_PAN_MIN, True

                    self.pan_pid.set_setpoint(self.scan_target_az)
                    pan_vel = self.pan_pid.update(self.internal_pan_pos)
                    self.tilt_pid.set_setpoint(self.current_scan_el)
                    tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)

                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Independent data logging and sharing
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


# Main entry point for this process
def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()