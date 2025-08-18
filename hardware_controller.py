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
SCAN_PAN_SPEED_DPS = 720.0  # Speed for scanning

SCAN_TURNAROUND_DEG = 0.1

# --- Define the boundaries and resolution for scanning ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1

# ==============================================================================
# --- SCAN CALIBRATION ---
SCAN_PAN_CALIBRATION_OFFSET_DEG = 0
# ==============================================================================


# ==============================================================================
# --- NEW: PROPORTIONAL CONTROL GAINS ---
# This is now the "P" in PID. It's a simple multiplier.
# If you set this too high, the motor will be very fast but will overshoot
# and oscillate wildly. If it's too low, it will be very slow to reach the target.
PAN_KP = 6.5
TILT_KP = 6.5
MAX_PAN_SPEED_DPS = 720.0
MAX_TILT_SPEED_DPS = 600.0


# ==============================================================================

# ==============================================================================
# PID CONTROLLER CLASS (REMOVED)
# We are no longer using the PIDController class for this implementation.
# ==============================================================================


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi, self.ser = None, None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=10)
        self.internal_pan_pos = shared_data["stepper_degrees"].value
        self.internal_tilt_pos = shared_data["servo_degrees"].value
        # PID controllers are no longer here
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.background_data_buffer = []
        self.scan_is_turning = False

    def _get_shortest_pan_error(self, setpoint, current_value):
        """Calculates the shortest error for a wrapped angle (e.g., 350 -> 10 is -20, not +340)"""
        error = setpoint - current_value
        return (error + 180) % 360 - 180

    def _get_proportional_velocity(self, error, Kp, max_speed):
        """
        NEW: This is our simple speed function.
        Velocity is directly proportional to the error.
        """
        velocity = error * Kp
        # Clamp the velocity to the maximum allowed speed
        return max(-max_speed, min(max_speed, velocity))

    def _lidar_reader_thread(self):
        # ... (lidar reader thread remains unchanged) ...
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
        # ... (motor execution logic remains unchanged) ...
        self.internal_pan_pos = (self.internal_pan_pos + (pan_velocity_dps * dt)) % 360
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_velocity_dps * dt)))
        if abs(pan_velocity_dps) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(min(abs(pan_velocity_dps) / MICROSTEP_ANGLE, 250000)), 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(500 + (self.internal_tilt_pos / 0.09) + (36 / 0.09)))

    def run(self):
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
            self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
            self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
            self.pi.write(STEPPER_ENABLE_PIN, 0)
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
                    if next_state == "BACKGROUND_SCAN":
                        self.current_scan_el, self.scan_pan_direction, self.scan_is_turning = SCAN_TILT_MAX, 1, False
                    current_state = next_state

                pan_vel, tilt_vel = 0, 0

                if current_state == "IDLE":
                    pass
                elif current_state == "GOTO_POSITION" or current_state == "HF_TRACKING":
                    target_az = self.shared_data["target_azimuth"].value if current_state == "GOTO_POSITION" else \
                        self.shared_data["predicted_azimuth"].value
                    target_el = self.shared_data["target_elevation"].value if current_state == "GOTO_POSITION" else \
                        self.shared_data["predicted_elevation"].value

                    # --- REPLACEMENT LOGIC ---
                    pan_error = self._get_shortest_pan_error(target_az, self.internal_pan_pos)
                    tilt_error = target_el - self.internal_tilt_pos

                    pan_vel = self._get_proportional_velocity(pan_error, PAN_KP, MAX_PAN_SPEED_DPS)
                    tilt_vel = self._get_proportional_velocity(tilt_error, TILT_KP, MAX_TILT_SPEED_DPS)
                    # --- END REPLACEMENT ---

                    target_reached = abs(pan_error) < TARGET_REACHED_THRESHOLD_DEG and abs(
                        tilt_error) < TARGET_REACHED_THRESHOLD_DEG

                    if current_state == "GOTO_POSITION":
                        self.shared_data["target_reached"].value = target_reached

                elif current_state == "BACKGROUND_SCAN":
                    # For scanning, we want a constant velocity, not one that slows down.
                    # So we don't use the proportional controller here.
                    pan_vel = SCAN_PAN_SPEED_DPS * self.scan_pan_direction

                    # Turnaround logic
                    is_past_max = self.scan_pan_direction == 1 and self.internal_pan_pos >= SCAN_PAN_MAX
                    is_past_min = self.scan_pan_direction == -1 and self.internal_pan_pos <= SCAN_PAN_MIN

                    if is_past_max or is_past_min:
                        pan_vel = 0  # Stop
                        self.scan_pan_direction *= -1
                        self.current_scan_el -= SCAN_STEP_DEG
                        print(
                            f"[HWCtrl-SCAN] Row finished. New elevation: {self.current_scan_el:.1f} deg, Direction: {self.scan_pan_direction}")

                        if self.current_scan_el < SCAN_TILT_MIN:
                            print("[HWCtrl] BACKGROUND_SCAN finished.")
                            # ... (save logic remains the same) ...
                            self.shared_data["background_scan_active"].value = False

                    # Tilt control for scanning
                    tilt_error = self.current_scan_el - self.internal_tilt_pos
                    tilt_vel = self._get_proportional_velocity(tilt_error, TILT_KP, MAX_TILT_SPEED_DPS)

                # Execute the calculated velocities
                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # ... (LIDAR data processing and shutdown logic remains the same) ...
                try:
                    while not self.lidar_queue.empty():
                        dist, strength, ts = self.lidar_queue.get_nowait()
                        with self.shared_data["lidar_data"].get_lock():
                            self.shared_data["lidar_data"][:] = [dist, strength, ts]

                        if current_state == "BACKGROUND_SCAN":
                            corrected_pan_pos = (self.internal_pan_pos + (
                                        self.scan_pan_direction * SCAN_PAN_CALIBRATION_OFFSET_DEG)) % 360.0
                            self.background_data_buffer.append(
                                [corrected_pan_pos, self.internal_tilt_pos, dist, strength])
                except queue.Empty:
                    pass

                self.shared_data["stepper_degrees"].value, self.shared_data[
                    "servo_degrees"].value = self.internal_pan_pos, self.internal_tilt_pos
                time.sleep(0.001)

        # ... (finally block remains the same) ...
        except Exception as e:
            import traceback
            print(f"[HWCtrl] CRITICAL ERROR: {e}")
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0)
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop()
                print("[HWCtrl] pigpio resources released.")
            if self.ser and self.ser.is_open: self.ser.close()


# ... (run_hardware_controller function remains the same) ...

def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()