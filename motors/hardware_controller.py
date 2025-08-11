# ==============================================================================
# hardware_controller.py (NEW, UNIFIED VERSION)
# ------------------------------------------------------------------------------
# A unified, high-performance hardware control process.
# This single process manages the LiDAR, stepper, and servo to provide
# multiple modes of operation via a state machine.
#
# STATES:
# - IDLE: Motors are off, waiting for a command.
# - GOTO_POSITION: Moves to a single target az/el and holds position.
# - HF_TRACKING: High-frequency PID loop for tracking a moving target.
# - BACKGROUND_SCAN: Systematically scans a defined area to map the environment,
#   storing data points in a buffer for later saving.
# - SEARCHING: Scans a defined area, checking each point for a valid target.
# ==============================================================================

import time
import serial
import pigpio
import threading
import queue
import numpy as np
from .motor_utils import PIDController

# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_SLEEP_PIN = 6
STEPPER_ENABLE_PIN = 4
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 1.0  # Looser tolerance for faster scanning

# Define the boundaries and resolution for scanning and searching
SCAN_PAN_MIN, SCAN_PAN_MAX = 45, 315  # A wide 270-degree pan
SCAN_TILT_MIN, SCAN_TILT_MAX = 10, 80  # Tilt from 10 to 80 degrees
SCAN_STEP_DEG = 1.0  # Move 1 degree at a time

# --- PID Tuning Gains (CRITICAL!) ---
MAX_PAN_SPEED_DPS = 360.0  # May need to be faster for quick scans
PAN_KP, PAN_KI, PAN_KD = 1.0, 0.05, 0.15
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 1.2, 0.1, 0.2


class HardwareController:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.pi = None
        self.ser = None
        self.shutdown_event = threading.Event()
        self.lidar_queue = queue.Queue(maxsize=1)

        # Internal state for high-speed loops
        self.internal_pan_pos = self.shared_data["stepper_degrees"].value
        self.internal_tilt_pos = self.shared_data["servo_degrees"].value

        # PID controllers
        self.pan_pid = PIDController(PAN_KP, PAN_KI, PAN_KD, output_limits=(-MAX_PAN_SPEED_DPS, MAX_PAN_SPEED_DPS))
        self.tilt_pid = PIDController(TILT_KP, TILT_KI, TILT_KD,
                                      output_limits=(-MAX_TILT_SPEED_DPS, MAX_TILT_SPEED_DPS))

        # State variables for scanning and searching
        self.current_scan_az = SCAN_PAN_MIN
        self.current_scan_el = SCAN_TILT_MAX
        self.scan_pan_direction = 1  # 1 for right, -1 for left
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
                print("[HWCtrl-LIDAR] Serial error. Thread stopping.")
                break
        print("[HWCtrl-LIDAR] LiDAR reader thread shut down.")

    def _execute_motor_commands(self, pan_velocity_dps, tilt_velocity_dps, dt):
        """Translates desired velocities into hardware commands."""
        # Pan Stepper
        pan_deg_change = pan_velocity_dps * dt
        if abs(pan_deg_change) > 0:
            pan_steps_to_move = round(abs(pan_deg_change) / MICROSTEP_ANGLE)
            if pan_steps_to_move > 0:
                self.pi.write(STEPPER_DIR_PIN, 0 if pan_velocity_dps < 0 else 1)
                frequency = min(pan_steps_to_move / dt, 300000)
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, frequency, 500000)
            else:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)
            self.internal_pan_pos = (self.internal_pan_pos + pan_deg_change)
            # Clamp pan to 0-360 to prevent windup issues, though raster scan should prevent this.
            if self.internal_pan_pos >= 360: self.internal_pan_pos -= 360
            if self.internal_pan_pos < 0: self.internal_pan_pos += 360
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

        # Tilt Servo
        tilt_deg_change = tilt_velocity_dps * dt
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + tilt_deg_change))
        pulse_width = 500 + (self.internal_tilt_pos / 0.09) + (28 / 0.09)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

    def _update_scan_pattern(self):
        """Calculates the next point in the raster scan pattern."""
        self.current_scan_az += SCAN_STEP_DEG * self.scan_pan_direction

        # Check pan boundaries
        if self.scan_pan_direction == 1 and self.current_scan_az > SCAN_PAN_MAX:
            self.scan_pan_direction = -1
            self.current_scan_az = SCAN_PAN_MAX  # Ensure it ends on the boundary
            self.current_scan_el -= SCAN_STEP_DEG
        elif self.scan_pan_direction == -1 and self.current_scan_az < SCAN_PAN_MIN:
            self.scan_pan_direction = 1
            self.current_scan_az = SCAN_PAN_MIN  # Ensure it ends on the boundary
            self.current_scan_el -= SCAN_STEP_DEG

        # Check if scan is complete
        if self.current_scan_el < SCAN_TILT_MIN:
            return False  # Scan finished
        return True  # Scan ongoing

    def run(self):
        """Main entry point and control loop for the hardware controller."""
        try:
            # --- Initialization ---
            self.pi = pigpio.pi()
            self.ser = serial.Serial(self.shared_data["lidar_port"], 115200, timeout=0.1)
            self.ser.write(b'\x42\x57\x02\x00\x00\x00\x01\x06')  # 1000Hz mode

            lidar_thread = threading.Thread(target=self._lidar_reader_thread);
            lidar_thread.daemon = True;
            lidar_thread.start()
            print("[HWCtrl] Hardware Controller process is running.")

            last_loop_time = time.monotonic()
            current_state = "IDLE"

            # --- Main Control Loop ---
            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                last_loop_time = time.monotonic()

                # --- State Determination (Priority Order) ---
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

                # --- State Transition Logic ---
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()

                    # If starting a scan/search, initialize the start position
                    if next_state in ["BACKGROUND_SCAN", "SEARCHING"]:
                        self.current_scan_az = SCAN_PAN_MIN
                        self.current_scan_el = SCAN_TILT_MAX
                        self.scan_pan_direction = 1
                        print(f"[HWCtrl] Starting scan at Az:{self.current_scan_az}, El:{self.current_scan_el}")
                    current_state = next_state

                # --- State Execution ---
                pan_vel, tilt_vel = 0, 0
                target_reached = False

                # Check for target reached condition (common to many states)
                pan_error = abs(self.pan_pid.setpoint - self.internal_pan_pos)
                tilt_error = abs(self.tilt_pid.setpoint - self.internal_tilt_pos)
                if pan_error < TARGET_REACHED_THRESHOLD_DEG and tilt_error < TARGET_REACHED_THRESHOLD_DEG:
                    target_reached = True

                if current_state == "HF_TRACKING":
                    self.pan_pid.set_setpoint(self.shared_data["predicted_azimuth"].value)
                    self.tilt_pid.set_setpoint(self.shared_data["predicted_elevation"].value)
                    pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                        self.internal_tilt_pos)

                elif current_state == "GOTO_POSITION":
                    self.pan_pid.set_setpoint(self.shared_data["target_azimuth"].value)
                    self.tilt_pid.set_setpoint(self.shared_data["target_elevation"].value)
                    self.shared_data["target_reached"].value = target_reached
                    if not target_reached:
                        pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                            self.internal_tilt_pos)

                elif current_state == "BACKGROUND_SCAN" or current_state == "SEARCHING":
                    self.pan_pid.set_setpoint(self.current_scan_az)
                    self.tilt_pid.set_setpoint(self.current_scan_el)

                    if target_reached:
                        # We've arrived at a scan point. Take action.
                        try:
                            dist, strength, _ = self.lidar_queue.get_nowait()

                            if current_state == "BACKGROUND_SCAN":
                                self.background_data_buffer.append(
                                    [self.internal_pan_pos, self.internal_tilt_pos, dist, strength])

                            elif current_state == "SEARCHING":
                                min_r, max_r = self.shared_data["lidar_acceptance_range"]
                                if (min_r * 100) < dist < (max_r * 100):
                                    print(
                                        f"[HWCtrl-SEARCH] Target FOUND at Az:{self.internal_pan_pos:.1f}, El:{self.internal_tilt_pos:.1f}, Dist:{dist}cm")
                                    sp = self.shared_data["satellite_points"]
                                    with sp.get_lock():
                                        sp[:] = [self.internal_pan_pos, self.internal_tilt_pos, dist, strength]
                                    self.shared_data["satellite_detected"].value = True
                                    self.shared_data["search_mode_active"].value = False  # Stop searching
                        except queue.Empty:
                            pass  # No lidar data, just move to the next point

                        # Move to the next point in the pattern
                        if not self._update_scan_pattern():
                            print(f"[HWCtrl] {current_state} finished.")
                            if current_state == "BACKGROUND_SCAN":
                                self.shared_data["background_scan_active"].value = False
                            elif current_state == "SEARCHING":
                                self.shared_data["search_mode_active"].value = False
                    else:
                        # If not at the target yet, keep PID active
                        pan_vel, tilt_vel = self.pan_pid.update(self.internal_pan_pos), self.tilt_pid.update(
                            self.internal_tilt_pos)

                # --- Motor Command Execution & Shared Memory Update ---
                self._execute_motor_commands(pan_vel, tilt_vel, dt)

                # Update shared memory for GUI
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos

                # Check for save trigger (can happen in any state)
                if self.shared_data["save_background_trigger"].value:
                    if self.background_data_buffer:
                        print(f"[HWCtrl] Saving {len(self.background_data_buffer)} points to background data...")
                        np.save(self.shared_data["background_path"], np.array(self.background_data_buffer))
                        self.background_data_buffer = []  # Clear buffer after saving
                    self.shared_data["save_background_trigger"].value = False

                time.sleep(0.002)  # Regulate loop speed

        except Exception as e:
            import traceback
            print(f"[HWCtrl] CRITICAL ERROR in main loop: {e}")
            traceback.print_exc()
        finally:
            self.shutdown_event.set()
            if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join(timeout=1)
            if self.pi: self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0); self.pi.set_servo_pulsewidth(SERVO_PIN,
                                                                                                    0); self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()
            print("[HWCtrl] Hardware Controller shut down.")


def run_hardware_controller(shared_data):
    """Entry point function for the process."""
    controller = HardwareController(shared_data)
    controller.run()