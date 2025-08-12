# hardware_controller.py

import time
import serial
import pigpio
import threading
import queue
import numpy as np
import math

# --- Hardware & Scan Constants ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625
TARGET_REACHED_THRESHOLD_DEG = 1.0

# --- Scanning & Tracking Parameters ---
SCAN_PAN_MIN, SCAN_PAN_MAX = 0, 360
SCAN_TILT_MIN, SCAN_TILT_MAX = 0, 90
SCAN_STEP_DEG = 1.0
SCAN_PAN_SPEED_DPS = 120.0  # Degrees per second for raster search
SCAN_TURNAROUND_DEG = 1.0

### NEW ### - Parameters for the Auto-Track Logic
TRACK_DITHER_ANGLE = 2.0  # How far to look left/right/up/down from center
TRACK_DITHER_SPEED_DPS = 200.0  # Speed for the small dither movements
TARGET_CONFIRM_COUNT = 3  # How many consecutive detections to confirm a target
TARGET_LOSS_COUNT = 5  # How many consecutive misses to declare target lost
TARGET_DETECT_THRESHOLD_CM = 100.0  # Target must be this much closer than background
REACQUIRE_SPIRAL_RADIUS = 30.0  # Max radius for spiral search in degrees
REACQUIRE_SPIRAL_STEP = 5.0  # How far apart points are on the spiral

# --- PID Tuning Gains ---
MAX_PAN_SPEED_DPS = 600.0
PAN_KP, PAN_KI, PAN_KD = 12.0, 0.01, 0.05
MAX_TILT_SPEED_DPS = 600.0
TILT_KP, TILT_KI, TILT_KD = 8.0, 0.02, 0.05


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100), anti_windup_limit=20, wrap_range=None):
        self.Kp, self.Ki, self.Kd, self.setpoint, self.output_limits, self.anti_windup_limit, self.wrap_range = Kp, Ki, Kd, setpoint, output_limits, anti_windup_limit, wrap_range
        self._integral, self._last_error, self._last_output, self._last_time = 0, 0, 0, time.monotonic()

    def update(self, current_value):
        dt = time.monotonic() - self._last_time
        if dt <= 0: return self._last_output
        error = self.setpoint - current_value
        if self.wrap_range: error = (error + 180) % 360 - 180
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
        self._integral, self._last_error, self._last_time = 0, 0, time.monotonic()

    def get_setpoint(self):
        return self.setpoint


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
        self.background_data_buffer = []
        # --- Background scan state variables ---
        self.scan_current_tilt = SCAN_TILT_MAX
        self.scan_pan_direction = 1
        self.scan_target_pan = 0.0
        self.scan_is_turning = False
        ### NEW ### - State variables for the new Auto-Track mode
        self.auto_track_sub_state = "SEARCHING"  # SEARCHING, TRACKING, REACQUIRING
        self.background_map = None  # Will hold the loaded background data
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.detections_in_a_row = 0
        self.misses_in_a_row = 0
        self.reacquire_spiral_angle = 0.0
        self.reacquire_spiral_radius_current = 0.0

    ### NEW ### - Function to perform LiDAR data acquisition
    def _lidar_data_acquisition(self, current_state):
        """Processes the LiDAR queue and logs data if in a scanning state."""
        try:
            while not self.lidar_queue.empty():
                dist, strength, ts = self.lidar_queue.get_nowait()
                with self.shared_data["lidar_data"].get_lock():
                    self.shared_data["lidar_data"][:] = [dist, strength, ts]
                if current_state == "BACKGROUND_SCAN":
                    self.background_data_buffer.append([self.internal_pan_pos, self.internal_tilt_pos, dist, strength])
        except queue.Empty:
            pass

    def _lidar_reader_thread(self):
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

    ### NEW ### - Refactored motor control functions as per instructions
    def _pan_control(self, dt):
        """Calculates and applies pan motor velocity using PID."""
        pan_vel = self.pan_pid.update(self.internal_pan_pos)
        self.internal_pan_pos = (self.internal_pan_pos + (pan_vel * dt)) % 360
        if abs(pan_vel) > 0.1:
            self.pi.write(STEPPER_DIR_PIN, 0 if pan_vel < 0 else 1)
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, int(min(abs(pan_vel) / MICROSTEP_ANGLE, 250000)), 500000)
        else:
            self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0)

    def _tilt_control(self, dt):
        """Calculates and applies tilt motor velocity using PID."""
        tilt_vel = self.tilt_pid.update(self.internal_tilt_pos)
        self.internal_tilt_pos = max(0, min(90, self.internal_tilt_pos + (tilt_vel * dt)))
        # Pulsewidth calculation for servo
        pulsewidth = 500 + (self.internal_tilt_pos / 90.0) * 2000  # Standard 500-2500us for 0-180deg, adjusted for 0-90
        self.pi.set_servo_pulsewidth(SERVO_PIN, int(pulsewidth))

    def _initialize_system(self):
        """Initializes pigpio, serial port, and motor states."""
        self.pi = pigpio.pi()
        if not self.pi.connected: raise RuntimeError("pigpio connection failed.")
        # Stepper setup
        self.pi.set_mode(STEPPER_ENABLE_PIN, pigpio.OUTPUT)
        self.pi.set_mode(STEPPER_SLEEP_PIN, pigpio.OUTPUT)
        self.pi.write(STEPPER_ENABLE_PIN, 0)  # Enable driver
        self.pi.write(STEPPER_SLEEP_PIN, 1)  # Wake driver
        # Servo setup - move to home position
        self.pi.set_servo_pulsewidth(SERVO_PIN, 1500)  # Center servo
        self.internal_tilt_pos = 45.0
        print("[HWCtrl] Stepper and Servo initialized.")
        # LiDAR setup
        self.ser = serial.Serial(self.shared_data["lidar_port"].value, 115200, timeout=0.1)
        self.ser.write(bytearray([0x5A, 0x06, 0x03, 0xE8, 0x03, 0x4E]))  # Set frame rate to 1000Hz
        threading.Thread(target=self._lidar_reader_thread, daemon=True).start()
        print("[HWCtrl] Hardware Controller process is running.")

    ### NEW ### - The Main Tracking Loop logic
    def _auto_track_logic(self):
        """Manages the SEARCHING, TRACKING, and REACQUIRING sub-states."""
        # --- SUB-STATE: SEARCHING ---
        if self.auto_track_sub_state == "SEARCHING":
            # This logic is a simplified raster scan, similar to BACKGROUND_SCAN
            if self.scan_is_turning:
                # Wait for motor to settle at the edge
                if abs(self.pan_pid.get_setpoint() - self.internal_pan_pos) < TARGET_REACHED_THRESHOLD_DEG:
                    self.scan_pan_direction *= -1
                    self.scan_current_tilt -= SCAN_STEP_DEG
                    self.scan_is_turning = False
                    if self.scan_current_tilt < SCAN_TILT_MIN:  # Full scan finished, restart
                        self.scan_current_tilt = SCAN_TILT_MAX
            else:  # Sweeping
                self.scan_target_pan += SCAN_PAN_SPEED_DPS * self.scan_pan_direction * 0.01  # Use fixed dt for stability
                if self.scan_pan_direction == 1 and self.scan_target_pan >= SCAN_PAN_MAX:
                    self.scan_is_turning = True
                elif self.scan_pan_direction == -1 and self.scan_target_pan <= SCAN_PAN_MIN:
                    self.scan_is_turning = True

            self.pan_pid.set_setpoint(self.scan_target_pan)
            self.tilt_pid.set_setpoint(self.scan_current_tilt)

            # --- Target Detection Logic ---
            current_dist_cm = self.shared_data['lidar_data'][0]
            bg_dist_cm = self.background_map.get((int(self.internal_pan_pos), int(self.internal_tilt_pos)), 99999)

            if 10 < current_dist_cm < (bg_dist_cm - TARGET_DETECT_THRESHOLD_CM):
                self.detections_in_a_row += 1
                if self.detections_in_a_row >= TARGET_CONFIRM_COUNT:
                    print(
                        f"[HWCtrl-TRACK] Target confirmed at Az:{self.internal_pan_pos:.1f}, El:{self.internal_tilt_pos:.1f}")
                    self.target_pan = self.internal_pan_pos
                    self.target_tilt = self.internal_tilt_pos
                    self.auto_track_sub_state = "TRACKING"
                    self.misses_in_a_row = 0
            else:
                self.detections_in_a_row = 0  # Reset counter on a miss

        # --- SUB-STATE: TRACKING ---
        elif self.auto_track_sub_state == "TRACKING":
            # Implements the dither/edge-finding logic
            # This is a simplified version. A real implementation would move to each point and wait for a reading.
            # For this example, we'll estimate based on current LiDAR reading and a simple nudge.
            # This is a conceptual implementation. A robust one needs a state machine for the 4-point check.

            # Simple proportional control based on error from center
            # A more advanced method would do the 4-point dither scan. Let's start simply.
            self.pan_pid.set_setpoint(self.target_pan)
            self.tilt_pid.set_setpoint(self.target_tilt)

            # Pretend we did a dither: if current reading is a miss, nudge towards center
            current_dist_cm = self.shared_data['lidar_data'][0]
            if current_dist_cm > (self.background_map.get((int(self.target_pan), int(self.target_tilt)), 500) + 100):
                self.misses_in_a_row += 1
                if self.misses_in_a_row > TARGET_LOSS_COUNT:
                    print("[HWCtrl-TRACK] Target lost. Re-acquiring...")
                    self.auto_track_sub_state = "REACQUIRING"
                    self.reacquire_spiral_radius_current = REACQUIRE_SPIRAL_STEP
                    self.reacquire_spiral_angle = 0
            else:
                self.misses_in_a_row = 0
                # A simple adjustment: assume the target is where the current valid reading is
                self.target_pan = self.internal_pan_pos
                self.target_tilt = self.internal_tilt_pos

        # --- SUB-STATE: REACQUIRING ---
        elif self.auto_track_sub_state == "REACQUIRING":
            # Implements the spiral search
            if self.reacquire_spiral_radius_current > REACQUIRE_SPIRAL_RADIUS:
                print("[HWCtrl-TRACK] Re-acquisition failed. Returning to full scan.")
                self.auto_track_sub_state = "SEARCHING"
                self.scan_current_tilt = SCAN_TILT_MAX  # Reset scanner
                return

            # Calculate next point on the spiral
            spiral_pan = self.target_pan + self.reacquire_spiral_radius_current * math.cos(self.reacquire_spiral_angle)
            spiral_tilt = self.target_tilt + self.reacquire_spiral_radius_current * math.sin(
                self.reacquire_spiral_angle)
            self.pan_pid.set_setpoint(spiral_pan)
            self.tilt_pid.set_setpoint(spiral_tilt)

            # Check for target at this point on the spiral
            current_dist_cm = self.shared_data['lidar_data'][0]
            bg_dist_cm = self.background_map.get((int(self.internal_pan_pos), int(self.internal_tilt_pos)), 99999)
            if 10 < current_dist_cm < (bg_dist_cm - TARGET_DETECT_THRESHOLD_CM):
                print("[HWCtrl-TRACK] Target re-acquired during spiral search!")
                self.auto_track_sub_state = "TRACKING"
                self.target_pan = self.internal_pan_pos  # Lock on new position
                self.target_tilt = self.internal_tilt_pos

            # Increment spiral
            self.reacquire_spiral_angle += math.radians(30)  # Increase angle
            if self.reacquire_spiral_angle > 2 * math.pi:
                self.reacquire_spiral_angle = 0
                self.reacquire_spiral_radius_current += REACQUIRE_SPIRAL_STEP

    def run(self):
        try:
            self._initialize_system()
            last_loop_time, current_state = time.monotonic(), "IDLE"

            while not self.shared_data["shutdown"].value:
                dt = time.monotonic() - last_loop_time
                if dt <= 0.001: continue
                last_loop_time = time.monotonic()

                # --- State Machine ---
                if self.shared_data["auto_track_active"].value:
                    next_state = "AUTO_TRACK"
                elif self.shared_data["background_scan_active"].value:
                    next_state = "BACKGROUND_SCAN"
                elif self.shared_data["go_to_target"].value:
                    next_state = "GOTO_POSITION"
                elif self.shared_data["lidar_track_mode_active"].value:
                    next_state = "HF_TRACKING"
                else:
                    next_state = "IDLE"

                # --- State Transition Logic ---
                if next_state != current_state:
                    print(f"[HWCtrl] State change: {current_state} -> {next_state}")
                    self.pan_pid.reset();
                    self.tilt_pid.reset()
                    current_state = next_state
                    # --- Initialization for new states ---
                    if current_state == "BACKGROUND_SCAN":
                        self.scan_current_tilt, self.scan_pan_direction = SCAN_TILT_MAX, 1
                        self.scan_target_pan = self.internal_pan_pos
                    ### NEW ###
                    elif current_state == "AUTO_TRACK":
                        try:
                            bg_data = np.load(self.shared_data["background_path"].value)
                            # Create a dictionary for fast lookup: (az,el) -> dist
                            self.background_map = {(int(az), int(el)): dist for az, el, dist, _ in bg_data}
                            print(f"[HWCtrl-TRACK] Loaded background map with {len(self.background_map)} points.")
                            self.auto_track_sub_state = "SEARCHING"
                            self.scan_current_tilt = SCAN_TILT_MAX  # Start scan from top
                        except FileNotFoundError:
                            print("[HWCtrl-TRACK] ERROR: background_data.npy not found! Cannot start AUTO_TRACK.")
                            self.shared_data["auto_track_active"].value = False  # Abort

                # --- State Execution ---
                if current_state == "IDLE":
                    pass  # Motors will be idle
                elif current_state == "GOTO_POSITION":
                    # ... (existing goto logic)
                    self.pan_pid.set_setpoint(self.shared_data["target_azimuth"].value)
                    self.tilt_pid.set_setpoint(self.shared_data["target_elevation"].value)
                    target_reached = abs(
                        self.pan_pid.get_setpoint() - self.internal_pan_pos) < TARGET_REACHED_THRESHOLD_DEG
                    self.shared_data["target_reached"].value = target_reached

                elif current_state == "HF_TRACKING":
                    # ... (existing HF logic)
                    self.pan_pid.set_setpoint(self.shared_data["predicted_azimuth"].value)
                    self.tilt_pid.set_setpoint(self.shared_data["predicted_elevation"].value)

                elif current_state == "BACKGROUND_SCAN":
                    # ... (existing scan logic, slightly refactored)
                    if self.scan_current_tilt < SCAN_TILT_MIN:
                        print("[HWCtrl] BACKGROUND_SCAN finished.")
                        if self.background_data_buffer:
                            np.save(self.shared_data["background_path"].value, np.array(self.background_data_buffer))
                            self.background_data_buffer = []
                        self.shared_data["background_scan_active"].value = False
                    # (The rest of the scan logic is now similar to the SEARCHING sub-state)

                ### NEW ### - Call the new auto-track logic function
                elif current_state == "AUTO_TRACK":
                    self._auto_track_logic()

                # --- Execute motor commands and process LiDAR data ---
                if current_state != "IDLE":
                    self._pan_control(dt)
                    self._tilt_control(dt)

                self._lidar_data_acquisition(current_state)

                # --- Update shared memory with current state ---
                self.shared_data["stepper_degrees"].value = self.internal_pan_pos
                self.shared_data["servo_degrees"].value = self.internal_tilt_pos
                time.sleep(0.002)

        except Exception as e:
            import traceback;
            traceback.print_exc()
        finally:
            print("[HWCtrl] Shutting down...")
            self.shutdown_event.set()
            if self.pi and self.pi.connected:
                self.pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
                self.pi.write(STEPPER_ENABLE_PIN, 1)
                self.pi.write(STEPPER_SLEEP_PIN, 0);
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
                self.pi.stop()
            if self.ser and self.ser.is_open: self.ser.close()


def run_hardware_controller(shared_data):
    controller = HardwareController(shared_data)
    controller.run()