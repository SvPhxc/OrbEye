# ==============================================================================
# motors/motor_controller.py (MODIFIED)
# ------------------------------------------------------------------------------
# This file now only contains the main process loop (`run_motor_control`).
# It imports low-level utilities from `motor_utils` and high-level logic
# from `tracking`, acting as the central orchestrator.
# ==============================================================================
from tracking.tle_tracker import acquire_target_from_tle, track_target_with_ekf

from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep
import signal

# --- Import from our new, decoupled files ---
from motors.motor_utils import stepper_worker, track_target, move, STEPPER_ENABLE_PIN, STEPPER_SLEEP_PIN
from tracking.tracker import active_track_target
from tracking.acquisition import run_acquisition_sequence, run_manual_acquisition_sequence
from tracking.tle_utils import parse_tle_file


def run_motor_control(shared_data, movement_queue):
    """Main process for controlling all physical movement and tracking logic."""

    def _graceful_stop(signum, frame):
        shared_data['shutdown'].value = True

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    print("[MotorControl] Starting...")
    GPIO.setwarnings(False);
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT);
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH);
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)

    pi = pigpio.pi()
    if not pi.connected:
        print("[MotorControl] pigpio connection failed.")
        return

    stepper_process = Process(target=stepper_worker, args=(movement_queue, shared_data))
    stepper_process.start()

    try:
        while not shared_data['shutdown'].value:
            if shared_data["acquire_points"].value:
                success = False
                try:
                    tle_data = None
                    try:
                        # try to load temp.tle if present
                        tle_list = parse_tle_file("temp.tle") if os.path.exists("temp.tle") else None
                        if tle_list:
                            tle_data = tle_list[0]
                    except Exception:
                        tle_data = None

                    # Use new TLE-guided acquisition (function adapts if debug_mode is True)
                    success = acquire_target_from_tle(pi, shared_data, movement_queue, tle_data)
                except Exception as e:
                    print(f"[MotorControl] Could not run acquisition: {e}")
                # acquire_target_from_tle is responsible for clearing shared_data['acquire_points']
                if not success:
                    shared_data["acquire_points"].value = False
                    print("[MotorControl] Acquisition sequence failed.")
                else:
                    # signal EKF process to initialize
                    shared_data["ekf_start"].value = True
                    print("[MotorControl] Handoff to EKF process initiated.")

            elif shared_data['ekf_running'].value:

        # Use TLETracker's EKF-guided fast tracker

                ok = track_target_with_ekf(pi, shared_data, movement_queue)

        # If tracking fails multiple times, consider fallback (optional)

        # (we keep it simple here; run_ekf_tracker will update ekf_running as needed)

            else:  # Manual control mode
                if shared_data['tilt_up'].value: move(pi, 'up', 1.0, None, movement_queue, shared_data); shared_data[
                    'tilt_up'].value = False
                if shared_data['tilt_down'].value: move(pi, 'down', 1.0, None, movement_queue, shared_data);
                shared_data['tilt_down'].value = False
                if shared_data['pan_left'].value: move(pi, 'left', 1.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_left'].value = False
                if shared_data['pan_right'].value: move(pi, 'right', 1.0, 0.0001, movement_queue, shared_data);
                shared_data['pan_right'].value = False
                if shared_data["go_to_target"].value: track_target(pi, shared_data["target_azimuth"].value,
                                                                   shared_data["target_elevation"].value, 0.0001,
                                                                   movement_queue, shared_data); shared_data[
                    "go_to_target"].value = False

            sleep(0.02)

    finally:
        print("[MotorControl] Shutting down...")
        try:
            movement_queue.put_nowait(None)
        except Exception:
            pass
        stepper_process.join(timeout=3)
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected: pi.stop()
        GPIO.cleanup()