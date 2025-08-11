from multiprocessing import Process, Queue, Array, Value
from webcam.blob_tracker import run_tracking
from LiDAR.Kalman_Filter import run_ekf_tracker
from tracking.tracker import tracking
from hardware_controller import run_hardware_controller
from GUI import run_gui
import time, os, signal, sys

def join_or_escalate(proc, name, timeout=8):
    if proc is None: 
        return
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] {name} still alive after {timeout}s → SIGTERM")
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except Exception as e:
            print(f"[main] SIGTERM {name} failed: {e}")
        proc.join(timeout=3)
    if proc.is_alive():
        print(f"[main] {name} still alive → terminate() (last resort)")
        try:
            proc.terminate()
        except Exception as e:
            print(f"[main] terminate() {name} failed: {e}")
        proc.join(timeout=2)

if __name__ == "__main__":
    # ===== Shared memory setup =====
    lidar_data = Array('d', 3)  # [distance, strength, timestamp]
    backgorund_data = Array('d', 4)
    shutdown_flag = Value('b', False)
    scan_trigger = Value('b', False)
    tilt_up = Value('b', False)
    tilt_down = Value('b', False)
    pan_left = Value('b', False)
    pan_right = Value('b', False)
    flipped = Value('b', False)
    go_to_target = Value('b', False)
    save_background = Value('b', False)
    background_ready = Value('b', False)
    acquire_points = Value('b', False)
    ekf_start = Value('b', False)
    ekf_running = Value('b', False)
    points_buffer = Array('d', 12)
    points_count = Value('i', 0)
    ekf_initialized = Value('b', False)
    estimated_azimuth = Value('d', 0.0)
    estimated_elevation = Value('d', 0.0)
    predicted_azimuth = Value('d', 0.0)
    predicted_elevation = Value('d', 0.0)
    ekf_confidence = Value('d', 0.0)
    lidar_acceptance_range = Array('d', [1.0, 2.0])  # min_m, max_m
    go_to_zero = Value('b', False)
    stepper_busy = Value('b', False)
    background_scan_active = Value('b', False)
    search_mode_active = Value('b', False)
    lidar_track_mode_active = Value('b', False)
    go_to_target = Value('b', False)
    save_background_trigger = Value('b', False)
    target_reached = Value('b', False)
    target_azimuth = Value('d', 0.0)
    target_elevation = Value('d', 0.0)
    stepper_degrees = Value('d', 0.0)  # Current position reported by HW Controller
    servo_degrees = Value('d', 90.0)
    background_path = "background_data.npy"

    direction = Value('i', -1)
    target = Value('i', -1)
    commanding = Value('i', -1)
    stepper_degrees = Value('d', 0.0)
    cumulative_error = Value('d', 0.0)
    servo_degrees = Value('d', 90.0)
    target_azimuth = Value('d', 0.0)
    target_elevation = Value('d', 0.0)
    background_lidar = Array('d', 360 * 90 * 2)  # shape (90,360,2) flattened
    satellite_points = Array('d', 4)
    satellite_detected = Value('b', False)

    shared_data = {
        "lidar_data": lidar_data,
        "background_data": backgorund_data,
        "shutdown": shutdown_flag,
        "direction": direction,
        "target": target,
        "commanding": commanding,
        "scan_trigger": scan_trigger,
        "stepper_degrees": stepper_degrees,
        "cumulative_error": cumulative_error,
        "servo_degrees": servo_degrees,
        "tilt_up": tilt_up,
        "tilt_down": tilt_down,
        "pan_left": pan_left,
        "pan_right": pan_right,
        "flipped": flipped,
        "go_to_target": go_to_target,
        "target_azimuth": target_azimuth,
        "target_elevation": target_elevation,
        "background_lidar": background_lidar,
        "satellite_points": satellite_points,
        "satellite_detected": satellite_detected,
        "save_background": save_background,
        "background_ready": background_ready,
        "background_path": background_path,
        "acquire_points": acquire_points,
        "ekf_start": ekf_start,
        "ekf_running": ekf_running,
        "points_buffer": points_buffer,
        "points_count": points_count,
        "ekf_initialized": ekf_initialized,
        "estimated_azimuth": estimated_azimuth,
        "estimated_elevation": estimated_elevation,
        "predicted_azimuth": predicted_azimuth,
        "predicted_elevation": predicted_elevation,
        "ekf_confidence": ekf_confidence,
        "lidar_acceptance_range": lidar_acceptance_range,
        "go_to_zero": go_to_zero,
        "stepper_busy": stepper_busy,
        "background_scan_active": background_scan_active,
        "search_mode_active": search_mode_active,
        "lidar_track_mode_active": lidar_track_mode_active,
        "save_background_trigger": save_background_trigger,
        "target_reached": target_reached,
    }

    # ===== Start processes (non-daemon; default) =====
    movement_queue = Queue()
    # p1 = Process(target=run_tracking, args=(shared_data,))
    p2 = Process(target=run_hardware_controller, args=(shared_data))
    p3 = Process(target=run_gui, args=(shared_data, movement_queue))
    p5 = Process(target=run_ekf_tracker, args=(shared_data,))

    for p in (p2, p3, p5):  # p1 if you enable it
        p.daemon = False

    # Start
    # p1.start()
    p2.start()
    p3.start()

    p5.start()

    # ===== Handle Ctrl-C / SIGTERM to flip the flag, not kill processes =====
    def _graceful(signum, frame):
        print(f"[main] Received signal {signum}, requesting shutdown...")
        shutdown_flag.value = True
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    try:
        while not shutdown_flag.value:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[main] Ctrl+C pressed")
        shutdown_flag.value = True
    finally:
        print("Terminating processes (graceful)...")
        # Unblock motor worker if it’s waiting on the queue
        try:
            movement_queue.put_nowait(None)
        except Exception:
            pass

        # Give each process a chance to run its finally/cleanup
        # p1: run_tracking (if enabled)
        # join_or_escalate(p1, "Tracking")
        join_or_escalate(p5, "EKF")
        join_or_escalate(p2, "HarwareControll")
        join_or_escalate(p3, "GUI")

        print("Program exited cleanly")
        sys.exit(0)
