# main.py (Updated)

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value, Manager
import traceback

# Import all process functions
from hardware_controller import run_hardware_controller
from GUI import run_gui
from tracker_logic import run_tracker_logic # <-- IMPORT THE NEW MODULE

def join_or_escalate(proc, name, timeout=5):
    """Helper function to gracefully terminate processes."""
    # ... (this function remains unchanged)
    if proc is None or not proc.is_alive():
        print(f"[main] '{name}' is already terminated.")
        return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive. Sending SIGTERM...")
        try:
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] Process '{name}' will not die. Forcing terminate()...");
        try:
            proc.terminate()
            proc.join(timeout=2)
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")


if __name__ == "__main__":
    print("[main] Initializing shared memory space...")

    manager = Manager()

    shared_data = {
        # --- System Control ---
        "shutdown": Value('b', False),
        "debug_mode": Value('b', False),

        # --- Hardware & Movement ---
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "target_azimuth": Value('d', 90.0),
        "target_elevation": Value('d', 45.0),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),

        # --- LiDAR Data ---
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),
        "lidar_acceptance_range": Array('d', [3.0, 50.0]),
        "lidar_port": manager.Value('c', "/dev/serial0"),

        # --- Background Scan ---
        "background_scan_active": Value('b', False),
        "background_path": manager.Value('c', "background_scan.npy"),

        # --- Auto-Tracking (NEW SECTION) ---
        "auto_track_active": Value('b', False),
        "tracker_status": Value('i', 0), # 0:IDLE, 1:ACQUIRING, 2:TRACKING, 3:REACQUIRING
        "tracker_target_pan": Value('d', 0.0),
        "tracker_target_tilt": Value('d', 0.0),
        # ------------------------------------

        # --- Acquirer (for EKF init) ---
        "acquire_points": Value('b', False),
        "acquirer_status": Value('i', 0),
        "points_buffer": Array('d', [0.0] * 15),
        "points_count": Value('i', 0),

        # (The rest of the shared_data dictionary remains the same)
        # ...
        "lidar_track_mode_active": Value('b', False),
        "satellite_detected": Value('b', False),
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0, 0.0]),
        "ekf_start": Value('b', False),
        "ekf_running": Value('b', False),
        "ekf_initialized": Value('b', False),
        "ekf_confidence": Value('d', 0.0),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "estimated_azimuth": Value('d', 0.0),
        "estimated_elevation": Value('d', 0.0),
        "generate_plot_on_stop": Value('b', False),
        "heatmap_measurement": Array('d', [0.0, 0.0, 0.0]),
        "heatmap_measurement_updated": Value('b', False),
    }

    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data,)),
        "TrackerLogic": Process(target=run_tracker_logic, args=(shared_data,)), # <-- ADD THE NEW PROCESS
    }

    def _graceful_shutdown(signum, frame):
        # ... (this function remains unchanged) ...
        if not shared_data["shutdown"].value:
            print(f"\n[main] Signal {signum} received. Requesting global shutdown...")
            shared_data["shutdown"].value = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        # ... (this block remains unchanged) ...
        print("[main] Starting all processes...")
        for name, p in processes.items():
            p.daemon = False
            p.start()
            print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        while not shared_data["shutdown"].value:
            running_procs = [p for p in processes.values() if p.is_alive()]
            if len(running_procs) < len(processes):
                print("[main] A critical process has terminated unexpectedly. Initiating shutdown.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        join_or_escalate(processes.get("GUI"), "GUI")
        join_or_escalate(processes.get("TrackerLogic"), "TrackerLogic") # <-- SHUTDOWN NEW PROCESS
        join_or_escalate(processes.get("HardwareController"), "HardwareController")
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)