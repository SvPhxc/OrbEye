# main.py

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value
import traceback

# Import all your process functions
from hardware_controller import run_hardware_controller
from GUI import run_gui
from acquirer import run_acquirer
from active_tracker import run_active_tracker
from kalman_filter import run_ekf_tracker
from heatmap_tracker import run_heatmap_tracker  # For debug mode


def join_or_escalate(proc, name, timeout=5):
    """Helper function to gracefully terminate processes."""
    if proc is None or not proc.is_alive(): return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive. Sending SIGTERM...")
        try:
            os.kill(proc.pid, signal.SIGTERM)
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

    shared_data = {
        # --- System Control ---
        "shutdown": Value('b', False),
        "debug_mode": Value('b', False),

        # --- Hardware & Movement ---
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "target_azimuth": Value('d', 0.0),
        "target_elevation": Value('d', 0.0),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),

        # --- LiDAR Data ---
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),  # dist_cm, strength, timestamp
        "lidar_acceptance_range": Array('d', [3.0, 50.0]),  # min_m, max_m
        "lidar_port": "/dev/serial0",

        # --- Acquirer (for EKF init) ---
        "acquire_points": Value('b', False),
        "acquirer_status": Value('i', 0),  # 0:idle, 1:running, 2:done, 3:failed
        "points_buffer": Array('d', 15),  # az,el,dist,str for 3 points
        "points_count": Value('i', 0),

        # --- Active Tracker (High-Frequency Hunt) ---
        "lidar_track_mode_active": Value('b', False),
        "satellite_detected": Value('b', False),
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0, 0.0]),  # az,el,dist_cm,str,ts

        # --- EKF State ---
        "ekf_start": Value('b', False),
        "ekf_running": Value('b', False),
        "ekf_initialized": Value('b', False),
        "ekf_confidence": Value('d', 0.0),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "estimated_azimuth": Value('d', 0.0),
        "estimated_elevation": Value('d', 0.0),

        # --- Heatmap Tracker (for Debug Mode) ---
        "heatmap_measurement": Array('d', [0.0, 0.0, 0.0]),
        "heatmap_measurement_updated": Value('b', False),
    }

    print("[main] Initializing processes...")
    processes = {
        "GUI": Process(target=run_gui, args=(shared_data, None)),
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "Acquirer": Process(target=run_acquirer, args=(shared_data,)),
        "ActiveTracker": Process(target=run_active_tracker, args=(shared_data,)),
        "HeatmapTracker": Process(target=run_heatmap_tracker, args=(shared_data,)),
        "EKF": Process(target=run_ekf_tracker, args=(shared_data,)),
    }


    def _graceful_shutdown(signum, frame):
        if not shared_data["shutdown"].value:
            print(f"\n[main] Signal {signum} received. Requesting global shutdown...")
            shared_data["shutdown"].value = True


    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        print("[main] Starting all processes...")
        for name, p in processes.items():
            p.daemon = False
            p.start()
            print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        # Main process waits for shutdown signal or critical process failure
        while not shared_data["shutdown"].value:
            if not all(p.is_alive() for p in processes.values()):
                print("[main] A critical process has terminated. Initiating shutdown.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.2)

    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["Acquirer"], "Acquirer")
        join_or_escalate(processes["ActiveTracker"], "ActiveTracker")
        join_or_escalate(processes["HeatmapTracker"], "HeatmapTracker")
        join_or_escalate(processes["EKF"], "EKF")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)