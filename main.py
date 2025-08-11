# main.py

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value

from hardware_controller import run_hardware_controller
from GUI import run_gui
from acquirer import run_acquirer
from active_tracker import run_active_tracker # <-- IMPORT THE NEW TRACKING LOGIC

try:
    from LiDAR.Kalman_Filter import run_ekf_tracker
except ImportError:
    print("[main] WARNING: LiDAR/Kalman_Filter.py not found. Using a placeholder.")
    def run_ekf_tracker(shared_data):
        print("[EKF] EKF process started (placeholder).")
        while not shared_data["shutdown"].value: time.sleep(0.1)
        print("[EKF] EKF process shutting down.")

def join_or_escalate(proc, name, timeout=5):
    if proc is None or not proc.is_alive(): return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive. Sending SIGTERM...")
        try:
            os.kill(proc.pid, signal.SIGTERM); proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] Process '{name}' will not die. Forcing terminate()...");
        try:
            proc.terminate(); proc.join(timeout=2)
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")

if __name__ == "__main__":
    print("[main] Initializing shared memory space...")

    shared_data = {
        "shutdown": Value('b', False),
        "background_scan_active": Value('b', False),
        "lidar_track_mode_active": Value('b', False), # The new active_tracker listens to this
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "target_azimuth": Value('d', 0.0), "target_elevation": Value('d', 0.0),
        "stepper_degrees": Value('d', 0.0), "servo_degrees": Value('d', 90.0),
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),
        "satellite_detected": Value('b', False), # Used by active_tracker->EKF
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0, 0.0]), # az, el, dist, str, ts
        "lidar_acceptance_range": Array('d', [3.0, 50.0]),
        "acquire_points": Value('b', False),
        "acquirer_status": Value('i', 0),
        "points_buffer": Array('d', 15),
        "points_count": Value('i', 0),
        "ekf_start": Value('b', False), "ekf_running": Value('b', False),
        "ekf_initialized": Value('b', False), "ekf_confidence": Value('d', 0.0),
        "generate_plot_on_stop": Value('b', False),
        "predicted_azimuth": Value('d', 0.0), "predicted_elevation": Value('d', 0.0),
        "debug_mode": Value('b', False),
        "lidar_port": "/dev/serial0", "background_path": "background_data.npy",
    }

    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data, None)),
        "Acquirer": Process(target=run_acquirer, args=(shared_data,)),
        "ActiveTracker": Process(target=run_active_tracker, args=(shared_data,)), # <-- ADD THE NEW PROCESS
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
            p.daemon = False; p.start(); print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        while not shared_data["shutdown"].value:
            if not all(p.is_alive() for p in processes.values()):
                 print("[main] A critical process has terminated. Initiating shutdown.")
                 shared_data["shutdown"].value = True; break
            time.sleep(0.2)
    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value: shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["Acquirer"], "Acquirer")
        join_or_escalate(processes["ActiveTracker"], "ActiveTracker")
        join_or_escalate(processes["EKF"], "EKF")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)