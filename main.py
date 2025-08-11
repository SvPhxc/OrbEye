# File: main.py

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value

# --- Import the process entry points ---
from hardware_controller import run_hardware_controller
from GUI import run_gui
from LiDAR.Kalman_Filter import run_ekf_tracker

def join_or_escalate(proc, name, timeout=5):
    """Helper to gracefully terminate a process."""
    if proc is None: return
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] {name} is still alive after {timeout}s. Terminating.")
        try:
            os.kill(proc.pid, signal.SIGTERM)
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        except Exception as e:
            print(f"[main] Error terminating {name}: {e}")

if __name__ == "__main__":
    # ===== Shared Memory Setup =====
    shared_data = {
        # --- System-wide ---
        "shutdown": Value('b', False),
        "lidar_port": "/dev/serial0",
        "debug_mode": Value('b', False),
        "generate_plot_on_stop": Value('b', True),

        # --- LiDAR & Detection ---
        "lidar_data": Array('d', [0.0, 0.0, 0.0]), # [dist_cm, strength, timestamp]
        "background_path": "background_data.npy",
        "save_background": Value('b', False),
        "background_ready": Value('b', False),

        # --- Motor & Hardware State ---
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),

        # --- GUI to Hardware Control Flags ---
        "background_scan_active": Value('b', False),
        "acquire_points": Value('b', False),
        "go_to_target": Value('b', False),
        "target_azimuth": Value('d', 0.0),
        "target_elevation": Value('d', 0.0),
        "tilt_up": Value('b', False),
        "tilt_down": Value('b', False),
        "pan_left": Value('b', False),
        "pan_right": Value('b', False),

        # --- EKF Control and Data ---
        "ekf_start": Value('b', False),
        "ekf_running": Value('b', False),
        "ekf_initialized": Value('b', False),
        "points_buffer": Array('d', 15), # 3 points * 5 values [az,el,dist,str,ts]
        "points_count": Value('i', 0),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "ekf_confidence": Value('d', 0.0),
        "lidar_acceptance_range": Array('d', [3.0, 12.0]),
    }

    # ===== Start Processes =====
    processes = {
        "HardwareCtrl": Process(target=run_hardware_controller, args=(shared_data,)),
        "EKF": Process(target=run_ekf_tracker, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data, None))
    }

    for name, p in processes.items():
        p.daemon = False
        p.start()
        print(f"[main] Started {name} process (PID: {p.pid}).")

    def _graceful_shutdown(signum, frame):
        print(f"\n[main] Signal {signum} received, requesting shutdown...")
        shared_data["shutdown"].value = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        # Main thread waits for shutdown signal
        while not shared_data["shutdown"].value:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        pass # Handle Ctrl+C press
    finally:
        print("[main] Shutdown sequence initiated...")
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["EKF"], "EKF")
        join_or_escalate(processes["HardwareCtrl"], "HardwareCtrl")
        print("[main] Program exited cleanly.")
        sys.exit(0)