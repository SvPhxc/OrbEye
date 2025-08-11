# main.py

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value, Queue

from hardware_controller import run_hardware_controller
from GUI import run_gui

# --- Placeholder for the EKF file, assuming it exists at LiDAR/Kalman_Filter.py ---
# We will create a dummy function here to prevent import errors if the file is missing
try:
    from LiDAR.Kalman_Filter import run_ekf_tracker
except ImportError:
    print("[main] WARNING: LiDAR/Kalman_Filter.py not found. Using a placeholder.")
    def run_ekf_tracker(shared_data):
        print("[EKF] EKF process started (placeholder).")
        # The line that caused the error is here, so we must check for the key
        if 'points_count' in shared_data:
            print(f"[EKF] Placeholder sees points_count: {shared_data['points_count'].value}")
        while not shared_data["shutdown"].value:
            time.sleep(0.1)
        print("[EKF] EKF process shutting down.")


def join_or_escalate(proc, name, timeout=5):
    """A robust function to join a process."""
    if proc is None or not proc.is_alive():
        return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive after {timeout}s. Sending SIGTERM...")
        try:
            os.kill(proc.pid, signal.SIGTERM)
            proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] Process '{name}' will not die. Forcing terminate() (last resort).")
        try:
            proc.terminate()
            proc.join(timeout=2)
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")

if __name__ == "__main__":
    print("[main] Initializing shared memory space...")

    # ==========================================================================
    # 1. SHARED MEMORY SETUP (with EKF variables restored)
    # ==========================================================================

    shared_data = {
        # --- System-wide Flags ---
        "shutdown": Value('b', False),

        # --- Hardware Controller State Flags ---
        "background_scan_active": Value('b', False),
        "search_mode_active": Value('b', False),
        "lidar_track_mode_active": Value('b', False),
        "go_to_target": Value('b', False),
        "save_background_trigger": Value('b', False),
        "target_reached": Value('b', False),

        # --- Target and Position Data ---
        "target_azimuth": Value('d', 0.0),
        "target_elevation": Value('d', 0.0),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),

        # --- LiDAR Data ---
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),  # [distance_cm, strength, timestamp]

        # --- Target Detection Data ---
        "satellite_detected": Value('b', False),
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0]),  # [az, el, dist, strength]
        "lidar_acceptance_range": Array('d', [3.0, 50.0]),

        # --- EKF Related Data ---
        "ekf_start": Value('b', False),
        "ekf_running": Value('b', False),
        "acquire_points": Value('b', False),
        "generate_plot_on_stop": Value('b', False),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "debug_mode": Value('b', False),

        # --- FIX: EKF Point Buffer and Counter (restored) ---
        # Buffer for 3 points: [az, el, dist, str] * 3 = 12 doubles
        "points_buffer": Array('d', 12),
        "points_count": Value('i', 0),
        # ----------------------------------------------------

        # --- Configuration Data ---
        "lidar_port": "/dev/serial0",
        "background_path": "background_data.npy",
    }

    # ==========================================================================
    # 2. PROCESS INITIALIZATION
    # ==========================================================================
    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data, None)),
        "EKF": Process(target=run_ekf_tracker, args=(shared_data,)),
    }

    # ==========================================================================
    # 3. STARTUP & SHUTDOWN HANDLING
    # ==========================================================================
    def _graceful_shutdown(signum, frame):
        if not shared_data["shutdown"].value:
            print(f"\n[main] Received signal {signum}. Requesting global shutdown...")
            shared_data["shutdown"].value = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        print("[main] Starting all processes...")
        for name, p in processes.items():
            p.daemon = False
            p.start()
        print("[main] All processes are running. System is active.")

        while not shared_data["shutdown"].value:
            # Check if any vital process has died, and trigger shutdown if so
            if not processes["GUI"].is_alive() or not processes["HardwareController"].is_alive():
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
        join_or_escalate(processes["EKF"], "EKF")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)