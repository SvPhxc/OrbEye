import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value, Queue

# --- Import the run functions from your process files ---
# This assumes your directory structure is clean.
# For example:
# /main.py
# /GUI.py
# /hardware_controller.py
# /LiDAR/Kalman_Filter.py

from hardware_controller import run_hardware_controller
from GUI import run_gui
from LiDAR.Kalman_Filter import run_ekf_tracker


def join_or_escalate(proc, name, timeout=5):
    """
    A robust function to join a process, giving it time to shut down
    gracefully before forcing termination.
    """
    if proc is None:
        return
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive after {timeout}s. Sending SIGTERM...")
        try:
            # In POSIX systems (Linux, macOS), this is a more graceful kill signal
            os.kill(proc.pid, signal.SIGTERM)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
        proc.join(timeout=3)
    if proc.is_alive():
        print(f"[main] Process '{name}' will not die. Forcing terminate() (last resort).")
        try:
            # terminate() is less graceful and should be a last resort
            proc.terminate()
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")
        proc.join(timeout=2)


if __name__ == "__main__":
    print("[main] Initializing shared memory space...")

    # ==========================================================================
    # 1. SHARED MEMORY SETUP
    # Define all variables that need to be accessed by multiple processes.
    # ==========================================================================

    # --- System-wide Flags ---
    shutdown_flag = Value('b', False)

    # --- Hardware Controller State Flags (set by GUI, read by HW Controller) ---
    background_scan_active = Value('b', False)
    search_mode_active = Value('b', False)
    lidar_track_mode_active = Value('b', False)
    go_to_target = Value('b', False)
    save_background_trigger = Value('b', False)
    target_reached = Value('b', False)

    # --- Target and Position Data ---
    target_azimuth = Value('d', 0.0)
    target_elevation = Value('d', 0.0)
    stepper_degrees = Value('d', 0.0)  # Current position reported by HW Controller
    servo_degrees = Value('d', 90.0)  # Current position reported by HW Controller

    # --- LiDAR Data (written by HW Controller, read by GUI/EKF) ---
    lidar_data = Array('d', [0.0, 0.0, 0.0])  # [distance_cm, strength, timestamp]

    # --- Target Detection Data (written by HW Controller, read by EKF/GUI) ---
    satellite_detected = Value('b', False)
    satellite_points = Array('d', [0.0, 0.0, 0.0, 0.0])  # [az, el, dist, strength]
    lidar_acceptance_range = Array('d', [3.0, 50.0])  # Default drone range [min_m, max_m]

    # --- EKF Related Data ---
    ekf_running = Value('b', False)  # GUI can toggle this to enable/disable EKF calculations
    acquire_points = Value('b', False)  # GUI triggers this for EKF initialization
    generate_plot_on_stop = Value('b', False)  # GUI requests EKF to generate a final plot
    predicted_azimuth = Value('d', 0.0)  # EKF writes, HW Controller reads for tracking
    predicted_elevation = Value('d', 0.0)  # EKF writes, HW Controller reads for tracking
    debug_mode = Value('b', False)  # Toggled by GUI to change acceptance range, etc.

    # --- Configuration Data ---
    # These are treated as constants but included here for easy access by all processes.
    lidar_port_str = "/dev/serial0"
    background_path_str = "background_data.npy"

    # ==========================================================================
    # 2. SHARED DATA DICTIONARY
    # Pack all shared objects into a single dictionary for easy passing.
    # ==========================================================================
    shared_data = {
        "shutdown": shutdown_flag,
        "background_scan_active": background_scan_active,
        "search_mode_active": search_mode_active,
        "lidar_track_mode_active": lidar_track_mode_active,
        "go_to_target": go_to_target,
        "save_background_trigger": save_background_trigger,
        "target_reached": target_reached,
        "target_azimuth": target_azimuth,
        "target_elevation": target_elevation,
        "stepper_degrees": stepper_degrees,
        "servo_degrees": servo_degrees,
        "lidar_data": lidar_data,
        "satellite_detected": satellite_detected,
        "satellite_points": satellite_points,
        "lidar_acceptance_range": lidar_acceptance_range,
        "ekf_running": ekf_running,
        "acquire_points": acquire_points,
        "generate_plot_on_stop": generate_plot_on_stop,
        "predicted_azimuth": predicted_azimuth,
        "predicted_elevation": predicted_elevation,
        "debug_mode": debug_mode,
        "lidar_port": lidar_port_str,
        "background_path": background_path_str,
    }

    # ==========================================================================
    # 3. PROCESS INITIALIZATION
    # Create the Process objects for each major component of the application.
    # ==========================================================================
    print("[main] Initializing processes...")

    # The single, unified process for all hardware control
    p_hw = Process(target=run_hardware_controller, args=(shared_data,))

    # The GUI process
    # The movement_queue is no longer used by the GUI, so we pass None.
    p_gui = Process(target=run_gui, args=(shared_data, None))

    # The EKF tracker process
    p_ekf = Process(target=run_ekf_tracker, args=(shared_data,))

    processes = {
        "HardwareController": p_hw,
        "GUI": p_gui,
        "EKF": p_ekf,
    }

    # Ensure processes are not daemonic, allowing for graceful shutdown
    for p in processes.values():
        p.daemon = False


    # ==========================================================================
    # 4. STARTUP & SHUTDOWN HANDLING
    # ==========================================================================

    # --- Graceful Shutdown Handler ---
    def _graceful_shutdown(signum, frame):
        print(f"\n[main] Received signal {signum}. Requesting global shutdown...")
        shared_data["shutdown"].value = True  # Signal all processes to stop their loops


    signal.signal(signal.SIGINT, _graceful_shutdown)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, _graceful_shutdown)  # Handle `kill` command

    try:
        # --- Start all processes ---
        print("[main] Starting all processes...")
        for name, p in processes.items():
            print(f"  - Starting {name}...")
            p.start()
        print("[main] All processes are running. System is active.")

        # --- Keep main process alive ---
        # The main process just waits for the shutdown signal.
        while not shared_data["shutdown"].value:
            time.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        # This block is redundant due to the signal handler but is good practice.
        if not shared_data["shutdown"].value:
            print("[main] Main loop interrupted. Requesting shutdown...")
            shared_data["shutdown"].value = True

    finally:
        # --- Clean Shutdown Sequence ---
        print("\n[main] Starting shutdown sequence...")

        # Shut down in a logical order: brains first, then body.
        # This gives the GUI and EKF a chance to finish before the hardware stops.
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["EKF"], "EKF")
        join_or_escalate(processes["Hardware