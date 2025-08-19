# main.py (Updated)

import sys
import os
import signal
import time
from multiprocessing import Process, Array, Value, Manager
import traceback
from tracking_logic import run_tracker_process
from tle_generator import run_tle_generator  # <-- IMPORT THE NEW PROCESS

# Import all process functions
from hardware_controller import run_hardware_controller
from GUI import run_gui


# --- Placeholder imports for other modules to make the system runnable ---


def join_or_escalate(proc, name, timeout=5):
    """Helper function to gracefully terminate processes."""
    if proc is None or not proc.is_alive():
        print(f"[main] '{name}' is already terminated.")
        return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive. Sending SIGTERM...")
        try:
            # Use os.kill on non-Windows platforms for more robust termination
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()  # Fallback for Windows
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

    # Using Manager for complex/dynamic data types that need to be shared
    manager = Manager()

    shared_data = {
        # --- System Control ---
        "shutdown": Value('b', False),
        "debug_mode": Value('b', False),

        # +++ ADDED FOR TLE GENERATION +++
        # --- Observer Location (can be updated from GUI) ---
        "observer_lat": Value('d', 0.0),  # Default: Los Angeles, CA
        "observer_lon": Value('d', 0.0),
        "observer_alt": Value('d', 0.0),  # Altitude in meters

        # --- TLE Data Flow & Control ---
        "generate_tle": Value('b', False),  # Set to True to trigger TLE generation
        "tracking_history": manager.list(),  # Stores tracker points (az, el, dist, str, ts)
        "generated_tle": manager.Value('c', "No TLE generated yet."),  # Stores the output TLE
        # +++ END OF TLE ADDITIONS +++

        # --- Hardware & Movement ---
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "target_azimuth": Value('d', 90.0),
        "target_elevation": Value('d', 45.0),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 90.0),

        # --- LiDAR Data ---
        "lidar_data": Array('d', [0.0, 0.0, 0.0]),  # dist_cm, strength, timestamp
        "lidar_acceptance_range": Array('d', [10.0, 16000.0]),  # min_m, max_m
        "lidar_port": manager.Value('c', "/dev/serial0"),

        # --- Background Scan ---
        "background_scan_active": Value('b', False),
        "save_background_trigger": Value('b', False),
        "background_path": manager.Value('c', "background_scan.npy"),

        # --- Acquirer (for EKF init) ---
        "acquire_points": Value('b', False),
        "acquirer_status": Value('i', 0),  # 0:idle, 1:running, 2:done, 3:failed
        "points_buffer": Array('d', [0.0] * 15),  # az,el,dist,str,ts for 3 points
        "points_count": Value('i', 0),
        "acquirer_timeout": Value('d', 5.0),
        "acquirer_max_points": Value('i', 3),
        "acquirer_min_distance": Value('d', 3.0),
        "acquirer_max_distance": Value('d', 50.0),
        "acquirer_azimuth": Value('d', 0.0),
        "acquirer_elevation": Value('d', 0.0),
        "acquirer_azimuth_step": Value('d', 10.0),
        "acquirer_elevation_step": Value('d', 10.0),
        "tracking_logic_ready": Value('b', False),
        "reactive_mode": Value('b', False),

        # --- Active Tracker (High-Frequency Hunt) ---
        "lidar_track_mode_active": Value('b', False),
        "satellite_detected": Value('b', False),
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0, 0.0]),  # az,el,dist_cm,str,ts
        # NOTE: "hand_tracker_history" has been removed in favor of the more flexible "tracking_history" list.

        # --- EKF State ---
        "ekf_start": Value('b', False),
        "ekf_running": Value('b', False),
        "ekf_initialized": Value('b', False),
        "ekf_confidence": Value('d', 0.0),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "estimated_azimuth": Value('d', 0.0),
        "estimated_elevation": Value('d', 0.0),
        "generate_plot_on_stop": Value('b', False),  # For GUI button

        # --- Heatmap Tracker (for Debug Mode) ---
        "heatmap_measurement": Array('d', [0.0, 0.0, 0.0]),
        "heatmap_measurement_updated": Value('b', False),
    }

    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data,)),
        "TrackingLogic": Process(target=run_tracking_logic, args=(shared_data,)),
        "TLEGenerator": Process(target=run_tle_generator, args=(shared_data,)),  # <-- ADD THE NEW PROCESS
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
            p.daemon = False  # Ensure processes are not auto-killed
            p.start()
            print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        while not shared_data["shutdown"].value:
            running_procs = [p for p in processes.values() if p.is_alive()]
            if len(running_procs) < len(processes):
                print("[main] A critical process has terminated unexpectedly. Initiating shutdown.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.01)

    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        join_or_escalate(processes["TrackingLogic"], "TrackingLogic")
        join_or_escalate(processes["TLEGenerator"], "TLEGenerator")  # <-- ADD TLE PROCESS TO SHUTDOWN
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)