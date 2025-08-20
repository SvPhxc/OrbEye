#!/usr/bin/env python3
"""
Main application entry point for the Integrated Target Tracker system.
This script initializes a shared memory space and launches all concurrent processes.
"""

import sys
import os
import signal
import time
from multiprocessing import Process, Manager
from ctypes import c_char

# --- Import process functions from other modules ---
from hardware_controller import run_hardware_controller
from GUI import run_gui
from tracking_logic import run_tracker_process
from tle_generator import run_tle_generator


# --- Define a picklable class for ctypes char arrays to enable sharing ---
class CCharArray(c_char * 256):
    pass


class CCharArrayLarge(c_char * 1024):
    pass


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
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] Process '{name}' will not die. Forcing terminate()...")
        try:
            proc.terminate()
            proc.join(timeout=2)
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")


if __name__ == "__main__":
    print("[main] Initializing shared memory space...")
    manager = Manager()

    # This dictionary defines the complete shared state for all processes.
    # It has been cleaned to remove old EKF keys and add required keys for
    # the new hardware_controller and tracking_logic.
    shared_data = {
        # --- System Control & State ---
        "shutdown": manager.Value('b', False),
        "debug_mode": manager.Value('b', False),
        "demo": manager.Value('b', False),
        "system_state": manager.Value('i', 0),  # 0:IDLE, 1:MOVING, 2:SCANNING, etc.

        # --- TLE Generation ---
        "observer_lat": manager.Value('d', 34.0522),  # Default: Los Angeles, CA
        "observer_lon": manager.Value('d', -118.2437),
        "observer_alt": manager.Value('d', 71.0),
        "generate_tle": manager.Value('b', False),
        "tracking_history": manager.list(),  # For storing points for TLE generation
        "generated_tle": manager.Value(CCharArrayLarge, b"No TLE generated yet."),

        # --- Hardware & Movement Control ---
        "stepper_degrees": manager.Value('d', 0.0),
        "servo_degrees": manager.Value('d', 45.0),
        "target_azimuth": manager.Value('d', 90.0),
        "target_elevation": manager.Value('d', 45.0),
        "go_to_target": manager.Value('b', False),
        "target_reached": manager.Value('b', False),
        "movement_request_id": manager.Value('i', 0),
        "movement_complete_id": manager.Value('i', 0),
        "movement_priority": manager.Value('i', 0),  # 0:NORMAL, 1:HIGH, 2:CRITICAL

        # --- LiDAR Data ---
        "lidar_data": manager.Array('d', [0.0, 0.0, 0.0]),  # dist_cm, strength, timestamp
        "lidar_position": manager.Array('d', [0.0, 0.0]),  # az, el when read
        "lidar_valid": manager.Value('b', False),

        # --- Background Scanning ---
        "background_scan_active": manager.Value('b', False),
        "background_scan_paused": manager.Value('b', False),
        "background_path": manager.Value(CCharArray, b'background_scan.npy'),
        "scan_progress": manager.Value('d', 0.0),

        # --- Tracking Logic Control & Output ---
        "acquire_points": manager.Value('b', False),  # Trigger for acquisition scan
        "satellite_points": manager.Array('d', [0.0] * 5),  # az, el, dist, str, time

        # --- Synchronization Locks ---
        "state_lock": manager.Lock(),
        "movement_lock": manager.Lock(),
        "lidar_lock": manager.Lock(),
    }

    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data,)),
        "TrackingLogic": Process(target=run_tracker_process, args=(shared_data,)),
        "TLEGenerator": Process(target=run_tle_generator, args=(shared_data,)),
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

        while not shared_data["shutdown"].value:
            # Check if any critical process has died
            if not all(p.is_alive() for p in processes.values()):
                for name, p in processes.items():
                    if not p.is_alive():
                        print(f"[main] CRITICAL ERROR: Process '{name}' has terminated unexpectedly.")
                print("[main] Initiating shutdown due to process failure.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["TrackingLogic"], "TrackingLogic")
        join_or_escalate(processes["TLEGenerator"], "TLEGenerator")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated.")

        print("[main] Shutting down manager...")
        manager.shutdown()
        print("[main] Program exited cleanly.")
        sys.exit(0)