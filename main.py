#!/usr/bin/env python3
"""
Main application entry point for the Integrated Target Tracker system.
This script initializes the shared memory space and launches all the
concurrent processes required for the system to operate:
- Hardware Controller
- Tracking Logic
- GUI
- TLE Generator
"""

import sys
import os
import signal
import time
from multiprocessing import Process, Manager, Value, Array, Lock
from enum import Enum

# --- Import process functions from other modules ---
# These imports assume that the corresponding files (hardware_controller.py,
# GUI.py, tracking_logic.py, tle_generator.py) are in the same directory
# or in Python's path.
from hardware_controller import run_hardware_controller
from GUI import run_gui
from tracking_logic import run_tracker_process
from tle_generator import run_tle_generator


# --- System-wide Enums ---
# It's good practice to define these in a central place if they are used
# by multiple processes via the shared state.

class SystemState(Enum):
    IDLE = 0
    MOVING = 1
    SCANNING = 2
    TRACKER_MOVE = 3
    ERROR = 4
    SHUTDOWN = 5
    PAUSED = 6


class Priority(Enum):
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2


def create_shared_data_manager():
    """
    Creates and returns a manager instance and the fully structured
    shared data dictionary for the entire application.
    This is the single source of truth for the system's shared state.
    """
    manager = Manager()

    shared_data = {
        # --- System Control & State ---
        "shutdown": manager.Value('b', False),
        "system_state": manager.Value('i', SystemState.IDLE.value),
        "last_state_change": manager.Value('d', time.time()),
        "debug_mode": manager.Value('b', False),
        "demo": manager.Value('b', False),

        # --- Position & Movement ---
        "stepper_degrees": manager.Value('d', 0.0),
        "servo_degrees": manager.Value('d', 45.0),
        "target_azimuth": manager.Value('d', 0.0),
        "target_elevation": manager.Value('d', 45.0),
        "go_to_target": manager.Value('b', False),
        "target_reached": manager.Value('b', False),
        "movement_request_id": manager.Value('i', 0),
        "movement_complete_id": manager.Value('i', 0),
        "movement_priority": manager.Value('i', Priority.NORMAL.value),

        # --- LiDAR Data ---
        "lidar_data": manager.Array('d', [0.0, 0.0, 0.0]),  # dist_cm, strength, timestamp
        "lidar_position": manager.Array('d', [0.0, 0.0]),  # az, el when read
        "lidar_valid": manager.Value('b', False),

        # --- Background Scanning ---
        "background_scan_active": manager.Value('b', False),
        "background_scan_paused": manager.Value('b', False),
        "background_path": manager.Array('c', b'background_scan.npy'[:256]), # Use manager.Array for C-string
        "scan_progress": manager.Value('d', 0.0),

        # --- Tracker Control & Output ---
        "acquire_points": manager.Value('b', False),
        "satellite_points": manager.Array('d', [0.0] * 5),  # az, el, dist, str, time

        # --- TLE Generation ---
        "observer_lat": manager.Value('d', 34.0522),  # Default: Los Angeles, CA
        "observer_lon": manager.Value('d', -118.2437),
        "observer_alt": manager.Value('d', 71.0),      # Altitude in meters
        "generate_tle": manager.Value('b', False),
        "tracking_history": manager.list(),  # Stores (az, el, dist, str, ts) for TLE
        "generated_tle": manager.Value('c', b"No TLE generated yet."[:1024]),

        # --- Synchronization Locks ---
        "state_lock": manager.Lock(),
        "movement_lock": manager.Lock(),
        "lidar_lock": manager.Lock(),

        # --- System Statistics ---
        "system_uptime": manager.Value('d', time.time()),
        "total_movements": manager.Value('i', 0),
        "failed_movements": manager.Value('i', 0),
        "lidar_reads": manager.Value('i', 0),
    }
    return shared_data


def join_or_escalate(proc, name, timeout=5):
    """Helper function to gracefully terminate processes."""
    if proc is None or not proc.is_alive():
        print(f"[main] '{name}' is already terminated.")
        return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] '{name}' is still alive. Sending SIGTERM...")
        try:
            # Use os.kill for more robust termination on non-Windows systems
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()  # Fallback for Windows
            proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] '{name}' could not be terminated. Forcing with terminate()...")
        try:
            proc.terminate()
            proc.join(timeout=2)
        except Exception as e:
            print(f"[main] terminate() for '{name}' failed: {e}")


if __name__ == "__main__":
    print("[main] Initializing shared memory space...")
    shared_data = create_shared_data_manager()

    print("[main] Initializing processes...")
    processes = {
        "HardwareController": Process(target=run_hardware_controller, args=(shared_data,)),
        "TrackingLogic": Process(target=run_tracker_process, args=(shared_data,)),
        "GUI": Process(target=run_gui, args=(shared_data,)),
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
            p.start()
            print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        # Main thread monitors for shutdown or process failure
        while not shared_data["shutdown"].value:
            if not all(p.is_alive() for p in processes.values()):
                print("[main] A critical process has terminated unexpectedly. Initiating shutdown.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.1) # Check for failures every 100ms

    except (KeyboardInterrupt, SystemExit):
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        # Terminate processes in a safe order (e.g., UI and logic first)
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["TrackingLogic"], "TrackingLogic")
        join_or_escalate(processes["TLEGenerator"], "TLEGenerator")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated. Program exited cleanly.")
        sys.exit(0)