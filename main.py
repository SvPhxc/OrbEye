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
from ctypes import c_char

# --- Import process functions from other modules ---
from hardware_controller import run_hardware_controller
from GUI import run_gui
from tracking_logic import run_tracker_process
from tle_generator import run_tle_generator


# --- FIX: Define a named, picklable class for the ctypes array ---
# This class is defined at the top level so other processes can import/find it.
class CCharArray(c_char * 256):
    pass


class CCharArrayLarge(c_char * 1024):
    pass


# --- System-wide Enums ---
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
        "lidar_data": manager.Array('d', [0.0, 0.0, 0.0]),
        "lidar_position": manager.Array('d', [0.0, 0.0]),
        "lidar_valid": manager.Value('b', False),

        # --- Background Scanning ---
        "background_scan_active": manager.Value('b', False),
        "background_scan_paused": manager.Value('b', False),
        # FIX: Use the named, picklable class here.
        "background_path": manager.Value(CCharArray, b'background_scan.npy'),
        "scan_progress": manager.Value('d', 0.0),

        # --- Tracker Control & Output ---
        "acquire_points": manager.Value('b', False),
        "satellite_points": manager.Array('d', [0.0] * 5),

        # --- TLE Generation ---
        "observer_lat": manager.Value('d', 34.0522),
        "observer_lon": manager.Value('d', -118.2437),
        "observer_alt": manager.Value('d', 71.0),
        "generate_tle": manager.Value('b', False),
        "tracking_history": manager.list(),
        # FIX: Use the named, picklable class for the larger TLE string as well.
        "generated_tle": manager.Value(CCharArrayLarge, b"No TLE generated yet."),

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
    return manager, shared_data


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
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.join(timeout=3)
        except Exception as e:
            print(f"[main] SIGTERM for '{name}' failed: {e}")
    if proc.is_alive():
        print(f"[main] '{name}' could not be terminated. Forcing with terminate()...")
        proc.terminate()
        proc.join(timeout=2)


if __name__ == "__main__":
    print("[main] Initializing shared memory space...")
    manager, shared_data = create_shared_data_manager()

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
            any_process_died = False
            for name, p in processes.items():
                if not p.is_alive():
                    print(f"[main] CRITICAL ERROR: Process '{name}' has terminated unexpectedly.")
                    any_process_died = True

            if any_process_died:
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