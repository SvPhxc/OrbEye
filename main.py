#!/usr/bin/env python3
"""
Main application entry point for the Integrated Target Tracker system.
This script initializes a shared memory space using a multiprocessing.Manager
and launches all concurrent processes required for the system to operate.
"""

import sys
import os
import signal
import time
from multiprocessing import Process, Manager
from ctypes import c_char

# --- Import process functions from other modules ---
# These files (hardware_controller.py, etc.) must be in the same directory.
from hardware_controller import run_hardware_controller
from GUI import run_gui
from tracking_logic import run_tracker_process
from tle_generator import run_tle_generator


# --- Define Picklable Classes for Shared Strings ---
# This is a necessary step to allow ctypes character arrays (used for strings)
# to be shared between processes via the Manager. The Manager needs a named
# class that it can pickle and reconstruct in its own process.

class CCharArray(c_char * 256):
    """A shared string buffer with a 256-character limit."""
    pass


class CCharArrayLarge(c_char * 1024):
    """A larger shared string buffer with a 1024-character limit."""
    pass


def join_or_escalate(proc, name, timeout=5):
    """
    Helper function to gracefully terminate a process.
    It first tries a clean join, then sends SIGTERM, and finally forces
    termination if the process does not exit.
    """
    if proc is None or not proc.is_alive():
        print(f"[main] '{name}' is already terminated.")
        return
    print(f"[main] Waiting for '{name}' to terminate...")
    proc.join(timeout=timeout)
    if proc.is_alive():
        print(f"[main] Process '{name}' is still alive. Sending SIGTERM...")
        try:
            # Use os.kill for a more reliable shutdown signal on Linux/macOS
            if sys.platform != "win32":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()  # Fallback for Windows
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
    # It uses the Manager for all shared objects for consistency and simplicity.
    shared_data = {
        # --- System Control & State ---
        "shutdown": manager.Value('b', False),
        "debug_mode": manager.Value('b', False),
        "demo": manager.Value('b', False),
        "system_state": manager.Value('i', 0),  # Based on the SystemState Enum

        # --- TLE Generation ---
        "observer_lat": manager.Value('d', 34.0522),  # Default: Los Angeles, CA
        "observer_lon": manager.Value('d', -118.2437),
        "observer_alt": manager.Value('d', 71.0),
        "generate_tle": manager.Value('b', False),
        "tracking_history": manager.list(),  # A shared list for TLE points
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
        "movement_priority": manager.Value('i', 0),  # Based on the Priority Enum

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
        """Signal handler to initiate a clean shutdown of the application."""
        if not shared_data["shutdown"].value:
            print(f"\n[main] Signal {signum} received. Requesting global shutdown...")
            shared_data["shutdown"].value = True


    # Register the signal handler for interrupt (Ctrl+C) and termination signals
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        print("[main] Starting all processes...")
        for name, p in processes.items():
            # Daemonic processes are not allowed to have children,
            # so we ensure they are regular processes for stability.
            p.daemon = False
            p.start()
            print(f"  - Started {name} (PID: {p.pid})")
        print("[main] All processes are running. System is active.")

        # The main process now enters a monitoring loop.
        # It will initiate a shutdown if any child process dies unexpectedly.
        while not shared_data["shutdown"].value:
            if not all(p.is_alive() for p in processes.values()):
                for name, p in processes.items():
                    if not p.is_alive():
                        print(f"[main] CRITICAL ERROR: Process '{name}' has terminated unexpectedly.")
                print("[main] Initiating shutdown due to process failure.")
                shared_data["shutdown"].value = True
                break
            time.sleep(0.1)  # Check process health every 100ms

    except (KeyboardInterrupt, SystemExit):
        # This handles cases where the main process itself is interrupted
        if not shared_data["shutdown"].value:
            shared_data["shutdown"].value = True
    finally:
        print("\n[main] Starting shutdown sequence...")
        # Terminate processes in a logical order
        join_or_escalate(processes["GUI"], "GUI")
        join_or_escalate(processes["TrackingLogic"], "TrackingLogic")
        join_or_escalate(processes["TLEGenerator"], "TLEGenerator")
        join_or_escalate(processes["HardwareController"], "HardwareController")
        print("[main] All processes have been terminated.")

        # Finally, explicitly shut down the manager process to release its resources
        print("[main] Shutting down manager...")
        manager.shutdown()
        print("[main] Program exited cleanly.")
        sys.exit(0)