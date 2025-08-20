# gui_only.py
import sys, math
from multiprocessing import Array, Value
from ctypes import c_wchar_p
from PyQt5 import QtCore, QtWidgets

# import your GUI runner exactly as you defined it
from GUI import run_gui  # expects run_gui(shared_data)

def make_shared():
    """
    Build only the keys your GUI reads/writes.
    Types match your GUI's .value and [] accesses.
    """
    shared = {}

    # Core flags & state
    shared["shutdown"]                = Value('b', False)

    # Manual control / movement
    shared["tilt_up"]                 = Value('b', False)
    shared["tilt_down"]               = Value('b', False)
    shared["pan_left"]                = Value('b', False)
    shared["pan_right"]               = Value('b', False)

    shared["go_to_target"]            = Value('b', False)
    shared["target_reached"]          = Value('b', False)
    shared["target_azimuth"]          = Value('d', 0.0)
    shared["target_elevation"]        = Value('d', 0.0)

    # Controller status (your GUI checks this exact name)
    shared["acquirer_status"]         = Value('i', 0)  # 0=idle,1=acquiring

    # Modes
    shared["background_scan_active"]  = Value('b', False)
    shared["lidar_track_mode_active"] = Value('b', False)
    shared["reactive_mode"]           = Value('b', False)
    shared["debug_mode"]              = Value('b', False)

    # EKF / acquisition
    shared["acquire_points"]          = Value('b', False)
    shared["generate_plot_on_stop"]   = Value('b', False)

    # Angles & LiDAR
    shared["stepper_degrees"]         = Value('d', 0.0)   # pan
    shared["servo_degrees"]           = Value('d', 0.0)   # tilt
    shared["lidar_data"]              = Array('d', 3)     # [distance(cm), strength, timestamp]
    shared["lidar_data"][0] = 0.0
    shared["lidar_data"][1] = 0.0
    shared["lidar_data"][2] = 0.0

    # Acceptance range (GUI rewrites both)
    shared["lidar_acceptance_range"]  = Array('d', [3.0, 50.0])

    # Heatmap source (GUI expects 5 fields: az, el, dist_cm, strength, ts)
    shared["satellite_points"]        = Array('d', 5)
    for i in range(5):
        shared["satellite_points"][i] = 0.0

    # Background file path
    shared["background_path"]         = Value(c_wchar_p, "background_data.npy")

    return shared


def add_simulation(shared):
    """
    Optional: animate pan/tilt, range, and heatmap input so the GUI looks alive.
    Uses a Qt timer so we stay single-process/thread in the GUI.
    """
    timer = QtCore.QTimer()

    state = {"t": 0.0}

    def tick():
        t = state["t"]
        # Sweep pan 0..360, tilt 10..70
        pan = (t * 20.0) % 360.0
        tilt = 40.0 + 30.0 * math.sin(t * 0.7)

        # Fake distance & strength
        dist_cm = 300.0 + 50.0 * math.sin(t * 0.9)   # 25m ± 5m
        strength = 50.0 + 40.0 * abs(math.sin(t * 0.5))
        ts = t

        shared["stepper_degrees"].value = pan
        shared["servo_degrees"].value   = tilt
        shared["lidar_data"][0]         = dist_cm
        shared["lidar_data"][1]         = strength
        shared["lidar_data"][2]         = ts

        # Feed the heatmap source (az, el, dist_cm, strength, ts)
        sp = shared["satellite_points"]
        sp[0] = pan
        sp[1] = tilt
        sp[2] = dist_cm
        sp[3] = strength
        sp[4] = ts

        # Toggle some statuses to see the color changes
        acq = 1 if (int(t) % 12) < 3 else 0
        shared["acquirer_status"].value = acq
        shared["background_scan_active"].value = (int(t) % 12) in (3, 4)
        shared["lidar_track_mode_active"].value = (int(t) % 12) in (5, 6, 7)

        state["t"] += 0.05

    timer.timeout.connect(tick)
    timer.start(50)  # 20 Hz-ish
    return timer


if __name__ == "__main__":
    # --- Parse a tiny flag manually (avoid argparse to keep it minimal)
    simulate = "--simulate" in sys.argv

    # Qt app must be in the main process/thread
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    shared = make_shared()
    sim_timer = add_simulation(shared) if simulate else None

    # Hand control to your existing GUI entrypoint
    # It should create the window and call app.exec_() internally per your code.
    run_gui(shared)
