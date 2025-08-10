# ==============================================================================
# LiDAR/lidar_handler.py
# ------------------------------------------------------------------------------
# This module reads data from the TF-Mini S, manages the background scan data,
# and performs target detection by comparing live readings to the stored background.
#
# Key Fixes:
# - Debug Mode now correctly uses background subtraction.
# - The detection logic for Debug Mode is based on a significant *range difference*
#   (e.g., a hand being much closer than a wall), rather than signal strength.
# ==============================================================================

import serial
import numpy as np
import time
from multiprocessing import Value
from collections import deque


def read_tfmini_data(serial_port):
    """Reads and parses a 9-byte data frame from the TF-Mini S."""
    buffer = bytearray()
    while True:
        data = serial_port.read(serial_port.in_waiting or 1)
        if not data:
            continue
        buffer += data
        while len(buffer) >= 9:
            if buffer[0] == 0x59 and buffer[1] == 0x59:
                distance = buffer[2] + (buffer[3] << 8)
                strength = buffer[4] + (buffer[5] << 8)
                buffer = buffer[9:]
                return distance, strength
            else:
                # Slide buffer window if no frame header is found
                buffer = buffer[1:]
    return None, None


def detect_satellite_direct_index(current_strength, current_range_cm, az_deg, el_deg, shared_data, bg_index):
    """
    Enhanced detection function that uses different logic for drone vs. debug mode.
    BOTH modes now use the background index.
    """
    az = int(round(az_deg)) % 360
    el = int(round(el_deg))

    # A background reading for this specific angle is required for detection.
    background_data = bg_index.get((az, el))
    if not background_data:
        shared_data["satellite_detected"].value = False
        return False

    bg_strength, bg_range_cm = background_data
    is_debug = shared_data["debug_mode"].value

    detected = False
    if is_debug:
        # --- DEBUG MODE LOGIC: Focus on Range Difference ---
        # A hand is detected if it's significantly closer than the static background.
        range_diff = bg_range_cm - current_range_cm
        min_m, max_m = shared_data["lidar_acceptance_range"]

        # Check if the reading is within the valid 'hand' range and is much closer than the wall.
        if (min_m * 100 <= current_range_cm <= max_m * 100) and (range_diff > 30):  # e.g., 30cm closer
            print(f"[DETECT-DBG] Hand detected! Range diff: {range_diff:.0f}cm")
            detected = True
    else:
        # --- DRONE MODE LOGIC: Multi-criteria Scoring (Original Logic) ---
        detection_score = 0.0

        # Criterion 1: Strength difference (drones are often highly reflective)
        if current_strength > bg_strength + 500:
            detection_score += 2.0

        # Criterion 2: Range difference (can be closer or farther)
        range_diff = abs(current_range_cm - bg_range_cm)
        if range_diff > 100:  # 1m difference
            detection_score += 1.0

        # Criterion 3: Must be within drone acceptance range
        min_m, max_m = shared_data["lidar_acceptance_range"]
        if not (min_m * 100 <= current_range_cm <= max_m * 100):
            detection_score = 0.0  # Disqualify if outside range

        if detection_score >= 2.0:
            print(f"[DETECT] Drone detected! Score: {detection_score:.1f}")
            detected = True

    # If a detection occurred, update the shared memory.
    if detected:
        sp = shared_data["satellite_points"]
        with sp.get_lock():
            sp[0], sp[1], sp[2], sp[3] = az_deg, el_deg, current_strength, current_range_cm
        shared_data["satellite_detected"].value = True
        return True
    else:
        shared_data["satellite_detected"].value = False
        return False


def build_bg_index(path):
    """Builds a dictionary index from the background .npy file for fast lookups."""
    idx = {}
    try:
        bg = np.load(path)
        if bg.ndim == 1 and bg.size % 4 == 0:
            bg = bg.reshape((-1, 4))

        position_groups = {}
        for row in bg:
            pos, dist_cm, strength = int(row[0]), float(row[1]), float(row[2])
            az, el = pos % 360, pos // 360
            if not (0 <= az < 360 and 0 <= el < 90 and np.isfinite(dist_cm) and np.isfinite(strength)):
                continue

            pos_key = (az, el)
            if pos_key not in position_groups:
                position_groups[pos_key] = []
            position_groups[pos_key].append({'distance': dist_cm, 'strength': strength})

        for pos_key, measurements in position_groups.items():
            if not measurements: continue

            if len(measurements) > 1:
                # Use median for robustness against outlier readings during the scan
                avg_distance = np.median([m['distance'] for m in measurements])
                avg_strength = np.median([m['strength'] for m in measurements])
            else:
                avg_distance = measurements[0]['distance']
                avg_strength = measurements[0]['strength']

            idx[pos_key] = (avg_strength, avg_distance)

        print(f"[TFmini] Background index built with {len(idx)} cells.")
    except Exception as e:
        print(f"[TFmini] Could not load or build background index from '{path}': {e}")
    return idx


def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    """Main process for the LiDAR sensor."""
    print("[TFmini] LiDAR process starting...")
    bg_index = {}
    background_array = np.empty((0, 4))  # [pos, dist, str, timestamp]

    lidar_sh = shared_data["lidar_data"]
    stepper_deg, servo_deg = shared_data["stepper_degrees"], shared_data["servo_degrees"]

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial port opened. Reading data...")
            while not shared_data["shutdown"].value:
                distance_cm, strength = read_tfmini_data(ser)
                if distance_cm is None:
                    time.sleep(0.005)
                    continue

                ts = time.time()
                with lidar_sh.get_lock():
                    lidar_sh[0], lidar_sh[1], lidar_sh[2] = float(distance_cm), float(strength), ts

                # --- Background Management ---
                if shared_data["background_ready"].value:
                    bg_index = build_bg_index(shared_data["background_path"])
                    shared_data["background_ready"].value = False

                if shared_data["scan_trigger"].value:
                    az, el = float(stepper_deg.value), float(servo_deg.value)
                    pos = (int(round(el)) * 360) + (int(round(az)) % 360)
                    new_row = np.array([[pos, distance_cm, strength, ts]])
                    background_array = np.append(background_array, new_row, axis=0)

                if shared_data["save_background"].value:
                    if background_array.size > 0:
                        np.save(shared_data["background_path"], background_array)
                        shared_data["background_ready"].value = True
                        print(f"[TFmini] Background data saved: {len(background_array)} points.")
                        background_array = np.empty((0, 4))  # Clear buffer
                    shared_data["save_background"].value = False

                # --- Detection Logic ---
                # Only run detection if a consumer is ready (acquisition or tracking)
                # and if we have a valid background map.
                if (shared_data["acquire_points"].value or shared_data["ekf_running"].value) and bg_index:
                    az, el = float(stepper_deg.value), float(servo_deg.value)
                    detect_satellite_direct_index(strength, distance_cm, az, el, shared_data, bg_index)

                time.sleep(0.005)  # Loop at ~200Hz, faster than LiDAR rate

    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")
    except Exception as e:
        import traceback
        print(f"[TFmini] CRITICAL ERROR in lidar_handler: {e}")
        traceback.print_exc()

    print("[TFmini] Shutting down.")