# lidar_handler.py

import serial
import numpy as np
import time


# ... (all other functions like read_tfmini_data, get_background_index remain the same) ...

def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    """
    Manages LiDAR data: reads it, populates background map during scans,
    saves the map, and detects satellites against it.
    """
    try:
        with serial.Serial(port, baudrate, timeout=0.1) as ser:
            print("[LiDAR] Serial opened, reading data...")
            while not shared_data["shutdown"].value:
                distance, strength = read_tfmini_data(ser)
                if distance is not None:
                    # --- Update shared LiDAR data ---
                    with shared_data["lidar_data"].get_lock():
                        shared_data["lidar_data"][0] = distance
                        shared_data["lidar_data"][1] = strength
                        shared_data["lidar_data"][2] = time.time()

                    # --- Background Scan Logic (Unchanged) ---
                    if shared_data["scan_trigger"].value:
                        with shared_data["stepper_degrees"].get_lock():
                            az = shared_data["stepper_degrees"].value
                        with shared_data["servo_degrees"].get_lock():
                            el = shared_data["servo_degrees"].value
                        index = get_background_index(az, el)
                        if index is not None:
                            with shared_data["background_lidar"].get_lock():
                                shared_data["background_lidar"][index] = strength
                                shared_data["background_lidar"][index + 1] = distance

                    # --- Save Background Logic (Unchanged) ---
                    if shared_data["save_background"].value:
                        print("[LiDAR] Saving background data to file...")
                        with shared_data["background_lidar"].get_lock():
                            background_np = np.array(shared_data["background_lidar"]).reshape((90, 360, 2))
                        np.save("background_data.npy", background_np)
                        print("[LiDAR] Background data saved to 'background_data.npy'.")
                        with shared_data["save_background"].get_lock():
                            shared_data["save_background"].value = False

                    # --- NEW: Check acquisition state BEFORE validation ---
                    # If we are in the centering state, we need to log the strongest point found
                    if shared_data["acquisition_state"].value == 1:  # 1 = CENTERING_P1
                        with shared_data["stepper_degrees"].get_lock():
                            az = shared_data["stepper_degrees"].value
                        with shared_data["servo_degrees"].get_lock():
                            el = shared_data["servo_degrees"].value
                        update_best_strength_point(az, el, strength, shared_data)

                    # --- Validate and Detect Anomaly ---
                    validate_lidar_data(distance, strength, shared_data)

                time.sleep(0.005)

    except serial.SerialException as e:
        print(f"[LiDAR] Serial error: {e}")
    print("[LiDAR] Shutting down.")


def validate_lidar_data(distance_cm, strength, shared_data):
    """Validates data and checks if it's a satellite."""
    if not (150 <= distance_cm <= 300 and strength > 3000):
        # Set detected flag to false if data is not valid
        with shared_data["satellite_detected"].get_lock():
            shared_data["satellite_detected"].value = False
        return False

    with shared_data["stepper_degrees"].get_lock():
        azimuth = shared_data["stepper_degrees"].value
    with shared_data["servo_degrees"].get_lock():
        elevation = shared_data["servo_degrees"].value

    detect_satellite_direct_index(distance_cm, strength, azimuth, elevation, shared_data)
    return True


# --- NEW HELPER FUNCTION ---
def update_best_strength_point(az, el, strength, shared_data):
    """During the centering scan, keeps track of the reading with the highest strength."""
    with shared_data["best_strength_point"].get_lock():
        if strength > shared_data["best_strength_point"][2]:
            shared_data["best_strength_point"][0] = az
            shared_data["best_strength_point"][1] = el
            shared_data["best_strength_point"][2] = strength


def detect_satellite_direct_index(current_range, current_strength, azimuth, elevation, shared_data):
    """Compares current reading to background map to find anomalies."""
    index = get_background_index(azimuth, elevation)
    if index is None:
        return False

    with shared_data["background_lidar"].get_lock():
        background_strength = shared_data["background_lidar"][index]
        background_range = shared_data["background_lidar"][index + 1]

    is_anomaly = (background_range == 0) or \
                 (abs(current_strength - background_strength) > 5000) or \
                 (abs(current_range - background_range) > 50)

    if is_anomaly:
        with shared_data["satellite_points"].get_lock():
            shared_data["satellite_points"][0] = azimuth
            shared_data["satellite_points"][1] = elevation
            shared_data["satellite_points"][2] = current_strength
            shared_data["satellite_points"][3] = current_range

        with shared_data["satellite_detected"].get_lock():
            if not shared_data["satellite_detected"].value:
                print(
                    f"SATELLITE DETECTED at Az: {azimuth:.1f}, El: {elevation:.1f}, Rng: {current_range}cm, Str: {current_strength}")
            shared_data["satellite_detected"].value = True
    else:
        with shared_data["satellite_detected"].get_lock():
            shared_data["satellite_detected"].value = False

    return is_anomaly