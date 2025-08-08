import serial
import numpy as np
import time


# --- NEW: Added the missing read_tfmini_data function ---
def read_tfmini_data(serial_port):
    """Reads a single data frame from the TFmini LiDAR."""
    buffer = bytearray()
    while True:
        # Read a chunk of data if available
        data = serial_port.read(serial_port.in_waiting or 1)
        if data:
            buffer += data
            # Look for the 9-byte frame header
            while len(buffer) >= 9:
                if buffer[0] == 0x59 and buffer[1] == 0x59:
                    distance = buffer[2] + (buffer[3] << 8)
                    strength = buffer[4] + (buffer[5] << 8)
                    # Clear the buffer up to the end of the frame
                    buffer = buffer[9:]
                    return distance, strength
                else:
                    # If header not found, discard the first byte and search again
                    buffer.pop(0)
        # If no data, return None to avoid blocking
        else:
            return None, None


def get_background_index(azimuth, elevation):
    az_idx = int(round(azimuth)) % 360
    el_idx = int(round(elevation))
    if not (0 <= el_idx < 90): return None
    return (el_idx * 360 + az_idx) * 2


def update_best_strength_point(az, el, strength, shared_data):
    with shared_data["best_strength_point"].get_lock():
        if strength > shared_data["best_strength_point"][2]:
            shared_data["best_strength_point"][0] = az
            shared_data["best_strength_point"][1] = el
            shared_data["best_strength_point"][2] = strength


def detect_satellite_direct_index(current_range, current_strength, azimuth, elevation, shared_data):
    index = get_background_index(azimuth, elevation)
    if index is None: return False
    with shared_data["background_lidar"].get_lock():
        background_strength = shared_data["background_lidar"][index]
        background_range = shared_data["background_lidar"][index + 1]
    is_anomaly = (background_range == 0) or \
                 (abs(current_strength - background_strength) > 5000) or \
                 (abs(current_range - background_range) > 50)
    with shared_data["satellite_detected"].get_lock():
        shared_data["satellite_detected"].value = is_anomaly
    if is_anomaly:
        with shared_data["satellite_points"].get_lock():
            shared_data["satellite_points"][0:4] = [azimuth, elevation, current_strength, current_range]
        # print(f"SATELLITE DETECTED at Az: {azimuth:.1f}, El: {elevation:.1f}") # Optional: for debugging
    return is_anomaly


def validate_lidar_data(distance_cm, strength, shared_data):
    if not (150 <= distance_cm <= 300 and strength > 3000):
        with shared_data["satellite_detected"].get_lock():
            shared_data["satellite_detected"].value = False
        return False
    with shared_data["stepper_degrees"].get_lock():
        azimuth = shared_data["stepper_degrees"].value
    with shared_data["servo_degrees"].get_lock():
        elevation = shared_data["servo_degrees"].value
    detect_satellite_direct_index(distance_cm, strength, azimuth, elevation, shared_data)
    return True


def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    try:
        with serial.Serial(port, baudrate, timeout=0.1) as ser:
            print("[LiDAR] Serial opened, reading data...")
            while not shared_data["shutdown"].value:
                distance, strength = read_tfmini_data(ser)  # This call is now valid
                if distance is not None:
                    current_time = time.time()
                    with shared_data["lidar_data"].get_lock():
                        shared_data["lidar_data"][0:3] = [distance, strength, current_time]

                    # Update background map if scanning
                    if shared_data["scan_trigger"].value:
                        with shared_data["stepper_degrees"].get_lock():
                            az = shared_data["stepper_degrees"].value
                        with shared_data["servo_degrees"].get_lock():
                            el = shared_data["servo_degrees"].value
                        index = get_background_index(az, el)
                        if index is not None:
                            with shared_data["background_lidar"].get_lock():
                                shared_data["background_lidar"][index:index + 2] = [strength, distance]

                    # If in centering state, update the best point found
                    if shared_data["acquisition_state"].value == STATE_CENTERING_P1:
                        with shared_data["stepper_degrees"].get_lock(): az = shared_data["stepper_degrees"].value
                        with shared_data["servo_degrees"].get_lock(): el = shared_data["servo_degrees"].value
                        update_best_strength_point(az, el, strength, shared_data)

                    validate_lidar_data(distance, strength, shared_data)

                time.sleep(0.005)
    except serial.SerialException as e:
        print(f"[LiDAR] Serial error: {e}")
    print("[LiDAR] Shutting down.")