# LiDAR/lidar_handler.py

import serial
import numpy as np
import time

def read_tfmini_data(serial_port):
    """Reads a single data frame from the TFmini LiDAR."""
    buffer = bytearray()
    while True:
        data = serial_port.read(serial_port.in_waiting or 1)
        if data:
            buffer += data
            while len(buffer) >= 9:
                if buffer[0] == 0x59 and buffer[1] == 0x59:
                    distance = buffer[2] + (buffer[3] << 8)
                    strength = buffer[4] + (buffer[5] << 8)
                    buffer = buffer[9:]
                    return distance, strength
                else:
                    buffer.pop(0)

def get_background_index(azimuth, elevation):
    """Calculates the base index in the 1D shared array for a given az/el."""
    az_idx = int(round(azimuth)) % 360
    el_idx = int(round(elevation))
    
    if not (0 <= el_idx < 90):
        return None
    return (el_idx * 360 + az_idx) * 2

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
                    with shared_data["lidar_data"].get_lock():
                        shared_data["lidar_data"][0] = distance
                        shared_data["lidar_data"][1] = strength
                        shared_data["lidar_data"][2] = time.time()
                    
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

                    if shared_data["save_background"].value:
                        print("[LiDAR] Saving background data to file...")
                        with shared_data["background_lidar"].get_lock():
                            background_np = np.array(shared_data["background_lidar"]).reshape((90, 360, 2))
                        np.save("background_data.npy", background_np)
                        print("[LiDAR] Background data saved to 'background_data.npy'.")
                        with shared_data["save_background"].get_lock():
                            shared_data["save_background"].value = False

                    validate_lidar_data(distance, strength, shared_data)
                
                time.sleep(0.005)

    except serial.SerialException as e:
        print(f"[LiDAR] Serial error: {e}")
    print("[LiDAR] Shutting down.")


def validate_lidar_data(distance_cm, strength, shared_data):
    """Validates data and checks if it's a satellite."""
    # --- MODIFIED FOR TESTING ---
    # Condition 1: Distance must be between 1.5m and 10m (150cm - 1000cm)
    # Condition 2: Strength must be greater than 1000 (much easier to achieve)
    if not (150 <= distance_cm <= 1000 and strength > 1000):
        return False
    
    # If the reading is valid, proceed to anomaly detection
    with shared_data["stepper_degrees"].get_lock():
        azimuth = shared_data["stepper_degrees"].value
    with shared_data["servo_degrees"].get_lock():
        elevation = shared_data["servo_degrees"].value

    detect_satellite_direct_index(distance_cm, strength, azimuth, elevation, shared_data)
    return True
  
def detect_satellite_direct_index(current_range, current_strength, azimuth, elevation, shared_data):
    """Compares current reading to background map to find anomalies."""
    index = get_background_index(azimuth, elevation)
    if index is None:
        return False

    with shared_data["background_lidar"].get_lock():
        background_strength = shared_data["background_lidar"][index]
        background_range = shared_data["background_lidar"][index + 1]

    if background_range == 0:
        # This spot was not scanned or was out of range during the scan,
        # so any valid reading is a potential satellite.
        is_anomaly = True
    else:
        # This spot was scanned, so we check if the new object is different enough.
        strength_diff = abs(current_strength - background_strength)
        range_diff = abs(current_range - background_range)
        is_anomaly = (strength_diff > 500 or range_diff > 20)

    if is_anomaly:
        with shared_data["satellite_points"].get_lock():
            shared_data["satellite_points"][0] = azimuth
            shared_data["satellite_points"][1] = elevation
            shared_data["satellite_points"][2] = current_strength
            shared_data["satellite_points"][3] = current_range
        
        with shared_data["satellite_detected"].get_lock():
            # Only print if the flag was previously false to avoid spamming the console
            if not shared_data["satellite_detected"].value:
                 print(f"SATELLITE DETECTED at Az: {azimuth:.1f}, El: {elevation:.1f}, Rng: {current_range}cm, Str: {current_strength}")
            shared_data["satellite_detected"].value = True
        return True
    else:
         # If it's not an anomaly, ensure the flag is false
        with shared_data["satellite_detected"].get_lock():
            shared_data["satellite_detected"].value = False
        return False
