# LiDAR/lidar_handler.py

import serial
import numpy as np
import time

def read_tfmini_data(serial_port):
    buffer = bytearray()
    
    while True:
        data = serial_port.read(serial_port.in_waiting or 1)
        buffer += data

        while len(buffer) >= 9:
            if buffer[0] == 0x59 and buffer[1] == 0x59:
                distance = buffer[2] + (buffer[3] << 8)
                strength = buffer[4] + (buffer[5] << 8)
                buffer = buffer[9:]
                return distance, strength
            else:
                buffer = buffer[1:]

# --- MODIFIED: Main LiDAR Process ---
def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    """
    TFmini process that reads sensor data, validates it, and triggers
    satellite detection logic.
    """
    background_array = np.array([])
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial opened, reading data...")
            while not shared_data["shutdown"].value:
                distance, strength = read_tfmini_data(ser)

                if distance is not None and strength is not None:
                    # Update the generic lidar_data for GUI or other uses
                    with shared_data["lidar_data"].get_lock():
                        shared_data["lidar_data"][0] = distance
                        shared_data["lidar_data"][1] = strength
                        shared_data["lidar_data"][2] = time.time()
                    
                    # --- NEW: Validate every reading for potential satellite detection ---
                    validate_lidar_data(distance, strength, shared_data)

                # Background scanning logic remains the same
                if shared_data["scan_trigger"].value:
                    stepper = shared_data["stepper_degrees"].value 
                    servo = shared_data["servo_degrees"].value
                    background_array = save_background(background_array, shared_data["lidar_data"], stepper, servo)

                if shared_data["save_background"].value:
                    np.save("background_data.npy", background_array)
                    print("Background data saved to background_data.npy")
                    shared_data["save_background"].value = False
                
                time.sleep(0.01)

    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")

# Unchanged functions
def save_background(background_array, lidar_data, stepper, servo):
    pos = int(str(round(stepper)) + str(round(servo)))
    new_row = np.array([[pos, lidar_data[0], lidar_data[1], lidar_data[2]]])
    background_array = np.append(background_array, new_row, axis=0)
    return background_array

def validate_lidar_data(distance_cm, strength, shared_data):
    """
    Validates LiDAR data. If valid, it proceeds to check if it's a satellite.
    """
    if distance_cm in [-1, -2, -4] or strength < 100 or strength == 65535:
        return False
    
    if not (300 <= distance_cm <= 1000): # 3-10 meters
        return False
    
    if strength < 50000:
        return False
    
    # If all checks pass, it might be a satellite.
    with shared_data["stepper_degrees"].get_lock():
        azimuth = shared_data["stepper_degrees"].value
    with shared_data["servo_degrees"].get_lock():
        elevation = shared_data["servo_degrees"].value

    detect_satellite_direct_index(distance_cm, strength, azimuth, elevation, shared_data)
    return True
  
def detect_satellite_direct_index(current_range, current_strength, azimuth, elevation, shared_data):
    """
    Compares the current reading to the background scan to detect anomalies (satellites).
    If an anomaly is found, it sets the satellite_detected flag for the EKF.
    """
    background_lidar = shared_data["background_lidar"]
    az_idx = int(azimuth)
    el_idx = int(elevation)
    
    background_strength, background_range = background_lidar[az_idx][el_idx]
    
    strength_diff = abs(current_strength - background_strength)
    range_diff = abs(current_range - background_range)
    
    # Check tolerances
    if strength_diff <= 5000 and range_diff <= 50:
        # This is likely a background object, do nothing
        return False
    else:
        # This is a potential satellite, update shared data for the EKF
        with shared_data["satellite_points"].get_lock():
            shared_data["satellite_points"][0] = azimuth
            shared_data["satellite_points"][1] = elevation
            shared_data["satellite_points"][2] = current_strength
            shared_data["satellite_points"][3] = current_range
        
        # --- THIS IS THE TRIGGER FOR THE KALMAN FILTER ---
        with shared_data["satellite_detected"].get_lock():
            shared_data["satellite_detected"].value = True
        
        print(f"SATELLITE DETECTED at Az: {azimuth}, El: {elevation}")
        return True
