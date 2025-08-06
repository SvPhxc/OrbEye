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
                # Extract and parse a full frame
                distance = buffer[2] + (buffer[3] << 8)
                strength = buffer[4] + (buffer[5] << 8)
                #print(f"Distance: {distance} cm, Strength: {strength}")
                buffer = buffer[9:]  # Remove this frame from the buffer
                return distance, strength
            else:
                buffer = buffer[1:]  # Skip until next potential frame

def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    """
    TFmini process for Raspberry Pi UART.
    Publishes [distance, strength, timestamp] to shared_data["lidar_data"]
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial opened, reading data...")
            while not shared_data["shutdown"].value:
                distance, strength = read_tfmini_data(ser)
                if distance is not None and strength is not None:
                    lidar_data = shared_data["lidar_data"]
                    lidar_data[0] = distance
                    lidar_data[1] = strength
                    lidar_data[2] = time.time()
                    #print("Wrote to shared_data:", list(lidar_data))
                time.sleep(0.01)
    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")

def pos_to_index(shared_data):
    scale = 1.5 #change later it should be equal to concentric search step size for both servo and stepper
    step_deg = shared_data["stepper_degrees"]
    servo_deg = shared_data["servo_degrees"]
    return int(step_deg/scale+servo_deg/scale*360/scale)

def append_lidar_data(np_array, shared_data):
    distance, strength, timestamp = shared_data["lidar_data"]
    np_array[index] = [strength, distance, timestamp]
    
#pass lidar_data.distance and lidar_data.strength from the shared memory
def validate_lidar_data(distance_cm, strength,shared_data):
    """
    Validates LiDAR data based on distance and signal strength.
    
    Args:
        distance_cm (int/float): Distance in centimeters (or special error codes)
        strength (int): Signal strength value between 0-65535
    
    Returns:
        bool: True if data is valid, False otherwise
        Call detect_satellite_direct_index if valid
    """
    
    # Check for special error conditions first
    if distance_cm == -1 or strength < 100:
        print("Reading is unreliable - strength is < 100")

        return False
    
    elif distance_cm == -2 or strength == 65535:
        print("Signal strength saturation")
        return False
    
    elif distance_cm == -4:
        print("Ambient light saturation")
        return False
    #have this in an array
    # Check for valid distance range (3-10 meters = 300-1000 cm)
    if distance_cm < 300 or distance_cm > 1000:
        print(f"Distance {distance_cm}cm is outside valid range (300-1000cm / 3-10m)")
        return False
    
    # Check for minimum strength requirement
    if strength < 50000:
        print(f"Signal strength {strength} is below minimum threshold (50,000)")
        return False
    # If all checks pass
    print(f"Valid reading: {distance_cm}cm, strength: {strength}")
    azimuth = shared_data["stepper_degrees"]
    elevation = shared_data["servo_degrees"]
    detect_satellite_direct_index(distance_cm, strength, azimuth, elevation, shared_data)
    return True
  
#To be called when the reading is valid
def detect_satellite_direct_index(current_strength, current_range, azimuth, elevation,shared_data):
    background_lidar = shared_data["background_lidar"]
    # Convert angles to array indices
    az_idx = int(azimuth)
    el_idx = int(elevation)
    
    # Direct lookup - O(1)!
    background_strength, background_range = background_lidar[az_idx][el_idx]
    
    # Simple comparison
    strength_diff = abs(current_strength - background_strength)
    range_diff = abs(current_range - background_range)
    
    # Check tolerances
    if strength_diff <= 5000 and range_diff <= 50:
        shared_data["satellite_detected"].value = False  # Background object
        return False  # Background object
    else:
        shared_data["satellite_points"][0] = az_idx
        shared_data["satellite_points"][1] = el_idx 
        shared_data["satellite_points"][2] = current_strength
        shared_data["satellite_points"][3] = current_range
        #saved the new points in this array later to be passed to kalman filter
        shared_data["satellite_detected"].value = True
        
        print(f"Satellite detected at azimuth: {azimuth}, elevation: {elevation}, strength: {current_strength}, range: {current_range}")
        return True   # Potential satellite

