import serial
import numpy as np
import time

def read_tfmini_data(serial_port):
    """
    Reads one data frame (9 bytes) from TFmini and extracts range and strength.
    """
    if serial_port.in_waiting >= 9:
        if serial_port.read() != b'\x59':
            return None, None
        if serial_port.read() != b'\x59':
            return None, None

        raw_data = serial_port.read(7)
        distance = raw_data[0] + raw_data[1] * 256
        strength = raw_data[2] + raw_data[3] * 256
        return distance, strength
    return None, None

def run_lidar(shared_data, port="/dev/serial0", baudrate=115200):
    """
    TFmini process for Raspberry Pi UART.
    Publishes np.array([distance, strength, timestamp]) to shared_data["lidar_array"]
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial opened, reading data...")
            while not shared_data.get("shutdown", False):
                distance, strength = read_tfmini_data(ser)
                if distance is not None and strength is not None:
                    timestamp = time.time()
                    shared_data["lidar_array"] = np.array([distance, strength, timestamp])
                time.sleep(0.01)
    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")
