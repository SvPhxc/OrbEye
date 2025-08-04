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

