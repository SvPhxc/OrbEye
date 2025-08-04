import serial
import time

def read_tfmini_data(serial_port):
    """
    Reads one data frame (9 bytes) from TFmini and extracts range and strength.
    """
    if serial_port.in_waiting >= 9:
        first_byte = serial_port.read()
        if first_byte != b'\x59':
            return None, None
        second_byte = serial_port.read()
        if second_byte != b'\x59':
            return None, None

        raw_data = serial_port.read(7)
        distance = raw_data[0] + raw_data[1] * 256
        strength = raw_data[2] + raw_data[3] * 256
        return distance, strength
    return None, None

def tfmini_process(shared_data, port="/dev/serial0", baudrate=115200):
    """
    TFmini worker process for Raspberry Pi GPIO UART.
    Updates shared_data with Range and Strength.
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial opened, reading data...")
            while True:
                distance, strength = read_tfmini_data(ser)
                if distance is not None and strength is not None:
                    shared_data["Range"] = distance
                    shared_data["Strength"] = strength
    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")