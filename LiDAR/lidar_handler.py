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

    background_array = np.array([])  # Placeholder for background data


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

                if shared_data["scan_trigger"].value:
                    stepper = shared_data["stepper_degrees"].value 
                    servo = shared_data["servo_degrees"].value
                    save_background(background_array, lidar_data, stepper, servo)

                if shared_data["save_background"].value:
                    np.save("background_data.npy", background_array)
                    #shared_data["background_data"] = background_array
                    shared_data["save_background"].value = False
    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")



def save_background(background_array, lidar_data, stepper, servo):
    pos = int(str(round(stepper)) + str(round(servo)))
    np.append(background_array, [pos, lidar_data[0], lidar_data[1], lidar_data[2]])
    return

def pos_to_index(shared_data):
    scale = 1.5 #change later it should be equal to concentric search step size for both servo and stepper
    step_deg = shared_data["stepper_degrees"]
    servo_deg = shared_data["servo_degrees"]
    return int(step_deg/scale+servo_deg/scale*360/scale)

def append_lidar_data(np_array, shared_data):
    distance, strength, timestamp = shared_data["lidar_data"]
    np_array[index] = [strength, distance, timestamp]

