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
    Publishes [distance_cm, strength, timestamp] into shared_data["lidar_data"].
    Also:
      - Builds/refreshes a background index from background_data.npy
      - Optionally collects 3 valid points during acquisition
      - Calls validation + detection while acquiring/running EKF
      - Saves background scan when requested
    """
    # Bind shared arrays/flags once to avoid shadowing
    lidar_sh = shared_data["lidar_data"]  # multiprocessing.Array('d', 3)
    stepper_deg = shared_data["stepper_degrees"]   # Value('d', ...)
    servo_deg   = shared_data["servo_degrees"]     # Value('d', ...)

    # Optional acceptance range in meters (Array('d', [min_m, max_m]))
    lidar_range_sh = shared_data.get("lidar_acceptance_range", None)

    # Background accumulation in RAM until we save to disk
    background_array = np.empty((0, 4))
    bg_index = {}
    bg_loaded_ts = 0.0

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini] Serial opened, reading data...")
            while not shared_data["shutdown"].value:
                # ---- Read one TFmini frame ----
                distance_cm, strength = read_tfmini_data(ser)

                # ---- Publish to shared memory (if valid frame) ----
                if distance_cm is not None and strength is not None:
                    ts = time.time()
                    with lidar_sh.get_lock():
                        lidar_sh[0] = float(distance_cm)
                        lidar_sh[1] = float(strength)
                        lidar_sh[2] = ts

                # Small pacing to avoid pegging the CPU
                time.sleep(0.01)

                # ---- Refresh background index if a new file is ready ----
                if shared_data.get("background_ready", Value('b', False)).value and (time.time() - bg_loaded_ts > 1.0):
                    bg_index = build_bg_index(shared_data["background_path"])
                    bg_loaded_ts = time.time()
                    # Optional: print how many cells were indexed
                    print(f"[TFmini] Background index built: {len(bg_index)} cells")

                # ---- Read the latest LiDAR sample + current mount angles ----
                with lidar_sh.get_lock():
                    distance_cm = float(lidar_sh[0])
                    strength    = float(lidar_sh[1])
                    ts          = float(lidar_sh[2])

                az = float(stepper_deg.value)  # degrees
                el = float(servo_deg.value)    # degrees

                # ---- Validate & detect only when acquiring or EKF is running ----
                if shared_data.get("acquire_points", Value('b', False)).value or \
                   shared_data.get("ekf_running", Value('b', False)).value:

                    # Optional acceptance range check (meters)
                    if lidar_range_sh is not None:
                        min_m = float(lidar_range_sh[0])
                        max_m = float(lidar_range_sh[1])
                        in_window = (min_m <= distance_cm / 100.0 <= max_m)
                    else:
                        # Default window: 3–12 m
                        in_window = (300.0 <= distance_cm <= 1200.0)

                    if in_window and validate_lidar_data(distance_cm, strength, shared_data):
                        # Note: detect_satellite_direct_index signature expects (strength, distance_cm, az, el, shared_data, bg_index)
                        detect_satellite_direct_index(strength, distance_cm, az, el, shared_data, bg_index)

                # ---- Background scan accumulation (when concentric scan is running) ----
                if shared_data.get("scan_trigger", Value('b', False)).value:
                    # Use current lidar_sh snapshot + mount angles
                    background_array = save_background(background_array, lidar_sh, az, el)

                # ---- 3-point acquisition for EKF init ----
                if shared_data.get("acquire_points", Value('b', False)).value and shared_data.get("satellite_detected", Value('b', False)).value:
                    # Pull the last detected "satellite" point and append to points_buffer
                    az_idx = shared_data["satellite_points"][0]
                    el_idx = shared_data["satellite_points"][1]
                    str_pt = shared_data["satellite_points"][2]
                    dist_m = shared_data["satellite_points"][3] / 100.0  # cm → m

                    with shared_data["points_count"].get_lock():
                        k = shared_data["points_count"].value
                        if k < 3:
                            base = 4 * k
                            pb = shared_data["points_buffer"]  # 12 doubles
                            pb[base + 0] = float(az_idx)
                            pb[base + 1] = float(el_idx)
                            pb[base + 2] = float(dist_m)
                            pb[base + 3] = float(str_pt)
                            shared_data["points_count"].value = k + 1

                    shared_data["satellite_detected"].value = False

                # ---- Persist background file when requested ----
                if shared_data.get("save_background", Value('b', False)).value:
                    np.save(shared_data["background_path"], background_array)
                    shared_data["background_ready"].value = True
                    print(f"[TFmini] Background data saved to {shared_data['background_path']}, rows={len(background_array)}")
                    shared_data["save_background"].value = False

    except serial.SerialException as e:
        print(f"[TFmini] Serial error: {e}")



def save_background(background_array, lidar_data, stepper, servo):
    pos = int(str(round(stepper)) + str(round(servo)))
    new_row = np.array([[pos, lidar_data[0], lidar_data[1], lidar_data[2]]])
    background_array = np.append(background_array, new_row, axis=0)
    print([pos, lidar_data[0], lidar_data[1], lidar_data[2]])
    return background_array

def pos_to_index(shared_data):
    scale = 1.5 #change later it should be equal to concentric search step size for both servo and stepper
    step_deg = shared_data["stepper_degrees"]
    servo_deg = shared_data["servo_degrees"]
    return int(step_deg/scale+servo_deg/scale*360/scale)

def append_lidar_data(np_array, shared_data):
    distance, strength, timestamp = shared_data["lidar_data"]
    np_array[index] = [strength, distance, timestamp]
    
#pass lidar_data.distance and lidar_data.strength from the shared memory
def validate_lidar_data(distance_cm, strength, shared_data):
    if distance_cm in (-1, -2, -4) or strength < 100:
        return False
    if distance_cm < 300 or distance_cm > 1200:   # 3–12 m for your EKF
        return False
    if strength < 5000:   # your 50,000 was too high for TFmini; start modest
        return False
    return True
  

def detect_satellite_direct_index(current_strength, current_range_cm, az_deg, el_deg, shared_data, bg_index):
    az = int(round(az_deg)) % 360
    el = int(round(el_deg))
    b = bg_index.get((az, el))
    if not b:
        return False  # no background ref here yet

    bg_strength, bg_range_cm = b
    strength_diff = abs(current_strength - bg_strength)
    range_diff = abs(current_range_cm - bg_range_cm)

    if strength_diff <= 5000 and range_diff <= 50:
        shared_data["satellite_detected"].value = False
        return False
    else:
        # write the *latest* point (we’ll collect 3 separately)
        sp = shared_data["satellite_points"]
        sp[0], sp[1], sp[2], sp[3] = az, el, current_strength, current_range_cm
        shared_data["satellite_detected"].value = True
        return True

def decode_pos(pos_int):
    s = str(int(pos_int)).zfill(4)
    az = int(s[:-2]) % 360
    el = int(s[-2:])  # 0..99 (you scan 0..90)
    return az, el

def build_bg_index(path):
    """Return a dict[(az,el)] = (strength, distance_cm)."""
    try:
        bg = np.load(path)
    except Exception:
        return {}
    idx = {}
    for row in bg:
        az, el = decode_pos(row[0])
        dist_cm = float(row[1])
        strength = float(row[2])
        if 10 < dist_cm < 2000:   # cheap sanity filter
            idx[(az, el)] = (strength, dist_cm)
    return idx