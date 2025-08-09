import serial
import numpy as np
import time
from multiprocessing import Value
from collections import deque

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
                    print(f"[TFmini] Background index built: {len(bg_index)} cells")
                    # ✅ prevent rebuild loop
                    with shared_data["background_ready"].get_lock():
                        shared_data["background_ready"].value = False

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
    """
    Enhanced version of your existing save_background function.
    Keeps same signature but adds timestamp for data freshness.
    """
    az = int(round(stepper)) % 360
    el = int(round(servo))
    if not (0 <= el < 90):
        return background_array  # skip out-of-scan elevations

    # Grid index: one unique cell per (el, az)
    pos = el * 360 + az

    # Keep your column order but add timestamp: [pos, distance, strength, timestamp]
    timestamp = time.time()
    new_row = np.array([[pos, lidar_data[0], lidar_data[1], timestamp]])

    # NOTE: np.append reallocates; acceptable for quick patch
    background_array = np.append(background_array, new_row, axis=0)
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
    """
    Enhanced version of your existing validate_lidar_data function.
    Keeps same signature but adds temporal consistency checking.
    """
    # Your existing basic checks
    if distance_cm in (-1, -2, -4) or strength < 100:
        return False
    if distance_cm < 100 or distance_cm > 300:
        return False
    if strength <2100:  # Keeping your existing threshold
        return False

    # Additional stability check (optional - can be disabled)
    # This adds temporal filtering without changing the interface
    current_time = time.time()

    # Store reading history in shared_data if not exists
    if not hasattr(validate_lidar_data, 'history'):
        validate_lidar_data.history = deque(maxlen=5)

    validate_lidar_data.history.append({
        'timestamp': current_time,
        'distance': distance_cm,
        'strength': strength
    })

    # Remove old entries
    while (validate_lidar_data.history and
           current_time - validate_lidar_data.history[0]['timestamp'] > 1.0):
        validate_lidar_data.history.popleft()

    # Check for reasonable stability if we have enough samples
    if len(validate_lidar_data.history) >= 3:
        distances = [h['distance'] for h in validate_lidar_data.history]
        strengths = [h['strength'] for h in validate_lidar_data.history]

        # If readings are too erratic, might be noise
        dist_std = np.std(distances)
        strength_std = np.std(strengths)

        if dist_std > 200 or strength_std > 8000:  # Adjusted for your system
            return False

    return True


def detect_satellite_direct_index(current_strength, current_range_cm, az_deg, el_deg, shared_data, bg_index):
    """
    Enhanced version of your existing detect_satellite_direct_index function.
    Keeps the same signature but adds multi-criteria detection.
    """
    az = int(round(az_deg)) % 360
    el = int(round(el_deg))
    b = bg_index.get((az, el))
    if not b:
        return False

    bg_strength, bg_range_cm = b

    # Multi-criteria detection scoring
    detection_score = 0.0

    # Criterion 1: Strength difference
    strength_diff = abs(current_strength - bg_strength)
    strength_ratio = current_strength / max(bg_strength, 1000)

    if strength_diff > 200:
        detection_score += min(strength_diff / 1000.0, 5.0)

    if strength_ratio > 1.1:
        detection_score += min(strength_ratio, 3.0)

    # Criterion 2: Range difference
    range_diff = abs(current_range_cm - bg_range_cm)

    if range_diff > 100:
        detection_score += min(range_diff / 100.0, 3.0)

    # Criterion 3: Expected drone characteristics
    if 100 <= current_range_cm <= 300:
        detection_score += 2.0
        if 500 <= current_range_cm <= 800:
            detection_score += 1.0

    # Criterion 4: Strength threshold
    if current_strength < 5000:
        detection_score *= 0.5

    # Final decision
    detection_threshold = 4.0
    is_detected = detection_score >= detection_threshold

    if is_detected:
        print(f"[DETECT] Detection at ({az_deg:.1f}°, {el_deg:.1f}°) Score: {detection_score:.2f}")
        sp = shared_data["satellite_points"]
        sp[0], sp[1], sp[2], sp[3] = az_deg, el_deg, current_strength, current_range_cm
        shared_data["satellite_detected"].value = True
        return True
    else:
        shared_data["satellite_detected"].value = False
        return False
def decode_pos(pos_int):
    pos = int(pos_int)
    az = pos % 360
    el = pos // 360
    return az, el


def build_bg_index(path):
    """
    Enhanced version of your existing build_bg_index function.
    Keeps same signature but adds quality filtering.
    """
    import numpy as np
    idx = {}

    try:
        bg = np.load(path)
    except Exception as e:
        print(f"[TFmini] Could not load background file '{path}': {e}")
        return idx

    # Accept both 1D (flattened) or 2D (rows) just in case
    if bg.ndim == 1 and bg.size % 4 == 0:
        bg = bg.reshape((-1, 4))

    # Group measurements by position for quality filtering
    position_groups = {}

    for row in bg:
        if len(row) < 3:
            continue

        pos = int(row[0])
        dist_cm = float(row[1])
        strength = float(row[2])

        # --- decode using the existing scheme ---
        az = pos % 360
        el = pos // 360

        # basic sanity filters (keeping your existing ones)
        if not (0 <= az < 360 and 0 <= el < 90):
            continue
        if not (10.0 <= dist_cm <= 2000.0):
            continue
        if not np.isfinite(dist_cm) or not np.isfinite(strength):
            continue

        # Group by position for quality filtering
        pos_key = (az, el)
        if pos_key not in position_groups:
            position_groups[pos_key] = []
        position_groups[pos_key].append({'distance': dist_cm, 'strength': strength})

    # Process each position group with quality filtering
    for pos_key, measurements in position_groups.items():
        if not measurements:
            continue

        if len(measurements) > 1:
            # Use median values for more robust background
            distances = [m['distance'] for m in measurements]
            strengths = [m['strength'] for m in measurements]

            # Remove obvious outliers
            dist_median = np.median(distances)
            strength_median = np.median(strengths)

            filtered_measurements = []
            for m in measurements:
                if (abs(m['distance'] - dist_median) < 100 and
                        abs(m['strength'] - strength_median) < 2000):
                    filtered_measurements.append(m)

            if filtered_measurements:
                avg_distance = np.mean([m['distance'] for m in filtered_measurements])
                avg_strength = np.mean([m['strength'] for m in filtered_measurements])
                idx[pos_key] = (avg_strength, avg_distance)
            else:
                idx[pos_key] = (strength_median, dist_median)
        else:
            m = measurements[0]
            idx[pos_key] = (m['strength'], m['distance'])

    print(f"[TFmini] Background index built: {len(idx)} cells")
    return idx
