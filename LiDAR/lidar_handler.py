# Enhanced LiDAR handler integration
# Add these functions to your lidar_handler.py

def enhanced_validate_lidar_data(distance_cm, strength, shared_data):
    """
    Enhanced validation for acquisition system with configurable parameters
    """
    # Basic validity checks
    if distance_cm in (-1, -2, -4) or strength < 100:
        return False

    # Use configurable distance range from GUI
    min_range_m = shared_data.get("lidar_acceptance_range", [0.5, 12.0])[0]
    max_range_m = shared_data.get("lidar_acceptance_range", [0.5, 12.0])[1]

    min_range_cm = min_range_m * 100
    max_range_cm = max_range_m * 100

    if distance_cm < min_range_cm or distance_cm > max_range_cm:
        return False

    # Dynamic strength threshold based on distance
    # Closer targets can have lower strength due to beam divergence
    distance_m = distance_cm / 100.0
    if distance_m < 8.0:
        min_strength = 4000  # Lower threshold for close targets
    elif distance_m < 10.0:
        min_strength = 5000  # Medium threshold
    else:
        min_strength = 6000  # Higher threshold for distant targets

    if strength < min_strength:
        return False

    return True


def enhanced_detect_satellite_with_context(current_strength, current_range_cm, az_deg, el_deg,
                                           shared_data, bg_index, acquisition_mode=False):
    """
    Enhanced detection with acquisition mode support and better filtering
    """
    az = int(round(az_deg)) % 360
    el = int(round(el_deg))

    # Get background reference
    bg_ref = bg_index.get((az, el))

    if acquisition_mode:
        # In acquisition mode, be more sensitive but still filter noise
        if not bg_ref:
            # No background - use absolute thresholds
            if current_strength > 7000 and 600 <= current_range_cm <= 1200:
                return True
        else:
            # Compare with background
            bg_strength, bg_range_cm = bg_ref
            strength_diff = abs(current_strength - bg_strength)
            range_diff = abs(current_range_cm - bg_range_cm)

            # More sensitive thresholds during acquisition
            if strength_diff > 3000 or range_diff > 30:
                return True
    else:
        # Normal tracking mode - use original thresholds
        if not bg_ref:
            return False

        bg_strength, bg_range_cm = bg_ref
        strength_diff = abs(current_strength - bg_strength)
        range_diff = abs(current_range_cm - bg_range_cm)

        if strength_diff <= 5000 and range_diff <= 50:
            return False
        else:
            return True

    return False


def run_lidar_enhanced(shared_data, port="/dev/serial0", baudrate=115200):
    """
    Enhanced LiDAR process with acquisition mode support
    """
    # Bind shared arrays/flags
    lidar_sh = shared_data["lidar_data"]
    stepper_deg = shared_data["stepper_degrees"]
    servo_deg = shared_data["servo_degrees"]

    # Background data
    background_array = np.empty((0, 4))
    bg_index = {}
    bg_loaded_ts = 0.0

    # Acquisition state tracking
    last_detection_time = 0.0
    detection_history = []  # Store recent detections for filtering

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("[TFmini Enhanced] Serial opened, reading data...")

            while not shared_data["shutdown"].value:
                # Read TFmini frame
                distance_cm, strength = read_tfmini_data(ser)

                if distance_cm is not None and strength is not None:
                    ts = time.time()
                    with lidar_sh.get_lock():
                        lidar_sh[0] = float(distance_cm)
                        lidar_sh[1] = float(strength)
                        lidar_sh[2] = ts

                time.sleep(0.01)

                # Refresh background index
                if shared_data.get("background_ready", Value('b', False)).value and (time.time() - bg_loaded_ts > 1.0):
                    bg_index = build_bg_index(shared_data["background_path"])
                    bg_loaded_ts = time.time()
                    print(f"[TFmini Enhanced] Background index updated: {len(bg_index)} cells")

                # Get current readings
                with lidar_sh.get_lock():
                    distance_cm = float(lidar_sh[0])
                    strength = float(lidar_sh[1])
                    ts = float(lidar_sh[2])

                az = float(stepper_deg.value)
                el = float(servo_deg.value)

                # Determine if we're in acquisition mode
                acquiring = shared_data.get("acquire_points", Value('b', False)).value
                ekf_running = shared_data.get("ekf_running", Value('b', False)).value

                # Enhanced detection logic
                if acquiring or ekf_running:
                    # Use enhanced validation
                    if enhanced_validate_lidar_data(distance_cm, strength, shared_data):
                        # Use context-aware detection
                        is_detection = enhanced_detect_satellite_with_context(
                            strength, distance_cm, az, el, shared_data, bg_index,
                            acquisition_mode=acquiring)

                        if is_detection:
                            # Add to detection history for filtering
                            detection = {
                                'az': az,
                                'el': el,
                                'distance_cm': distance_cm,
                                'strength': strength,
                                'timestamp': ts
                            }

                            detection_history.append(detection)

                            # Keep only recent detections (last 2 seconds)
                            detection_history = [d for d in detection_history
                                                 if ts - d['timestamp'] < 2.0]

                            # Filter detections - require consistency
                            if len(detection_history) >= 2 or acquiring:
                                # Store the detection
                                sp = shared_data["satellite_points"]
                                sp[0], sp[1], sp[2], sp[3] = az, el, strength, distance_cm
                                shared_data["satellite_detected"].value = True

                                if acquiring:
                                    print(f"[TFmini Enhanced] Detection during acquisition: "
                                          f"Az={az:.1f}°, El={el:.1f}°, Str={strength:.0f}")

                # Background scan accumulation
                if shared_data.get("scan_trigger", Value('b', False)).value:
                    background_array = save_background(background_array, lidar_sh, az, el)

                # Handle 3-point acquisition feedback
                if acquiring and shared_data.get("satellite_detected", Value('b', False)).value:
                    # The acquisition system will handle this detection
                    pass

                # Save background when requested
                if shared_data.get("save_background", Value('b', False)).value:
                    np.save(shared_data["background_path"], background_array)
                    shared_data["background_ready"].value = True
                    print(f"[TFmini Enhanced] Background saved: {len(background_array)} points")
                    shared_data["save_background"].value = False

    except serial.SerialException as e:
        print(f"[TFmini Enhanced] Serial error: {e}")


def adaptive_strength_threshold(distance_m, base_strength=5000):
    """
    Calculate adaptive strength threshold based on distance and environmental factors
    """
    # TFmini strength typically decreases with distance
    # But can vary based on target reflectivity and atmospheric conditions

    if distance_m < 7:
        # Very close - lower threshold due to potential oversaturation
        return base_strength * 0.8
    elif distance_m < 10:
        # Medium range - standard threshold
        return base_strength
    else:
        # Far range - higher threshold needed for reliability
        return base_strength * 1.3


def detection_confidence_filter(detections, min_consistency=0.7):
    """
    Filter detections based on spatial and temporal consistency
    """
    if len(detections) < 2:
        return True  # Allow single detections during initial acquisition

    # Calculate position consistency
    positions = np.array([[d['az'], d['el']] for d in detections])
    pos_std = np.std(positions, axis=0)

    # Calculate strength consistency
    strengths = [d['strength'] for d in detections]
    strength_cv = np.std(strengths) / np.mean(strengths) if np.mean(strengths) > 0 else 1.0

    # Combined consistency score
    pos_consistency = 1.0 / (1.0 + np.mean(pos_std))
    strength_consistency = 1.0 / (1.0 + strength_cv)
    overall_consistency = (pos_consistency + strength_consistency) / 2.0

    return overall_consistency >= min_consistency


# Performance monitoring functions
class AcquisitionPerformanceMonitor:
    """Monitor and log acquisition performance metrics"""

    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        self.detection_counts = {'phase1': 0, 'phase2': 0, 'phase3': 0}
        self.strength_history = {'phase1': [], 'phase2': [], 'phase3': []}

    def start_acquisition(self):
        self.start_time = time.time()
        self.phase_times = {}
        self.detection_counts = {'phase1': 0, 'phase2': 0, 'phase3': 0}
        self.strength_history = {'phase1': [], 'phase2': [], 'phase3': []}

    def log_phase_start(self, phase_name):
        self.phase_times[f"{phase_name}_start"] = time.time()

    def log_phase_end(self, phase_name):
        self.phase_times[f"{phase_name}_end"] = time.time()

    def log_detection(self, phase, strength):
        self.detection_counts[phase] += 1
        self.strength_history[phase].append(strength)

    def get_performance_summary(self):
        total_time = time.time() - self.start_time if self.start_time else 0

        summary = {
            'total_time': total_time,
            'phase_durations': {},
            'detection_stats': {},
            'strength_stats': {}
        }

        # Calculate phase durations
        for phase in ['phase1', 'phase2', 'phase3']:
            start_key = f"{phase}_start"
            end_key = f"{phase}_end"
            if start_key in self.phase_times and end_key in self.phase_times:
                summary['phase_durations'][phase] = (
                        self.phase_times[end_key] - self.phase_times[start_key]
                )

        # Detection statistics
        for phase in ['phase1', 'phase2', 'phase3']:
            summary['detection_stats'][phase] = self.detection_counts[phase]
            if self.strength_history[phase]:
                summary['strength_stats'][phase] = {
                    'mean': np.mean(self.strength_history[phase]),
                    'max': np.max(self.strength_history[phase]),
                    'min': np.min(self.strength_history[phase])
                }

        return summary