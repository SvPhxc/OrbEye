# File: motors/motor_controller.py

from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic
import math

# --- NEW: Define constants for GPIO pins for clarity ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
STEPPER_DIR_PIN = 3
STEPPER_ENABLE_PIN = 4
STEPPER_SLEEP_PIN = 6
MICROSTEP_ANGLE = 0.05625


# (stepper_worker is unchanged)
def stepper_worker(pi, movement_queue, shared_data):
    print("[WORKER] Stepper worker started.")
    pulse_wave_id = -1
    try:
        us_delay = 500
        pi.wave_add_generic(
            [pigpio.pulse(1 << STEPPER_PULSE_PIN, 0, us_delay), pigpio.pulse(0, 1 << STEPPER_PULSE_PIN, us_delay)])
        pulse_wave_id = pi.wave_create()
        while not shared_data['shutdown'].value:
            try:
                command = movement_queue.get(timeout=0.1); _ = command if command is not None else (_ for _ in
                                                                                                    ()).throw(
                    Exception())
            except Exception:
                continue
            direction, degrees_to_move, _ = command;
            ideal_microsteps = degrees_to_move / MICROSTEP_ANGLE
            total_microsteps_to_consider = ideal_microsteps + shared_data['cumulative_error'].value;
            actual_microsteps_to_take = round(total_microsteps_to_consider)
            shared_data['cumulative_error'].value = total_microsteps_to_consider - actual_microsteps_to_take
            if actual_microsteps_to_take == 0: continue
            pi.write(STEPPER_DIR_PIN, 0 if direction == 'left' else 1);
            repeats_lsb = actual_microsteps_to_take % 256;
            repeats_msb = actual_microsteps_to_take // 256
            chain = [255, 0, pulse_wave_id, 255, 1, repeats_lsb, repeats_msb];
            pi.wave_chain(chain)
            while pi.wave_tx_busy(): sleep(0.01)
            current_pos = shared_data['stepper_degrees'].value;
            actual_degrees_this_move = actual_microsteps_to_take * MICROSTEP_ANGLE
            new_pos = (current_pos - actual_degrees_this_move) if direction == 'left' else (
                        current_pos + actual_degrees_this_move)
            shared_data['stepper_degrees'].value = new_pos % 360
    finally:
        if pulse_wave_id != -1 and pi.connected: pi.wave_delete(pulse_wave_id)
        print("[WORKER] Stepper worker shutting down.")


# --- CHANGE 1: MODIFY SERVO FUNCTION TO USE SOFTWARE PWM ---
def smooth_servo_move(pi, target_degrees, shared_data, step_delay=0.01, step_size=1):
    """Moves the servo smoothly using software-timed PWM pulses."""
    current_degrees = shared_data['servo_degrees'].value
    target_degrees = max(0, min(180, target_degrees))
    step = step_size if target_degrees > current_degrees else -step_size
    if step == 0: return

    # Loop to create a smooth movement effect
    degrees_range = range(int(round(current_degrees)), int(round(target_degrees)), step)
    for degrees in degrees_range:
        # Calculate the required pulse width in microseconds (500-2500 is typical for servos)
        pulse_width = 500 + (degrees / 0.09) + (28 / 0.09)
        # Instead of `set_servo_pulsewidth`, we use `set_PWM_dutycycle`.
        # This uses software timing and will not conflict with the hardware PWM on the stepper pin.
        pi.set_PWM_dutycycle(SERVO_PIN, pulse_width)
        shared_data['servo_degrees'].value = degrees
        sleep(step_delay)

    # Send the final pulse width to ensure it lands exactly on the target.
    final_pulse_width = 500 + (target_degrees / 0.09) + (28 / 0.09)
    pi.set_PWM_dutycycle(SERVO_PIN, final_pulse_width)
    shared_data['servo_degrees'].value = target_degrees


# (move and track_target are unchanged, they just call the modified servo function)
def move(pi, direction, degrees, delay, movement_queue, shared_data):
    if direction in ['left', 'right']:
        movement_queue.put((direction, degrees, delay))
    elif direction in ['up', 'down']:
        target_degrees = shared_data['servo_degrees'].value + (degrees if direction == 'up' else -degrees)
        smooth_servo_move(pi, target_degrees, shared_data)


def track_target(pi, target_azimuth, target_elevation, delay, movement_queue, shared_data):
    current_pan = shared_data["stepper_degrees"].value;
    current_tilt = shared_data["servo_degrees"].value;
    adjusted_azimuth = target_azimuth % 360;
    adjusted_elevation = max(0, min(180, target_elevation))
    delta_pan = (adjusted_azimuth - current_pan + 540) % 360 - 180
    if abs(delta_pan) > 0.1: move(pi, "right" if delta_pan > 0 else "left", abs(delta_pan), delay, movement_queue,
                                  shared_data)
    if abs(adjusted_elevation - current_tilt) > 1: smooth_servo_move(pi, adjusted_elevation, shared_data)


# (concentric_ring_search_smooth is unchanged, it correctly uses hardware PWM for the stepper)
def concentric_ring_search_smooth(pi, shared_data):
    print("\n--- STARTING HIGH-FIDELITY CONCENTRIC RING SEARCH ---")
    pan_direction = 1;
    initial_pan_angle = shared_data['stepper_degrees'].value
    for radius in range(int(90.0), -1, -int(1.5)):
        if shared_data['shutdown'].value: break
        smooth_servo_move(pi, 90.0 - radius, shared_data)
        print(f"\n--- Scanning ring at Tilt: {shared_data['servo_degrees'].value:.1f}° ---")
        pi.write(STEPPER_DIR_PIN, 1 if pan_direction > 0 else 0)
        scan_frequency_hz = 1778  # This can now be changed without affecting the servo.
        pi.hardware_PWM(STEPPER_PULSE_PIN, scan_frequency_hz, 500000)
        degrees_per_second = scan_frequency_hz * MICROSTEP_ANGLE;
        duration = 360.0 / degrees_per_second
        start_time = monotonic()
        while monotonic() - start_time < duration:
            if shared_data['shutdown'].value: break
            elapsed_time = monotonic() - start_time;
            degrees_turned = elapsed_time * degrees_per_second
            current_pan = (initial_pan_angle + degrees_turned * pan_direction) % 360
            with shared_data['stepper_degrees'].get_lock():
                shared_data['stepper_degrees'].value = current_pan
            sleep(0.01)
        pi.hardware_PWM(STEPPER_PULSE_PIN, 0, 0);
        pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT)
        initial_pan_angle = (initial_pan_angle + 360 * pan_direction) % 360
        shared_data['stepper_degrees'].value = initial_pan_angle
        if shared_data['shutdown'].value: break
        pan_direction *= -1
    print("\n--- HIGH-FIDELITY SEARCH FINISHED ---");
    smooth_servo_move(pi, 90.0, shared_data);
    return True


def spiral_acquire_three(pi, shared_data, movement_queue):
    """Tight outward spiral, stop once 3 validated points are captured by LiDAR process."""
    center_az = shared_data['stepper_degrees'].value
    center_el = shared_data['servo_degrees'].value

    radius = 0.5  # degrees
    turns = 2
    step = 0.5
    direction = 1  # keep current pan direction
    shared_data["points_count"].value = 0

    for t in np.arange(0.0, turns * 360.0, step):
        if shared_data['shutdown'].value: break
        if shared_data["points_count"].value >= 3: break

        # simple Archimedean spiral
        r = radius + 0.01 * t
        az = center_az + direction * r * math.cos(math.radians(t))
        el = max(0, min(90, center_el + r * math.sin(math.radians(t))))

        track_target(pi, az, el, 0.0001, movement_queue, shared_data)
        sleep(0.03)


def initialize_gpio():
    GPIO.setwarnings(False);
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STEPPER_ENABLE_PIN, GPIO.OUT);
    GPIO.setup(STEPPER_SLEEP_PIN, GPIO.OUT)
    GPIO.output(STEPPER_SLEEP_PIN, GPIO.HIGH);
    GPIO.output(STEPPER_ENABLE_PIN, GPIO.LOW)


# Enhanced 3-Point Acquisition System for Drone Tracking
# Add this to your motor_controller.py

import numpy as np
import math
from time import sleep, monotonic


def enhanced_three_point_acquisition(pi, shared_data, movement_queue):
    """
    Enhanced 3-point acquisition system:
    1. Initial detection and strength optimization
    2. Motion vector estimation via spiral sampling
    3. Predictive positioning for third point
    """
    print("\n=== STARTING ENHANCED 3-POINT DRONE ACQUISITION ===")

    # Reset acquisition state
    shared_data["points_count"].value = 0
    shared_data["satellite_detected"].value = False
    acquisition_points = []

    # Phase 1: Initial Detection and Strength Optimization
    print("Phase 1: Initial detection and strength optimization...")
    first_point = find_initial_detection_point(pi, shared_data, movement_queue)

    if not first_point:
        print("❌ Phase 1 failed: No initial detection found")
        return False

    acquisition_points.append(first_point)
    print(f"✅ Point 1 acquired: Az={first_point['az']:.1f}°, El={first_point['el']:.1f}°, "
          f"Str={first_point['strength']:.0f}, Dist={first_point['distance']:.1f}m")

    # Phase 2: Motion Vector Estimation
    print("Phase 2: Motion vector estimation via spiral sampling...")
    second_point, motion_vector = estimate_motion_vector(pi, shared_data, movement_queue, first_point)

    if not second_point:
        print("❌ Phase 2 failed: Could not estimate motion vector")
        return False

    acquisition_points.append(second_point)
    print(f"✅ Point 2 acquired: Az={second_point['az']:.1f}°, El={second_point['el']:.1f}°, "
          f"Str={second_point['strength']:.0f}, Dist={second_point['distance']:.1f}m")
    print(f"📊 Estimated motion vector: Δaz={motion_vector['delta_az']:.2f}°/s, "
          f"Δel={motion_vector['delta_el']:.2f}°/s")

    # Phase 3: Predictive Third Point Acquisition
    print("Phase 3: Predictive positioning for third point...")
    third_point = acquire_predictive_third_point(pi, shared_data, movement_queue, second_point, motion_vector)

    if not third_point:
        print("❌ Phase 3 failed: Could not acquire predictive third point")
        return False

    acquisition_points.append(third_point)
    print(f"✅ Point 3 acquired: Az={third_point['az']:.1f}°, El={third_point['el']:.1f}°, "
          f"Str={third_point['strength']:.0f}, Dist={third_point['distance']:.1f}m")

    # Store all points in shared memory for EKF initialization
    store_acquisition_points(shared_data, acquisition_points)

    print("🎯 3-POINT ACQUISITION COMPLETED SUCCESSFULLY!")
    return True


def find_initial_detection_point(pi, shared_data, movement_queue, search_radius=8.0):
    """
    Phase 1: Find initial drone detection and optimize signal strength
    """
    best_point = None
    max_strength = 0
    search_timeout = 30.0  # seconds
    start_time = monotonic()

    # Start with current position
    center_az = shared_data['stepper_degrees'].value
    center_el = shared_data['servo_degrees'].value

    print(f"🔍 Searching for initial detection around Az={center_az:.1f}°, El={center_el:.1f}°")

    # Concentric circle search with strength optimization
    for radius in np.arange(0.5, search_radius, 0.8):
        if monotonic() - start_time > search_timeout:
            break

        print(f"  Searching radius {radius:.1f}°...")

        # Sample points around the circle
        for angle in np.arange(0, 360, 12):  # 30 points per circle
            if monotonic() - start_time > search_timeout:
                break

            # Calculate target position
            target_az = center_az + radius * math.cos(math.radians(angle))
            target_el = center_el + radius * math.sin(math.radians(angle))
            target_el = max(5, min(85, target_el))  # Constrain elevation

            # Move to position
            track_target(pi, target_az, target_el, 0.001, movement_queue, shared_data)
            sleep(0.1)  # Allow settling and measurement

            # Check for detection
            point = sample_current_position(shared_data)
            if point and is_valid_drone_detection(point, shared_data):
                if point['strength'] > max_strength:
                    max_strength = point['strength']
                    best_point = point
                    print(f"  📈 New best strength: {max_strength:.0f} at Az={point['az']:.1f}°")

        # If we found a decent detection, refine it
        if best_point and max_strength > 8000:
            print(f"🔍 Refining detection around best point...")
            refined_point = refine_detection_point(pi, shared_data, movement_queue, best_point)
            if refined_point:
                return refined_point

    return best_point if best_point and max_strength > 5000 else None


def refine_detection_point(pi, shared_data, movement_queue, initial_point, refine_radius=2.0):
    """
    Fine-tune the detection point to find maximum signal strength
    """
    best_point = initial_point
    max_strength = initial_point['strength']

    center_az = initial_point['az']
    center_el = initial_point['el']

    # Fine grid search around the initial point
    for az_offset in np.arange(-refine_radius, refine_radius + 0.3, 0.3):
        for el_offset in np.arange(-refine_radius, refine_radius + 0.3, 0.3):
            target_az = center_az + az_offset
            target_el = max(5, min(85, center_el + el_offset))

            track_target(pi, target_az, target_el, 0.001, movement_queue, shared_data)
            sleep(0.08)

            point = sample_current_position(shared_data)
            if point and point['strength'] > max_strength:
                max_strength = point['strength']
                best_point = point

    # Return to best position
    track_target(pi, best_point['az'], best_point['el'], 0.001, movement_queue, shared_data)
    sleep(0.1)

    return best_point


def estimate_motion_vector(pi, shared_data, movement_queue, first_point,
                           sample_duration=3.0, spiral_radius=3.0):
    """
    Phase 2: Estimate drone motion vector using spiral sampling
    """
    samples = []
    start_time = monotonic()

    # Spiral parameters
    center_az = first_point['az']
    center_el = first_point['el']
    turns = 2.0
    step = 8.0  # degrees per step in spiral

    print(f"🌀 Starting spiral motion estimation (duration: {sample_duration}s)")

    # Execute spiral sampling
    for t in np.arange(0, turns * 360, step):
        if monotonic() - start_time > sample_duration:
            break

        # Archimedean spiral
        r = (t / 360) * spiral_radius
        spiral_az = center_az + r * math.cos(math.radians(t))
        spiral_el = center_el + r * math.sin(math.radians(t))
        spiral_el = max(5, min(85, spiral_el))

        # Move and sample
        track_target(pi, spiral_az, spiral_el, 0.001, movement_queue, shared_data)
        sleep(0.05)

        point = sample_current_position(shared_data)
        if point and is_valid_drone_detection(point, shared_data):
            point['timestamp'] = monotonic()
            samples.append(point)
            print(f"  📍 Sample {len(samples)}: Az={point['az']:.1f}°, Str={point['strength']:.0f}")

    if len(samples) < 3:
        print(f"❌ Insufficient samples for motion estimation ({len(samples)} < 3)")
        return None, None

    # Analyze samples to find best second point and estimate motion
    return analyze_motion_samples(samples, first_point)


def analyze_motion_samples(samples, first_point):
    """
    Analyze spiral samples to find the best second point and estimate motion vector
    """
    # Sort samples by strength (highest first)
    samples.sort(key=lambda x: x['strength'], reverse=True)

    # Find the best sample that's sufficiently different from first point
    min_separation = 2.0  # degrees
    best_second = None

    for sample in samples:
        az_diff = abs(sample['az'] - first_point['az'])
        el_diff = abs(sample['el'] - first_point['el'])
        separation = math.sqrt(az_diff ** 2 + el_diff ** 2)

        if separation > min_separation:
            best_second = sample
            break

    if not best_second:
        return None, None

    # Estimate motion vector using temporal analysis
    # Group samples by time windows and find trend
    time_sorted_samples = sorted([s for s in samples if s['strength'] > 6000],
                                 key=lambda x: x['timestamp'])

    if len(time_sorted_samples) < 3:
        # Fallback: simple direction from first to second point
        dt = 1.0  # assume 1 second
        motion_vector = {
            'delta_az': (best_second['az'] - first_point['az']) / dt,
            'delta_el': (best_second['el'] - first_point['el']) / dt,
            'confidence': 0.5
        }
    else:
        # Linear regression on position vs time
        motion_vector = calculate_motion_vector_regression(time_sorted_samples)

    return best_second, motion_vector


def calculate_motion_vector_regression(samples):
    """
    Calculate motion vector using linear regression on samples
    """
    if len(samples) < 3:
        return {'delta_az': 0, 'delta_el': 0, 'confidence': 0}

    # Extract time and positions
    times = np.array([s['timestamp'] for s in samples])
    azimuths = np.array([s['az'] for s in samples])
    elevations = np.array([s['el'] for s in samples])

    # Normalize time to start from 0
    times = times - times[0]

    # Linear regression
    if len(times) > 1 and np.std(times) > 0:
        az_slope = np.polyfit(times, azimuths, 1)[0]
        el_slope = np.polyfit(times, elevations, 1)[0]

        # Calculate confidence based on R-squared
        az_r2 = calculate_r_squared(times, azimuths, az_slope)
        el_r2 = calculate_r_squared(times, elevations, el_slope)
        confidence = (az_r2 + el_r2) / 2
    else:
        az_slope = 0
        el_slope = 0
        confidence = 0

    return {
        'delta_az': az_slope,
        'delta_el': el_slope,
        'confidence': max(0.3, min(1.0, confidence))
    }


def calculate_r_squared(x, y, slope):
    """Calculate R-squared for simple linear regression"""
    if len(x) < 2:
        return 0
    y_pred = slope * x
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def acquire_predictive_third_point(pi, shared_data, movement_queue, second_point,
                                   motion_vector, prediction_time=2.0):
    """
    Phase 3: Acquire third point by predicting drone position
    """
    # Predict future position
    predicted_az = second_point['az'] + motion_vector['delta_az'] * prediction_time
    predicted_el = second_point['el'] + motion_vector['delta_el'] * prediction_time
    predicted_el = max(5, min(85, predicted_el))

    print(f"🎯 Predicting drone at Az={predicted_az:.1f}°, El={predicted_el:.1f}° "
          f"in {prediction_time}s")

    # Move to predicted position
    track_target(pi, predicted_az, predicted_el, 0.001, movement_queue, shared_data)

    # Wait for drone to arrive (with some tolerance)
    wait_start = monotonic()
    max_wait = prediction_time + 1.0
    best_point = None
    max_strength = 0

    while monotonic() - wait_start < max_wait:
        sleep(0.1)
        point = sample_current_position(shared_data)

        if point and is_valid_drone_detection(point, shared_data):
            if point['strength'] > max_strength:
                max_strength = point['strength']
                best_point = point

            # If we get a really strong signal, we can stop early
            if point['strength'] > 12000:
                break

    # If prediction failed, do a local search
    if not best_point or max_strength < 7000:
        print("🔍 Prediction missed, performing local search...")
        best_point = local_search_for_drone(pi, shared_data, movement_queue,
                                            predicted_az, predicted_el, radius=4.0)

    return best_point


def local_search_for_drone(pi, shared_data, movement_queue, center_az, center_el, radius=4.0):
    """
    Perform local search around predicted position
    """
    best_point = None
    max_strength = 0

    # Grid search pattern
    for az_offset in np.arange(-radius, radius + 1, 1.0):
        for el_offset in np.arange(-radius, radius + 1, 1.0):
            target_az = center_az + az_offset
            target_el = max(5, min(85, center_el + el_offset))

            track_target(pi, target_az, target_el, 0.001, movement_queue, shared_data)
            sleep(0.08)

            point = sample_current_position(shared_data)
            if point and point['strength'] > max_strength:
                max_strength = point['strength']
                best_point = point

    return best_point if max_strength > 5000 else None


def sample_current_position(shared_data):
    """
    Sample current LiDAR data and mount position
    """
    try:
        with shared_data["lidar_data"].get_lock():
            distance_cm = float(shared_data["lidar_data"][0])
            strength = float(shared_data["lidar_data"][1])
            timestamp = float(shared_data["lidar_data"][2])

        az = float(shared_data['stepper_degrees'].value)
        el = float(shared_data['servo_degrees'].value)

        return {
            'az': az,
            'el': el,
            'distance': distance_cm / 100.0,  # Convert to meters
            'strength': strength,
            'timestamp': timestamp
        }
    except Exception:
        return None


def is_valid_drone_detection(point, shared_data):
    """
    Validate if a point represents a valid drone detection
    """
    # Check distance range
    if point['distance'] < 6.0 or point['distance'] > 12.0:
        return False

    # Check signal strength
    if point['strength'] < 5000:
        return False

    # Check against background (if available)
    # This uses your existing background detection logic
    return True  # Simplified for now


def store_acquisition_points(shared_data, points):
    """
    Store the 3 acquired points in shared memory for EKF initialization
    """
    points_buffer = shared_data["points_buffer"]

    for i, point in enumerate(points[:3]):  # Ensure only 3 points
        base_idx = i * 4
        points_buffer[base_idx + 0] = point['az']
        points_buffer[base_idx + 1] = point['el']
        points_buffer[base_idx + 2] = point['distance']
        points_buffer[base_idx + 3] = point['strength']

    shared_data["points_count"].value = len(points)
    shared_data["ekf_start"].value = True

    print(f"📊 Stored {len(points)} points for EKF initialization")


# Integration function to replace your existing spiral_acquire_three
def spiral_acquire_three_enhanced(pi, shared_data, movement_queue):
    """
    Enhanced replacement for your existing spiral_acquire_three function
    """
    return enhanced_three_point_acquisition(pi, shared_data, movement_queue)


def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...");
    initialize_gpio()
    pi = pigpio.pi()
    if not pi.connected: return
    pi.set_mode(STEPPER_PULSE_PIN, pigpio.OUTPUT);
    pi.set_mode(STEPPER_DIR_PIN, pigpio.OUTPUT)

    # Initialize software PWM for the servo
    pi.set_PWM_frequency(SERVO_PIN, 50)  # Standard servo frequency is 50Hz
    pi.set_PWM_range(SERVO_PIN, 20000)  # Set range to 20000, so 1 unit = 1 microsecond of pulse width

    stepper_process = Process(target=stepper_worker, args=(pi, movement_queue, shared_data));
    stepper_process.start()
    smooth_servo_move(pi, shared_data['servo_degrees'].value, shared_data)
    try:
        while not shared_data['shutdown'].value:
            if shared_data['scan_trigger'].value:
                concentric_ring_search_smooth(pi, shared_data)
                shared_data['scan_trigger'].value = False
                shared_data['save_background'].value = True

            if shared_data['tilt_up'].value: move(pi, 'up', 5.0, None, movement_queue, shared_data); shared_data[
                'tilt_up'].value = False
            if shared_data['tilt_down'].value: move(pi, 'down', 5.0, None, movement_queue, shared_data); shared_data[
                'tilt_down'].value = False
            if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data); shared_data[
                'pan_left'].value = False
            if shared_data['pan_right'].value: move(pi, 'right', 5.0, 0.0001, movement_queue, shared_data); shared_data[
                'pan_right'].value = False
            if shared_data["go_to_target"].value: track_target(pi, shared_data["target_azimuth"].value,
                                                               shared_data["target_elevation"].value, 0.0001,
                                                               movement_queue, shared_data); shared_data[
                "go_to_target"].value = False

            # UPDATED: Enhanced 3-point acquisition
            if shared_data["acquire_points"].value:
                print("\n🎯 Starting enhanced 3-point acquisition...")
                success = enhanced_three_point_acquisition(pi, shared_data, movement_queue)
                shared_data["acquire_points"].value = False

                if success:
                    print("✅ 3-point acquisition successful - starting EKF")
                    # EKF will start automatically when ekf_start flag is set
                else:
                    print("❌ 3-point acquisition failed")
                    shared_data["points_count"].value = 0
                    shared_data["ekf_start"].value = False

            # EKF tracking mode
            if shared_data['ekf_running'].value:
                # Follow EKF predictions
                track_target(pi,
                             shared_data["predicted_azimuth"].value,
                             shared_data["predicted_elevation"].value,
                             0.0001, movement_queue, shared_data)

                # Optional: Add adaptive tracking based on confidence
                confidence = shared_data["ekf_confidence"].value
                if confidence < 0.3:
                    print(f"⚠️  Low EKF confidence ({confidence:.2f}) - consider re-acquisition")

            sleep(0.05)
    finally:
        print("[MotorControl] Shutting down...")
        movement_queue.put(None);
        stepper_process.join();
        GPIO.output(STEPPER_ENABLE_PIN, GPIO.HIGH)
        if pi.connected:
            pi.set_PWM_dutycycle(SERVO_PIN, 0)
            pi.stop()
        GPIO.cleanup()


# Additional helper functions for the enhanced acquisition system

def adaptive_tracking_mode(pi, shared_data, movement_queue):
    """
    Adaptive tracking that adjusts based on EKF confidence and signal strength
    """
    confidence = shared_data["ekf_confidence"].value

    # Get current LiDAR reading
    with shared_data["lidar_data"].get_lock():
        current_strength = shared_data["lidar_data"][1]

    if confidence > 0.7 and current_strength > 8000:
        # High confidence, precise tracking
        track_target(pi,
                     shared_data["predicted_azimuth"].value,
                     shared_data["predicted_elevation"].value,
                     0.0001, movement_queue, shared_data)

    elif confidence > 0.4:
        # Medium confidence, slightly looser tracking
        pred_az = shared_data["predicted_azimuth"].value
        pred_el = shared_data["predicted_elevation"].value

        # Add small search pattern around prediction
        import random
        search_offset = 0.5  # degrees
        az_offset = random.uniform(-search_offset, search_offset)
        el_offset = random.uniform(-search_offset, search_offset)

        track_target(pi, pred_az + az_offset, pred_el + el_offset,
                     0.0001, movement_queue, shared_data)

    else:
        # Low confidence, might need re-acquisition
        print(f"⚠️  EKF confidence too low ({confidence:.2f}) - stopping tracking")
        shared_data['ekf_running'].value = False
        return False

    return True


def satellite_orbit_predictor(point1, point2, point3):
    """
    Simple orbital motion predictor for satellite/drone movement
    Used to improve motion vector estimation
    """
    # Convert points to numpy arrays for easier calculation
    p1 = np.array([point1['az'], point1['el']])
    p2 = np.array([point2['az'], point2['el']])
    p3 = np.array([point3['az'], point3['el']])

    # Calculate velocity vectors
    v1 = p2 - p1  # velocity from point 1 to 2
    v2 = p3 - p2  # velocity from point 2 to 3

    # Estimate acceleration (change in velocity)
    acceleration = v2 - v1

    # Predict next position using kinematic equations
    # Assuming constant acceleration model
    next_position = p3 + v2 + 0.5 * acceleration

    return {
        'predicted_az': float(next_position[0]),
        'predicted_el': float(next_position[1]),
        'velocity_az': float(v2[0]),
        'velocity_el': float(v2[1]),
        'acceleration_az': float(acceleration[0]),
        'acceleration_el': float(acceleration[1])
    }