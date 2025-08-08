# motor_controller.py

import numpy as np
from multiprocessing import Process, Queue
import pigpio
import RPi.GPIO as GPIO
from time import sleep, monotonic, time
import math

# --- Constants (Unchanged) ---
SERVO_PIN = 13
STEPPER_PULSE_PIN = 19
# ... other constants

# --- State Definitions ---
STATE_SEARCHING = 0
STATE_CENTERING_P1 = 1
STATE_SPIRAL_P2 = 2
STATE_PREDICT_P3 = 3
STATE_COMPLETE = 4
STATE_TRACKING = 5


# --- Low-level motor functions (stepper_worker, smooth_servo_move, etc. are unchanged) ---
# ... (paste your existing stepper_worker, smooth_servo_move, move, track_target, and concentric_ring_search_smooth here) ...


# --- NEW: State Machine Logic Functions ---

def center_for_point_1(pi, shared_data, movement_queue):
    """
    Performs a small scan around the initial detection point to find the peak signal strength.
    This becomes the first confirmed point.
    """
    print("[Acquisition] STATE 1: Centering for Point 1...")

    # Reset best strength tracker
    with shared_data["best_strength_point"].get_lock():
        shared_data["best_strength_point"][0] = 0.0
        shared_data["best_strength_point"][1] = 0.0
        shared_data["best_strength_point"][2] = 0.0

    # Get initial detection position
    with shared_data["satellite_points"].get_lock():
        center_az = shared_data["satellite_points"][0]

    # Scan a 10-degree arc (5 left, 5 right)
    scan_width = 10.0
    start_az = center_az - (scan_width / 2)

    print(f"[Acquisition] Scanning from {start_az:.1f}° to {start_az + scan_width:.1f}°")
    track_target(pi, start_az, shared_data["servo_degrees"].value, 0.0001, movement_queue, shared_data)
    sleep(0.5)  # Wait to arrive

    # Sweep right
    move(pi, 'right', scan_width, 0.0001, movement_queue, shared_data)
    sleep(1.0)  # Time for the scan to complete

    with shared_data["best_strength_point"].get_lock():
        best_az = shared_data["best_strength_point"][0]
        best_el = shared_data["best_strength_point"][1]
        best_str = shared_data["best_strength_point"][2]

    if best_str > 0:
        print(f"[Acquisition] Found peak strength {best_str} at Az: {best_az:.1f}, El: {best_el:.1f}")
        # Go to the best point
        track_target(pi, best_az, best_el, 0.0001, movement_queue, shared_data)
        sleep(0.2)

        # Lock in Point 1
        with shared_data["initial_points"].get_lock():
            with shared_data["lidar_data"].get_lock():
                shared_data["initial_points"][0] = best_az
                shared_data["initial_points"][1] = best_el
                shared_data["initial_points"][2] = shared_data["lidar_data"][0]  # distance
                shared_data["initial_points"][3] = shared_data["lidar_data"][2]  # timestamp

        # Transition to next state
        with shared_data["acquisition_state"].get_lock():
            shared_data["acquisition_state"].value = STATE_SPIRAL_P2
    else:
        print("[Acquisition] Failed to find peak strength. Resetting.")
        with shared_data["acquisition_state"].get_lock():
            shared_data["acquisition_state"].value = STATE_SEARCHING


def spiral_for_point_2(pi, shared_data, movement_queue):
    """
    Performs a tight outward spiral to find the second point.
    """
    print("[Acquisition] STATE 2: Spiraling for Point 2...")
    with shared_data["initial_points"].get_lock():
        center_az = shared_data["initial_points"][0]
        center_el = shared_data["initial_points"][1]

    # Reset detection flag to catch the new point
    shared_data["satellite_detected"].value = False

    radius = 1.0  # degrees
    max_radius = 15.0
    step = 5.0  # degrees step

    t = 0.0
    start_time = monotonic()
    while monotonic() - start_time < 5.0:  # 5-second timeout for spiral
        if shared_data['shutdown'].value: break

        # Check if LiDAR process found the point
        if shared_data["satellite_detected"].value:
            print("[Acquisition] Point 2 acquired during spiral.")
            with shared_data["initial_points"].get_lock():
                with shared_data["satellite_points"].get_lock():
                    shared_data["initial_points"][4] = shared_data["satellite_points"][0]
                    shared_data["initial_points"][5] = shared_data["satellite_points"][1]
                    shared_data["initial_points"][6] = shared_data["satellite_points"][3]  # distance
                    shared_data["initial_points"][7] = time()

            with shared_data["acquisition_state"].get_lock():
                shared_data["acquisition_state"].value = STATE_PREDICT_P3
            return

        # Archimedean spiral calculation
        r = radius + 0.05 * t
        if r > max_radius:
            print("[Acquisition] Spiral radius exceeded. Resetting.");
            break

        az = center_az + r * math.cos(math.radians(t))
        el = max(0, min(90, center_el + r * math.sin(math.radians(t))))

        track_target(pi, az, el, 0.0001, movement_queue, shared_data)
        sleep(0.01)
        t += step

    # If loop finishes without detection
    with shared_data["acquisition_state"].get_lock():
        if shared_data["acquisition_state"].value == STATE_SPIRAL_P2:
            print("[Acquisition] Failed to find Point 2. Resetting.")
            shared_data["acquisition_state"].value = STATE_SEARCHING


def predict_point_3(pi, shared_data, movement_queue):
    """
    Calculates velocity from P1 and P2, predicts P3's location, and moves there to wait.
    """
    print("[Acquisition] STATE 3: Predicting Point 3...")
    with shared_data["initial_points"].get_lock():
        p1_az, p1_el, _, p1_time = shared_data["initial_points"][0:4]
        p2_az, p2_el, _, p2_time = shared_data["initial_points"][4:8]

    dt = p2_time - p1_time
    if dt < 0.01:  # Avoid division by zero
        print("[Acquisition] Time delta too small. Resetting.")
        with shared_data["acquisition_state"].get_lock():
            shared_data["acquisition_state"].value = STATE_SEARCHING
        return

    # Calculate angular velocity (degrees per second)
    az_vel = (p2_az - p1_az) / dt
    el_vel = (p2_el - p1_el) / dt

    # Predict the next position after another dt
    pred_az = p2_az + az_vel * dt
    pred_el = p2_el + el_vel * dt

    print(f"[Acquisition] Predicted P3 at Az: {pred_az:.1f}, El: {pred_el:.1f}. Moving to target.")
    track_target(pi, pred_az, pred_el, 0.0001, movement_queue, shared_data)
    shared_data["satellite_detected"].value = False

    # Wait for detection at the predicted spot
    start_time = monotonic()
    while monotonic() - start_time < 3.0:  # 3-second timeout
        if shared_data["satellite_detected"].value:
            print("[Acquisition] Point 3 acquired at predicted location!")
            with shared_data["initial_points"].get_lock():
                with shared_data["satellite_points"].get_lock():
                    shared_data["initial_points"][8] = shared_data["satellite_points"][0]
                    shared_data["initial_points"][9] = shared_data["satellite_points"][1]
                    shared_data["initial_points"][10] = shared_data["satellite_points"][3]  # distance
                    shared_data["initial_points"][11] = time()

            with shared_data["acquisition_state"].get_lock():
                shared_data["acquisition_state"].value = STATE_COMPLETE
            return
        sleep(0.05)

    print("[Acquisition] Failed to find Point 3. Resetting.")
    with shared_data["acquisition_state"].get_lock():
        shared_data["acquisition_state"].value = STATE_SEARCHING


# --- MAIN MOTOR CONTROL PROCESS ---

def run_motor_control(shared_data, movement_queue):
    print("[MotorControl] Starting...");
    initialize_gpio()
    pi = pigpio.pi()
    # ... (rest of hardware initialization is the same)

    try:
        while not shared_data['shutdown'].value:
            # --- Get current state ---
            state = shared_data['acquisition_state'].value

            # --- Manual Override Controls (Unchanged) ---
            if shared_data['pan_left'].value: move(pi, 'left', 5.0, 0.0001, movement_queue, shared_data); shared_data[
                'pan_left'].value = False; continue
            # ... (other manual controls)

            # --- STATE MACHINE ---
            if shared_data['acquire_points'].value:
                if state == STATE_SEARCHING and shared_data["satellite_detected"].value:
                    # Initial detection has occurred, start the centering process
                    shared_data['acquisition_state'].value = STATE_CENTERING_P1

                elif state == STATE_CENTERING_P1:
                    center_for_point_1(pi, shared_data, movement_queue)

                elif state == STATE_SPIRAL_P2:
                    spiral_for_point_2(pi, shared_data, movement_queue)

                elif state == STATE_PREDICT_P3:
                    predict_point_3(pi, shared_data, movement_queue)

                elif state == STATE_COMPLETE:
                    print("\n--- ACQUISITION COMPLETE: 3 points found. Initializing EKF. ---\n")
                    shared_data["ekf_start"].value = True
                    shared_data["acquire_points"].value = False  # Turn off the acquisition trigger
                    shared_data["acquisition_state"].value = STATE_TRACKING  # Move to tracking state

            elif state == STATE_TRACKING and shared_data['ekf_running'].value:
                # EKF Tracking logic (Unchanged)
                track_target(pi,
                             shared_data["predicted_azimuth"].value,
                             shared_data["predicted_elevation"].value,
                             0.0001, movement_queue, shared_data)

            else:
                # Reset if acquire is toggled off
                if not shared_data['acquire_points'].value and state != STATE_SEARCHING and state != STATE_TRACKING:
                    print("[MotorControl] Acquisition cancelled. Resetting to SEARCHING.")
                    shared_data['acquisition_state'].value = STATE_SEARCHING

            sleep(0.05)

    finally:
        # ... (cleanup code is unchanged) ...
        print("[MotorControl] Shutting down...")