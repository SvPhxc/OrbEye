# drone_controller.py

import time
import math
from enum import Enum

# --- NEW: State machine for the controller ---
class TrackingState(Enum):
    IDLE = 0
    GATHERING = 1
    TRACKING = 2

def command_motors_to_target(azimuth, elevation, shared_data):
    """Sets shared data to command the motor_controller to a specific target."""
    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True

    start_time = time.time()
    while shared_data["go_to_target"].value:
        time.sleep(0.05)
        if time.time() - start_time > 5:
            print("[DroneController] Warning: Motor controller timeout.")
            break

def run_drone_control(shared_data):
    """
    Controls drone movement based on a state machine:
    IDLE -> GATHERING (first 3 points) -> TRACKING (predict & circle)
    """
    print("[DroneController] Starting...")
    
    while not shared_data["ekf_initialized"].value:
        if shared_data["shutdown"].value: return
        print("[DroneController] Waiting for EKF initialization...")
        time.sleep(0.5)
    
    print("[DroneController] EKF initialized. Ready for commands.")

    state = TrackingState.IDLE
    gathered_points = []
    GATHER_COUNT = 3  # Number of points to gather for initial trajectory

    while not shared_data["shutdown"].value:
        follow_enabled = shared_data["follow_drone_enabled"].value

        # --- State Machine Logic ---
        
        # If disabled, always reset to IDLE
        if not follow_enabled:
            if state != TrackingState.IDLE:
                print("[DroneController] Following disabled. Returning to IDLE.")
                state = TrackingState.IDLE
                gathered_points = []
            time.sleep(0.2)
            continue

        # If enabled, transition from IDLE to GATHERING
        if state == TrackingState.IDLE and follow_enabled:
            print("[DroneController] Follow mode enabled. Starting GATHERING phase.")
            state = TrackingState.GATHERING
            gathered_points = []

        # --- GATHERING State ---
        if state == TrackingState.GATHERING:
            print(f"[DroneController] Waiting for detection {len(gathered_points) + 1}/{GATHER_COUNT}...")
            # Wait for a new detection from the LiDAR/Kalman pipeline
            if shared_data["satellite_detected"].value:
                with shared_data["satellite_detected"].get_lock():
                    shared_data["satellite_detected"].value = False # Consume the flag

                with shared_data['estimated_azimuth'].get_lock():
                    est_az = shared_data['estimated_azimuth'].value
                with shared_data['estimated_elevation'].get_lock():
                    est_el = shared_data['estimated_elevation'].value
                
                gathered_points.append((est_az, est_el))
                print(f"[DroneController] Point {len(gathered_points)} gathered. Moving to Az={est_az:.1f}, El={est_el:.1f}")
                
                # Move directly to the estimated position of the detected point
                command_motors_to_target(est_az, est_el, shared_data)

                if len(gathered_points) >= GATHER_COUNT:
                    print("[DroneController] Initial points gathered. Transitioning to TRACKING phase.")
                    state = TrackingState.TRACKING
            time.sleep(0.1)

        # --- TRACKING State ---
        elif state == TrackingState.TRACKING:
            with shared_data['predicted_azimuth'].get_lock():
                predicted_az = shared_data['predicted_azimuth'].value
            with shared_data['predicted_elevation'].get_lock():
                predicted_el = shared_data['predicted_elevation'].value
            
            print(f"[DroneController] Tracking. Moving to PREDICTED target: Az={predicted_az:.2f}, El={predicted_el:.2f}")
            command_motors_to_target(predicted_az, predicted_el, shared_data)
            time.sleep(0.5)

            # Perform circular scan
            circling_radius_deg = 5.0
            circling_steps = 8
            print("[DroneController] Performing circular scan...")
            for i in range(circling_steps):
                if not shared_data["follow_drone_enabled"].value or shared_data["shutdown"].value:
                    break
                angle = (2 * math.pi / circling_steps) * i
                offset_az = circling_radius_deg * math.cos(angle)
                offset_el = circling_radius_deg * math.sin(angle)
                command_motors_to_target(predicted_az + offset_az, predicted_el + offset_el, shared_data)
                time.sleep(0.2)
            
            if not shared_data["follow_drone_enabled"].value: continue
            
            print("[DroneController] Circle scan complete. Awaiting next prediction.")
            time.sleep(1)

    print("[DroneController] Shutting down.")
