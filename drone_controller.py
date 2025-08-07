# drone_controller.py

import time
import math

def command_motors_to_target(azimuth, elevation, shared_data):
    """Sets shared data to command the motor_controller to a specific target."""
    with shared_data["target_azimuth"].get_lock():
        shared_data["target_azimuth"].value = azimuth
    with shared_data["target_elevation"].get_lock():
        shared_data["target_elevation"].value = elevation
    with shared_data["go_to_target"].get_lock():
        shared_data["go_to_target"].value = True

    # Give the motor controller time to start the move
    time.sleep(0.1) 
    
    # Wait until the go_to_target flag is cleared by the motor controller
    while shared_data["go_to_target"].value:
        time.sleep(0.05)


def run_drone_control(shared_data):
    """
    Controls the drone's motors based on predictions from the Kalman filter.
    It moves to the predicted point, circles it, and then repeats.
    """
    print("[DroneController] Starting...")

    # Wait until the Kalman Filter is initialized and ready to provide predictions
    print("[DroneController] Waiting for EKF initialization...")
    while not shared_data["ekf_initialized"].value:
        if shared_data["shutdown"].value:
            print("[DroneController] Shutdown signal received during init wait. Exiting.")
            return
        time.sleep(0.5)
    
    print("[DroneController] EKF is initialized. Starting control loop.")

    circling_radius_deg = 5.0  # The radius of the circular search pattern in degrees
    circling_steps = 8         # The number of points in the circle
    
    last_predicted_az = 0
    last_predicted_el = 0

    while not shared_data["shutdown"].value:
        try:
            # 1. Get the latest prediction from the Kalman Filter
            with shared_data['predicted_azimuth'].get_lock():
                predicted_az = shared_data['predicted_azimuth'].value
            with shared_data['predicted_elevation'].get_lock():
                predicted_el = shared_data['predicted_elevation'].value

            # Only act if the prediction has changed significantly
            if abs(predicted_az - last_predicted_az) < 0.5 and abs(predicted_el - last_predicted_el) < 0.5:
                time.sleep(0.1)
                continue

            last_predicted_az = predicted_az
            last_predicted_el = predicted_el
            
            print(f"[DroneController] New target received: Az={predicted_az:.2f}, El={predicted_el:.2f}")

            # 2. Command the motors to go to the predicted point
            print("[DroneController] Moving to predicted target...")
            command_motors_to_target(predicted_az, predicted_el, shared_data)
            print("[DroneController] Arrived at target.")

            time.sleep(0.5) # Pause at the target before circling

            # 3. Perform a circular scan around the point
            print(f"[DroneController] Starting circular scan with radius {circling_radius_deg}...")
            for i in range(circling_steps):
                if shared_data["shutdown"].value: break

                angle = (2 * math.pi / circling_steps) * i
                offset_az = circling_radius_deg * math.cos(angle)
                offset_el = circling_radius_deg * math.sin(angle)

                circle_point_az = predicted_az + offset_az
                circle_point_el = predicted_el + offset_el
                
                print(f"[DroneController] Circling to: Az={circle_point_az:.2f}, El={circle_point_el:.2f}")
                command_motors_to_target(circle_point_az, circle_point_el, shared_data)
                time.sleep(0.2) # Short pause at each point in the circle

            print("[DroneController] Circle scan complete. Awaiting next prediction.")
            time.sleep(1) # Wait before fetching the next global prediction

        except Exception as e:
            print(f"[DroneController] An error occurred: {e}")
            time.sleep(1)

    print("[DroneController] Shutting down.")
