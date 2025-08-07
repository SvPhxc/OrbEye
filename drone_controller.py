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

    # Wait for the motor controller to acknowledge and start the move
    start_time = time.time()
    while shared_data["go_to_target"].value:
        time.sleep(0.05)
        if time.time() - start_time > 5: # 5-second timeout
            print("[DroneController] Warning: Motor controller took too long to respond.")
            break

def run_drone_control(shared_data):
    """
    When enabled, controls drone movement based on Kalman filter predictions.
    """
    print("[DroneController] Starting...")
    
    # Wait for the Kalman Filter to initialize
    print("[DroneController] Waiting for EKF initialization...")
    while not shared_data["ekf_initialized"].value:
        if shared_data["shutdown"].value:
            print("[DroneController] Shutdown during init wait. Exiting.")
            return
        time.sleep(0.5)
    
    print("[DroneController] EKF initialized. Awaiting 'Follow' command from GUI.")

    circling_radius_deg = 5.0
    circling_steps = 8
    
    while not shared_data["shutdown"].value:
        # --- THIS IS THE MAIN CONTROL SWITCH ---
        if not shared_data["follow_drone_enabled"].value:
            time.sleep(0.2) # Sleep while idle to reduce CPU usage
            continue
        
        try:
            # 1. Get latest prediction from the Kalman Filter
            with shared_data['predicted_azimuth'].get_lock():
                predicted_az = shared_data['predicted_azimuth'].value
            with shared_data['predicted_elevation'].get_lock():
                predicted_el = shared_data['predicted_elevation'].value
            
            print(f"[DroneController] Following mode ACTIVE. Target: Az={predicted_az:.2f}, El={predicted_el:.2f}")

            # 2. Command motors to the predicted point
            command_motors_to_target(predicted_az, predicted_el, shared_data)
            print("[DroneController] Arrived at predicted target.")
            time.sleep(0.5)

            # 3. Perform a circular scan around the point to re-acquire the target
            print("[DroneController] Performing circular scan...")
            for i in range(circling_steps + 1): # +1 to return to center
                if not shared_data["follow_drone_enabled"].value or shared_data["shutdown"].value:
                    print("[DroneController] Following disabled during circle scan.")
                    break

                angle = (2 * math.pi / circling_steps) * i
                offset_az = circling_radius_deg * math.cos(angle)
                offset_el = circling_radius_deg * math.sin(angle)
                
                # On the last step, move back to the center prediction
                if i == circling_steps:
                    offset_az, offset_el = 0, 0

                command_motors_to_target(predicted_az + offset_az, predicted_el + offset_el, shared_data)
                time.sleep(0.2)

            print("[DroneController] Circle scan complete. Awaiting next prediction update.")
            time.sleep(1) # Wait before repeating the whole process

        except Exception as e:
            print(f"[DroneController] An error occurred: {e}")
            time.sleep(1)

    print("[DroneController] Shutting down.")
