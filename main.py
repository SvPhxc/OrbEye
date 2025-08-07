# main.py

from multiprocessing import Process, Queue, Array, Value
from webcam.blob_tracker import run_tracking
from motors.motor_controller import run_motor_control
from LiDAR.lidar_handler import run_lidar
from LiDAR.Kalman_Filter import run_ekf_tracker, setup_ekf_shared_data
from drone_controller import run_drone_control
from GUI import run_gui
import time

if __name__ == "__main__":
    # Shared memory setup
    lidar_data = Array('d', 3)  # [distance, strength, timestamp]
    shutdown_flag = Value('b', False)
    scan_trigger = Value('b', False)
    save_background = Value('b', False)
    
    # Manual motor controls
    tilt_up = Value('b', False)
    tilt_down = Value('b', False)
    pan_left = Value('b', False)
    pan_right = Value('b', False)
    
    # Motor/System State
    stepper_degrees = Value('d', 0.0)
    servo_degrees = Value('d', 90.0)
    cumulative_error = Value('d', 0.0)
    flipped = Value('b', False)
    
    # Targeting
    go_to_target = Value('b', False)
    target_azimuth = Value('d', 0.0)
    target_elevation = Value('d', 0.0)
    
    # LiDAR Background & Detection
    # Stores [strength, range] for each point. Azimuth: 0-359, Elevation: 0-89
    background_lidar = Array('d', 360 * 90 * 2) 
    satellite_points = Array('d', 4)  # [azimuth, elevation, strength, range]
    satellite_detected = Value('b', False)
    
    # --- NEW: Flag to enable/disable autonomous drone following ---
    follow_drone_enabled = Value('b', False)

    # Build shared_data dictionary
    shared_data = {
        "lidar_data": lidar_data,
        "shutdown": shutdown_flag,
        "scan_trigger": scan_trigger,
        "save_background": save_background,
        "tilt_up": tilt_up,
        "tilt_down": tilt_down,
        "pan_left": pan_left,
        "pan_right": pan_right,
        "stepper_degrees": stepper_degrees,
        "servo_degrees": servo_degrees,
        "cumulative_error": cumulative_error,
        "flipped": flipped,
        "go_to_target": go_to_target,
        "target_azimuth": target_azimuth,
        "target_elevation": target_elevation,
        "background_lidar": background_lidar,
        "satellite_points": satellite_points,
        "satellite_detected": satellite_detected,
        "follow_drone_enabled": follow_drone_enabled, # NEW
    }

    # Add EKF specific shared data
    shared_data = setup_ekf_shared_data(shared_data)
    
    # Process setup
    movement_queue = Queue()
    p_motor = Process(target=run_motor_control, args=(shared_data, movement_queue))
    p_gui = Process(target=run_gui, args=(shared_data, movement_queue))
    p_lidar = Process(target=run_lidar, args=(shared_data,))
    p_ekf = Process(target=run_ekf_tracker, args=(shared_data,))
    p_drone_controller = Process(target=run_drone_control, args=(shared_data,))

    # Start processes
    p_motor.start()
    p_gui.start()
    p_lidar.start()
    p_ekf.start()
    p_drone_controller.start()

    try:
        # The main process now simply waits for the GUI to exit or a manual interrupt
        p_gui.join()
        print("GUI closed, initiating shutdown...")
        shutdown_flag.value = True

    except KeyboardInterrupt:
        print("Ctrl+C pressed, initiating shutdown...")
        shutdown_flag.value = True

    print("Terminating processes...")
    # Give processes a moment to shut down cleanly
    time.sleep(2)

    # Terminate any stubborn processes
    if p_motor.is_alive(): p_motor.terminate()
    if p_lidar.is_alive(): p_lidar.terminate()
    if p_ekf.is_alive(): p_ekf.terminate()
    if p_drone_controller.is_alive(): p_drone_controller.terminate()
    
    print("Program exited cleanly")
