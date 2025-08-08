from multiprocessing import Process, Queue, Array, Value
from webcam.blob_tracker import run_tracking
from motors.motor_controller import run_motor_control  # if used
from LiDAR.lidar_handler import run_lidar
from LiDAR.Kalman_Filter import run_ekf_tracker
from tracking.tracker import tracking  # if used
from GUI import run_gui
import time


if __name__ == "__main__":
    # Shared memory setup
    lidar_data = Array('d', 3)  # [distance, strength, timestamp]
    backgorund_data = Array('d', 4)  #background array after scan
    shutdown_flag = Value('b', False)  # Boolean flag
    scan_trigger = Value('b', False)  # GUI sets this to True to trigger scan
    tilt_up = Value('b', False)
    tilt_down = Value('b', False)
    pan_left = Value('b', False)
    pan_right = Value('b', False)
    flipped = Value('b', False)  
    go_to_target = Value('b', False)
    save_background = Value('b', False)
    background_ready = Value('b', False)
    acquire_points = Value('b', False)   # GUI triggers this
    ekf_start = Value('b', False)        # set True after 3 points collected
    ekf_running = Value('b', False)      # EKF has started
    points_buffer = Array('d', 12)       # 3 points x [az, el, dist_m, strength]
    points_count = Value('i', 0)  # Number of points collected
    ekf_initialized = Value('b', False)  # Flag to indicate if EKF is initialized
    estimated_azimuth = Value('d', 0.0)
    estimated_elevation = Value('d', 0.0)
    predicted_azimuth = Value('d', 0.0)
    predicted_elevation = Value('d', 0.0)
    ekf_confidence = Value('d', 0.0)
    lidar_acceptance_range = Array('d', [1.0, 2.0])  # min_m, max_m
    background_path = "background_data.npy"

    # Optional shared values for future expansion
    direction = Value('i', -1)
    target = Value('i', -1)
    commanding = Value('i', -1)
    stepper_degrees = Value('d', 0.0)  # For stepper motor position
    cumulative_error = Value('d', 0.0)  # For PID control
    servo_degrees = Value('d', 90.0)  # For servo position
    target_azimuth = Value('d', 0.0)
    target_elevation = Value('d', 0.0)
    # Background LiDAR data for satellite detection
    background_lidar = Array('d', 360 * 90 * 2)  # [azimuth, elevation, [strength, range]]
    satellite_points = Array('d', 4)  # [azimuth, elevation, strength, range]
    satellite_detected = Value('b', False)  # Flag to indicate if a satellite is detected
    acquisition_state = Value('i', 0)  # 0: SEARCHING, 1: CENTERING_P1, 2: SPIRAL_P2, 3: PREDICT_P3, 4: COMPLETE
    initial_points = Array('d', [0.0] * 12)  # Stores 3 points (az, el, dist, time)
    best_strength_point = Array('d', [0.0] * 3)  # Temp storage for finding peak strength (az, el, str)

    
    

    # Build shared_data dictionary
    shared_data = {
        "lidar_data": lidar_data,
        "background_data": backgorund_data,
        "shutdown": shutdown_flag,
        "direction": direction,
        "target": target,
        "commanding": commanding,
        "scan_trigger": scan_trigger,  # GUI sets this to True to trigger scan
        "stepper_degrees": stepper_degrees,
        "cumulative_error": cumulative_error,  # For PID
        "servo_degrees": servo_degrees,  # For servo position
        "tilt_up": tilt_up,
        "tilt_down": tilt_down,
        "pan_left": pan_left,
        "pan_right": pan_right,
        "flipped": flipped,  # For GUI to know if the camera is flipped
        "go_to_target": go_to_target,  # For GUI to trigger go to target
        "target_azimuth": target_azimuth,  # For target azimuth
        "target_elevation": target_elevation,  # For target elevation
        "background_lidar": background_lidar,  # Shared background LiDAR data
        "satellite_points": satellite_points,  # Shared array for satellite points
        "satellite_detected": satellite_detected,  # Flag for satellite detection
        "save_background": save_background,  # Flag to save background data
        "background_ready": background_ready,       
        "background_path": background_path,
        "acquire_points": acquire_points,
        "ekf_start": ekf_start,
        "ekf_running": ekf_running,
        "points_buffer": points_buffer,
        "points_count": points_count,
        "ekf_initialized": ekf_initialized,
        "estimated_azimuth": estimated_azimuth,
        "estimated_elevation": estimated_elevation,
        "predicted_azimuth": predicted_azimuth,
        "predicted_elevation": predicted_elevation,
        "ekf_confidence": ekf_confidence, 
        "lidar_acceptance_range": lidar_acceptance_range,  # [min_m, max_m]
        "initial_points": initial_points,  # Stores 3 points (az, el, dist, time)
        "best_strength_point": best_strength_point,  # Temp storage for finding peak strength (az, el, str)
        "acquisition_state": acquisition_state,  # 0: SEARCHING, 1: CENTERING_P1, 2: SPIRAL_P2, 3: PREDICT_P3, 4: COMPLETE

    }

    # Start processes
    # p1 = Process(target=run_tracking, args=(shared_data,))
    movement_queue = Queue()
    p2 = Process(target=run_motor_control, args=(shared_data, movement_queue))
    p3 = Process(target=run_gui, args=(shared_data, movement_queue))
    p4 = Process(target=run_lidar, args=(shared_data,))
    p5 = Process(target=run_ekf_tracker, args=(shared_data,))

    # p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    try:
        while not shutdown_flag.value:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Ctrl+C pressed")
        shutdown_flag.value = True

    print("Terminating processes...")
    # p1.terminate()
    p2.terminate()
    p3.terminate()
    p4.terminate()
    p5.terminate()

    # p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    print("Program exited cleanly")
