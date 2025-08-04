from multiprocessing import Process, Queue, Array, Value
from webcam.blob_tracker import run_tracking
from motors.motor_controller import run_motor_control  # if used
from LiDAR.lidar_handler import run_lidar
from tracking.tracker import tracking  # if used
from GUI import run_gui
import time

if __name__ == "__main__":
    # Shared memory setup
    lidar_data = Array('d', 3)  # [distance, strength, timestamp]
    shutdown_flag = Value('b', False)  # Boolean flag
    scan_trigger = Value('b', False)  # GUI sets this to True to trigger scan

    # Optional shared values for future expansion
    direction = Value('i', -1)
    target = Value('i', -1)
    commanding = Value('i', -1)
    stepper_degrees = Value('d', 0.0)  # For stepper motor position
    cumulative_error = Value('d', 0.0)  # For PID control
    servo_degrees = Value('d', 90.0)  # For servo position

    # Build shared_data dictionary
    shared_data = {
        "lidar_data": lidar_data,
        "shutdown": shutdown_flag,
        "direction": direction,
        "target": target,
        "commanding": commanding,
        "scan_trigger": scan_trigger,  # GUI sets this to True to trigger scan
        "stepper_degrees": stepper_degrees,
        "cumulative_error": cumulative_error,  # For PID
        "servo_degrees": servo_degrees  # For servo position
    }

    # Start processes
    # p1 = Process(target=run_tracking, args=(shared_data,))
    movement_queue = Queue()
    p2 = Process(target=run_motor_control, args=(shared_data, movement_queue))
    p3 = Process(target=run_gui, args=(shared_data,))
    p4 = Process(target=run_lidar, args=(shared_data,))
    # p5 = Process(target=tracking, args=(shared_data,))

    # p1.start()
    p2.start()
    p3.start()
    p4.start()
    # p5.start()

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
    # p5.terminate()

    # p1.join()
    p2.join()
    p3.join()
    p4.join()
    # p5.join()
    print("Program exited cleanly")
