from multiprocessing import Process, Manager
from webcam.blob_tracker import run_tracking
from motors.controller import run_motor_control  # if used
from GUI import run_gui
import time

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data["direction"] = None
        shared_data["target"] = None
        shared_data["selected_blob"] = None
        shared_data["shutdown"] = False
        shared_data["commanding"] = None
        shared_data["Range"] = None
        shared_data["Strength"] = None

        shared_data["lidar_array"] = None
        shared_data["pan_tilt"] = None

        #p1 = Process(target=run_tracking, args=(shared_data,))
        #p2 = Process(target=run_motor_control, args=(shared_data,))
        p3 = Process(target=run_gui, args=(shared_data,))
        #p1.start()
        #p2.start()
        p3.start()

        try:
            while not shared_data["shutdown"]:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Ctrl+C pressed")

        print("Terminating processes...")
        #p1.terminate()
        #p2.terminate()
        p3.terminate()
        #p1.join()
        #p2.join()
        p3.join
        print("Program exited cleanly")
