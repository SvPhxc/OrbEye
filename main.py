from multiprocessing import Process, Manager
from webcam_test import run_tracking
from motorcontroller import run_motor_control
import time

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data["direction"] = None
        shared_data["target"] = None
        shared_data["selected_blob"] = None
        shared_data["shutdown"] = False  # The shutdown flag

        p1 = Process(target=run_tracking, args=(shared_data,))
        p2 = Process(target=run_motor_control, args=(shared_data,))
        p1.start()
        p2.start()

        # Monitor shutdown flag
        try:
            while not shared_data["shutdown"]:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Ctrl+C pressed")

        print("Terminating processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("Program exited cleanly")
