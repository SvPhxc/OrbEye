from multiprocessing import Process, Manager
from webcam_test import run_tracking
import time

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        p = Process(target=run_tracking, args=(shared_data,))
        p.start()

        try:
            while True:
                target = shared_data.get("target")
                if target:
                    print(f"Target at: {target}")
                else:
                    print("Waiting for lock...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping tracking...")
            p.terminate()
