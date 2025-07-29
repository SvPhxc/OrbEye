import time

def run_motor_control(shared_data):
    while not shared_data.get("shutdown", False):
        direction = shared_data.get("direction", None)

        if direction == "left":
            print("Pan left")
        elif direction == "right":
            print("Pan right")
        elif direction == "up":
            print("Tilt up")
        elif direction == "down":
            print("Tilt down")
        elif direction == "center":
            print("Target centered")
        else:
            print("No target")

        time.sleep(0.1)
