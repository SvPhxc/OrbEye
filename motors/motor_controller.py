from math import degrees
import pigpio
import RPi.GPIO as GPIO
from time import sleep
import threading
import queue
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(3, GPIO.OUT)
GPIO.setup(2, GPIO.OUT)

GPIO.setup(6, GPIO.OUT)
GPIO.output(6, GPIO.HIGH)
# delay = 0.0001
GPIO.output(4, GPIO.LOW)
pwm = pigpio.pi()
pwm.set_mode(13, pigpio.OUTPUT)
pwm.set_PWM_frequency(13, 50)
movement_queue = queue.Queue()

#4 enable 3 dir 2 step 6 is mode
def Step():
    global degrees_moved, cumulative_error
    
    while True:
        command = movement_queue.get()
        if command is None:
            print("Stepper worker received stop signal.")
            break
        print(f"Received command: {command}")
        direction, degrees, delay = command
        ideal_microsteps = degrees / 0.05625
        total_microsteps_to_consider = ideal_microsteps + cumulative_error
        actual_microsteps_to_take = round(total_microsteps_to_consider)
        cumulative_error = total_microsteps_to_consider - actual_microsteps_to_take
        print(f"Direction: {direction}, Degrees: {degrees}, Ideal microsteps: {ideal_microsteps}, "
              f"Actual microsteps: {actual_microsteps_to_take}, Cumulative error: {cumulative_error}")
        if direction == 'right':
            GPIO.output(2, True)
        else:
            GPIO.output(2, False)
        for i in range(actual_microsteps_to_take):
            GPIO.output(3, GPIO.HIGH)
            sleep(delay)
            GPIO.output(3, GPIO.LOW)
            sleep(delay)
        actual_degrees_this_move = actual_microsteps_to_take * 0.5625
        if direction == 'left':
            degrees_moved -= actual_degrees_this_move
        else:
            degrees_moved += actual_degrees_this_move
        print(f"Degrees moved: {degrees_moved}")
        movement_queue.task_done()

def control_servo(direction, degrees, delay):
    global degreess

    print(f"Control servo called with direction: {direction}, degrees: {degrees}")
    if direction == 'up':
        test = 500+((degreess+degrees)/0.09)
        pwm.set_servo_pulsewidth(13,test)
        degreess+=degrees
    elif direction == 'down':
        test = 500+((degreess-degrees)/0.09)
        pwm.set_servo_pulsewidth(13,test)
        degreess -= degrees
    print(f"Servo position updated. Degrees moved: {degreess}")
    sleep(delay)  # Simulate delay for servo movement

def move(direction, degrees, delay):
    print(f"Move called with direction: {direction}, degrees: {degrees}")
    if direction in ['left', 'right']:
        command = (direction, degrees, delay)
        movement_queue.put(command)
        print(f"Command added to queue: {command}")
    elif direction in ['up', 'down']:
        control_servo(direction, degrees, delay)

def stop_stepper_worker():
    print("Stopping stepper worker.")
    movement_queue.put(None)


# --- Example Usage ---
if __name__ == '__main__':
    # Create and start the single worker thread.
    # daemon=True means the script will exit even if this thread is still running.
    worker_thread = threading.Thread(target=Step, daemon=True)
    worker_thread.start()
    degreess = 0
    degrees_moved = 0.0
    cumulative_error = 0.0

    print("MAIN: Stepper worker thread started in the background.")
    print("MAIN: Queuing up a sequence of movements...\n")
    

    move('up', 45, 0.1)  # Move servo up by 25 degrees
    sleep(1) 
    # move('down', 0, 0.1)  # Move servo up by 25 degrees
     # Wait a moment to show the servo starting
    for i in range(40):
        move('up', i*2, 0.5)
        sleep(0.0001*32*2*i)
        move('left', i*2, 0.0001)
        sleep(0.0001*32*2*i)
        move('down', i*3, 0.5)
        sleep(0.0001*32*2*i)
        move('right', i*2, 0.0001)
        sleep(0.0001*32*2*i*2)

    
    
    # move('up',0)
    # sleep(1)  # Wait a moment to show the worker starting
    # # Queue several movements instantly.
    # move('right', 180)
    # move('left', 360)
    # move('right', 180)
    # move('right', 1.5)

    # print("\nMAIN: All stepper moves are queued. The main program is free.")
    # print("MAIN: Now commanding the servo while the stepper works.\n")

    # # The servo can move immediately, even while the stepper queue is being processed.  # Wait a moment to show the worker starting
    # move('up',25)
    # sleep(3)
    # move('up',25)
    # sleep(3)
    # move('down', 50)
    # for i in range(50):
    #     move('up',1)
    #     sleep(0.01)

    

    
    print("\nMAIN: Waiting for all queued movements to complete...")
    movement_queue.join()  # This blocks until the queue is empty
    print("MAIN: Queue is empty. All movements are done.")

    # Cleanly stop the worker thread
    stop_stepper_worker()
    worker_thread.join(timeout=1)  # Wait briefly for the thread to exit

    print("\nScript finished.")
    print(f"Final tracked position: {degrees_moved:.4f} degrees.")
    GPIO.output(4, GPIO.HIGH)  # Disable the stepper motor
    # GPIO.cleanup()  # Clean up GPIO settings
      # Stop the PWM on the servo pin
    pwm.set_servo_pulsewidth(13, 0)  # Stop the PWM on the servo pin
