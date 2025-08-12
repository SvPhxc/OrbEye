# tracker_logic.py

import time
import numpy as np

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Target Acquisition (Raster Scan)
SCAN_PAN_MIN, SCAN_PAN_MAX = 5, 175  # Search within a 170-degree frontal arc
SCAN_TILT_MIN, SCAN_TILT_MAX = 10, 80 # Search within a 70-degree vertical range
SCAN_STEP_PAN = 5.0  # Degrees to step in the pan axis for each row
SCAN_STEP_TILT = 5.0 # Degrees to step in the tilt axis after each row

# Target Detection
TARGET_MAX_DIST_M = 8.0  # Maximum distance to consider a valid target (in meters)
TARGET_CONFIRM_READINGS = 2 # How many consecutive valid readings to confirm a target

# Target Tracking
TRACK_DITHER_ANGLE = 2.0  # How far (in degrees) to look left/right/up/down from center
TRACK_KP_PAN = 0.8        # Proportional gain for pan correction
TRACK_KP_TILT = 0.8       # Proportional gain for tilt correction

# Re-acquisition (Spiral Search)
REACQUIRE_SPIRAL_START_RADIUS = 5.0  # Starting radius of the spiral search (degrees)
REACQUIRE_SPIRAL_MAX_RADIUS = 45.0   # Max radius before giving up (degrees)
REACQUIRE_SPIRAL_STEP_SIZE = 5.0     # Angle step size for the spiral (degrees)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def command_gimbal(shared_data, pan, tilt):
    """Commands the gimbal to a specific pan/tilt and waits for arrival."""
    shared_data["target_azimuth"].value = pan
    shared_data["target_elevation"].value = tilt
    shared_data["go_to_target"].value = True
    shared_data["target_reached"].value = False

    # Wait until the hardware controller confirms the target position is reached
    wait_start_time = time.monotonic()
    while not shared_data["target_reached"].value:
        time.sleep(0.05)
        # Timeout to prevent getting stuck if hardware fails
        if time.monotonic() - wait_start_time > 5.0:
            print("[Tracker] WARN: Timeout waiting for gimbal to reach target.")
            return False
    # Short sleep for vibrations to settle before taking a reading
    time.sleep(0.1)
    return True

def get_stable_lidar_reading(shared_data):
    """Gets the latest LiDAR reading from shared memory."""
    # The hardware controller continuously updates this, so we just grab the latest
    dist_cm = shared_data['lidar_data'][0]
    return dist_cm / 100.0  # Convert to meters

# ==============================================================================
# CORE LOGIC IMPLEMENTATION
# ==============================================================================

def acquire_target(shared_data):
    """
    Performs a raster scan of a predefined area to find a target.
    Implements the 'Target Acquisition Strategy'.
    """
    print("[Tracker] Starting target acquisition scan...")
    shared_data["tracker_status"].value = 1  # 1: ACQUIRING

    # Start scan from the top-left
    current_tilt = SCAN_TILT_MAX
    pan_direction = 1  # 1 for right, -1 for left

    while current_tilt >= SCAN_TILT_MIN:
        # Define the pan range for the current row
        pan_start, pan_end = (SCAN_PAN_MIN, SCAN_PAN_MAX) if pan_direction == 1 else (SCAN_PAN_MAX, SCAN_PAN_MIN)

        # Scan across the pan axis
        for current_pan in np.arange(pan_start, pan_end, SCAN_STEP_PAN * pan_direction):
            # Check for shutdown signal from main process
            if shared_data["shutdown"].value or not shared_data["auto_track_active"].value:
                return None, None

            command_gimbal(shared_data, current_pan, current_tilt)
            distance = get_stable_lidar_reading(shared_data)

            # Logic to detect a potential target
            if distance < TARGET_MAX_DIST_M:
                print(f"[Tracker] Potential target at P:{current_pan:.1f}, T:{current_tilt:.1f}, D:{distance:.2f}m")
                # Once a potential target is found, confirm it
                # For simplicity, we lock on first detection. A more robust system
                # would check a few nearby points before confirming.
                print("[Tracker] Target Acquired!")
                return current_pan, current_tilt

        # Move to the next tilt level and reverse pan direction
        current_tilt -= SCAN_STEP_TILT
        pan_direction *= -1

    print("[Tracker] Acquisition scan complete. No target found.")
    return None, None


def track_target(shared_data, target_pan, target_tilt):
    """
    Keeps the LiDAR aimed at the target and follows its movement.
    Implements the 'Target Tracking Strategy'.
    """
    shared_data["tracker_status"].value = 2  # 2: TRACKING

    # Center on the target's last known position first
    command_gimbal(shared_data, target_pan, target_tilt)
    center_dist = get_stable_lidar_reading(shared_data)

    # If we immediately lose the target, enter re-acquisition
    if center_dist > TARGET_MAX_DIST_M:
        print("[Tracker] Target lost immediately after lock. Re-acquiring.")
        return "REACQUIRE"

    # --- Perform a small, localized scan (dithering) ---
    # Check Left, Right, Up, Down to find the target's edges
    readings = {}
    # Check Left
    command_gimbal(shared_data, target_pan - TRACK_DITHER_ANGLE, target_tilt)
    readings['left'] = get_stable_lidar_reading(shared_data)
    # Check Right
    command_gimbal(shared_data, target_pan + TRACK_DITHER_ANGLE, target_tilt)
    readings['right'] = get_stable_lidar_reading(shared_data)
    # Check Up
    command_gimbal(shared_data, target_pan, target_tilt + TRACK_DITHER_ANGLE)
    readings['up'] = get_stable_lidar_reading(shared_data)
    # Check Down
    command_gimbal(shared_data, target_pan, target_tilt - TRACK_DITHER_ANGLE)
    readings['down'] = get_stable_lidar_reading(shared_data)

    # --- Apply Proportional Control based on dithering results ---
    pan_error = 0
    if readings['left'] < TARGET_MAX_DIST_M and readings['right'] > TARGET_MAX_DIST_M:
        pan_error = -1 # Target moved left
    elif readings['left'] > TARGET_MAX_DIST_M and readings['right'] < TARGET_MAX_DIST_M:
        pan_error = 1  # Target moved right

    tilt_error = 0
    if readings['down'] < TARGET_MAX_DIST_M and readings['up'] > TARGET_MAX_DIST_M:
        tilt_error = -1 # Target moved down
    elif readings['down'] > TARGET_MAX_DIST_M and readings['up'] < TARGET_MAX_DIST_M:
        tilt_error = 1  # Target moved up

    # If all readings are bad, target is lost
    if all(r > TARGET_MAX_DIST_M for r in readings.values()):
        return "REACQUIRE"

    # --- Calculate and apply the correction ---
    # This is our Proportional Controller
    pan_adjustment = TRACK_KP_PAN * pan_error
    tilt_adjustment = TRACK_KP_TILT * tilt_error

    new_pan = target_pan + pan_adjustment
    new_tilt = target_tilt + tilt_adjustment

    # Update the target's position for the next loop iteration
    shared_data["tracker_target_pan"].value = new_pan
    shared_data["tracker_target_tilt"].value = new_tilt

    # Command the gimbal to the newly calculated center
    command_gimbal(shared_data, new_pan, new_tilt)
    print(f"[Tracker] Corrected Position -> P:{new_pan:.1f}, T:{new_tilt:.1f}")

    return "TRACKING" # Continue tracking

def reacquire_target(shared_data, last_known_pan, last_known_tilt):
    """
    Performs an expanding spiral search from the target's last known position.
    """
    print("[Tracker] Attempting to re-acquire target with spiral search...")
    shared_data["tracker_status"].value = 3  # 3: REACQUIRING

    for radius in np.arange(REACQUIRE_SPIRAL_START_RADIUS, REACQUIRE_SPIRAL_MAX_RADIUS, REACQUIRE_SPIRAL_STEP_SIZE):
        for angle in np.arange(0, 360, REACQUIRE_SPIRAL_STEP_SIZE):
             if shared_data["shutdown"].value or not shared_data["auto_track_active"].value:
                return None, None

             # Convert polar (radius, angle) to Cartesian (pan, tilt) offset
             pan_offset = radius * np.cos(np.radians(angle))
             tilt_offset = radius * np.sin(np.radians(angle))

             check_pan = last_known_pan + pan_offset
             check_tilt = last_known_tilt + tilt_offset

             command_gimbal(shared_data, check_pan, check_tilt)
             distance = get_stable_lidar_reading(shared_data)

             if distance < TARGET_MAX_DIST_M:
                 print(f"[Tracker] Target Re-acquired at P:{check_pan:.1f}, T:{check_tilt:.1f}")
                 return check_pan, check_tilt

    print("[Tracker] Spiral search failed. Target is lost.")
    return None, None

# ==============================================================================
# MAIN PROCESS FUNCTION
# ==============================================================================

def run_tracker_logic(shared_data):
    """The main entry point and state machine for the tracker process."""
    print("[Tracker] Logic process started.")
    current_state = "IDLE"
    target_pan, target_tilt = None, None

    while not shared_data["shutdown"].value:
        # Check for external commands (from GUI)
        if shared_data["auto_track_active"].value and current_state == "IDLE":
            current_state = "ACQUIRING"
        elif not shared_data["auto_track_active"].value and current_state != "IDLE":
            print("[Tracker] Auto-tracking disabled by user.")
            current_state = "IDLE"
            shared_data["tracker_status"].value = 0 # 0: IDLE
            # Stop any current movement
            shared_data["go_to_target"].value = False
            continue

        # --- State Machine ---
        if current_state == "IDLE":
            time.sleep(0.2)
            continue

        elif current_state == "ACQUIRING":
            target_pan, target_tilt = acquire_target(shared_data)
            if target_pan is not None:
                shared_data["tracker_target_pan"].value = target_pan
                shared_data["tracker_target_tilt"].value = target_tilt
                current_state = "TRACKING"
            else:
                # If no target found, go back to idle and wait for next command
                current_state = "IDLE"
                shared_data["auto_track_active"].value = False
                shared_data["tracker_status"].value = 0

        elif current_state == "TRACKING":
            # Get latest position from shared data
            current_target_pan = shared_data["tracker_target_pan"].value
            current_target_tilt = shared_data["tracker_target_tilt"].value
            # Perform one tracking iteration
            track_result = track_target(shared_data, current_target_pan, current_target_tilt)
            if track_result == "REACQUIRE":
                current_state = "REACQUIRING"
            elif shared_data["shutdown"].value or not shared_data["auto_track_active"].value:
                current_state = "IDLE"

        elif current_state == "REACQUIRING":
            last_pan = shared_data["tracker_target_pan"].value
            last_tilt = shared_data["tracker_target_tilt"].value
            target_pan, target_tilt = reacquire_target(shared_data, last_pan, last_tilt)
            if target_pan is not None:
                shared_data["tracker_target_pan"].value = target_pan
                shared_data["tracker_target_tilt"].value = target_tilt
                current_state = "TRACKING"
            else:
                # If re-acquisition fails, go idle
                current_state = "IDLE"
                shared_data["auto_track_active"].value = False
                shared_data["tracker_status"].value = 0

    print("[Tracker] Logic process shut down.")