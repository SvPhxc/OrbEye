# simple_tracker.py

import numpy as np
import time
import traceback


def run_simple_tracker(shared_data):
    """
    A simple, reactive tracker. Instead of predicting movement with a Kalman
    filter, it simply tells the hardware to re-scan the area around the
    last successful measurement. It reuses the same HF_TRACKING hardware
    state as the EKF.
    """
    print("[SimpleTracker] Process started.")

    # State for this simple tracker
    is_initialized = False
    last_known_good_position = {'az': 90.0, 'el': 45.0}

    while not shared_data["shutdown"].value:
        try:
            # --- This tracker only runs if the EKF is DISABLED ---
            if shared_data["use_ekf_tracker"].value:
                time.sleep(0.5)
                is_initialized = False  # Reset if EKF is re-enabled
                continue

            # --- PHASE 1: WAIT FOR ACQUISITION & INITIALIZE ---
            if not is_initialized:
                if shared_data["lidar_track_mode_active"].value:
                    shared_data["lidar_track_mode_active"].value = False

                acquirer_status = shared_data["acquirer_status"].value
                if acquirer_status == 2 and shared_data["points_count"].value >= 1:  # 2:done
                    print("[SimpleTracker] Acquisition complete. Initializing tracker.")
                    points_buffer = shared_data["points_buffer"][:]
                    # Use the last acquired point as our starting position
                    last_point_idx = (shared_data["points_count"].value - 1) * 5
                    last_known_good_position['az'] = points_buffer[last_point_idx]
                    last_known_good_position['el'] = points_buffer[last_point_idx + 1]

                    is_initialized = True
                    shared_data["ekf_initialized"].value = True  # Use same flag for GUI status
                    shared_data["lidar_track_mode_active"].value = True  # Auto handoff
                    shared_data["acquirer_status"].value = 0
                    print(
                        f"[SimpleTracker] Initialized at ({last_known_good_position['az']:.1f}, {last_known_good_position['el']:.1f}). Engaging tracking.")
                else:
                    if acquirer_status == 3: shared_data["acquirer_status"].value = 0
                    time.sleep(0.2)
                continue

            # --- PHASE 2: ACTIVE REACTIVE TRACKING ---
            if not shared_data["lidar_track_mode_active"].value:
                is_initialized = False
                shared_data["ekf_initialized"].value = False
                time.sleep(0.2)
                continue

            # --- THE CORE "FAKE PREDICT" -> SCAN -> UPDATE LOOP ---

            # 1. "PREDICT": Our prediction is just the last known good position.
            pred_az = last_known_good_position['az']
            pred_el = last_known_good_position['el']

            # 2. REQUEST SCAN: Tell hardware to scan at this position.
            shared_data["predicted_azimuth"].value = pred_az
            shared_data["predicted_elevation"].value = pred_el
            shared_data["new_prediction_available"].value = True

            # 3. WAIT for hardware to finish its scan
            wait_start_time = time.time()
            while not shared_data["refined_measurement_updated"].value:
                time.sleep(0.002)
                if not shared_data["lidar_track_mode_active"].value or shared_data["shutdown"].value or shared_data[
                    "use_ekf_tracker"].value: break
                if time.time() - wait_start_time > 1.5:
                    print("[SimpleTracker] Timeout waiting for hardware. Disabling.");
                    shared_data["lidar_track_mode_active"].value = False;
                    break
            if not shared_data["lidar_track_mode_active"].value: continue

            # 4. UPDATE: Consume the refined measurement from HW
            refined_meas = shared_data["refined_measurement"][:];
            shared_data["refined_measurement_updated"].value = False
            az_deg, el_deg, dist_m, _, _ = refined_meas

            if dist_m > 500:  # Failure signal from hardware
                print("[SimpleTracker] Lost track signal. Disabling.");
                shared_data["lidar_track_mode_active"].value = False;
                continue

            # The new best position becomes our next "prediction"
            last_known_good_position['az'] = az_deg
            last_known_good_position['el'] = el_deg

            # Publish results for GUI
            shared_data["estimated_azimuth"].value = az_deg
            shared_data["estimated_elevation"].value = el_deg
            shared_data["ekf_confidence"].value = 0.5  # A dummy confidence value

        except Exception as e:
            print(f"[SimpleTracker] CRITICAL ERROR: {e}");
            traceback.print_exc();
            time.sleep(1)

    print("[SimpleTracker] Shutting down.")