# --- CORRECTED SNIPPET: LiDAR/Kalman_Filter.py ---

def run_ekf_tracker(shared_data):
    # ... (setup code is the same) ...
    waiting_for_init = True
    # ... (shared memory refs are the same) ...

    while not shared_data["shutdown"].value:
        try:
            now = time.time()

            # --- EKF Initialization Logic ---
            if waiting_for_init and ekf_start.value:
                with points_count.get_lock():
                    k = points_count.value

                # We need at least 2 points to start
                if k >= 2:
                    pb = points_buffer

                    # --- CRITICAL FIX: Use REAL timestamps from the buffer ---
                    # Layout is now [az, el, dist, str, time]
                    p1_data = {'z': np.array([np.deg2rad(pb[0]), np.deg2rad(pb[1]), pb[2]]), 'time': pb[4]}
                    p2_data = {'z': np.array([np.deg2rad(pb[5]), np.deg2rad(pb[6]), pb[7]]), 'time': pb[9]}

                    init_ekf(ekf, [p1_data, p2_data])  # Pass data to init function

                    ekf.last_time = p2_data['time']  # Start the clock from the second measurement
                    ekf.initialized = True
                    ekf_initialized.value = True
                    ekf_running.value = True
                    waiting_for_init = False
                    print(
                        f"[EKF] Initialized from 2 acquired points with real dt={p2_data['time'] - p1_data['time']:.3f}s.")

                    # If we have a 3rd point, use it immediately as the first update
                    if k >= 3:
                        z3 = np.array([np.deg2rad(pb[10]), np.deg2rad(pb[11]), pb[12]])
                        R3 = create_measurement_noise_matrix(pb[13], 1)
                        # Make sure not to update with a timestamp from the past
                        if pb[14] > ekf.last_time:
                            ekf.update_with_angle_wrapping(z3, ekf.HJacobian, ekf.h, R3)
                            ekf.last_time = pb[14]
                            print("[EKF] Applied 3rd point as first update.")

                    continue

            # ... (rest of the tracking loop is fine) ...


# --- Helper function needs to be updated to accept the new data format ---
def init_ekf(ekf, initial_points):
    """Initialize EKF state using first two measurements."""
    meas1 = initial_points[0]
    meas2 = initial_points[1]

    x1, y1, z1 = spherical_to_cartesian(meas1['z'][0], meas1['z'][1], meas1['z'][2])
    x2, y2, z2 = spherical_to_cartesian(meas2['z'][0], meas2['z'][1], meas2['z'][2])

    # --- CRITICAL FIX: Calculate dt from REAL timestamps ---
    dt = meas2['time'] - meas1['time']
    if dt <= 0.01:  # Avoid division by zero or nonsensical values
        dt = 0.1
        print(f"[EKF WARN] Invalid dt in init ({dt:.4f}s), defaulting to {dt}s.")

    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    vz = (z2 - z1) / dt

    # Set initial state (use second measurement as starting position)
    ekf.x = np.array([x2, y2, z2, vx, vy, vz])


def run_manual_acquisition_sequence(pi, shared_data, movement_queue):
    """
    A simplified acquisition sequence for debug mode (e.g., tracking a hand).
    It assumes the user is already pointing the LiDAR at the target.
    """
    print("\n--- STARTING MANUAL ACQUISITION SEQUENCE (DEBUG MODE) ---")
    acquired_points = []

    # User should be pointing at the target already. We just need to refine.
    current_az = shared_data['stepper_degrees'].value
    current_el = shared_data['servo_degrees'].value

    print("[ACQUIRE-DBG] Acquiring first point...")
    point1 = _refine_target(pi, shared_data, movement_queue, current_az, current_el)
    if not point1: return False
    acquired_points.append(point1)

    print("[ACQUIRE-DBG] Acquiring second point after a short delay...")
    time.sleep(0.7)  # A longer delay for slower hand movements
    point2 = _refine_target(pi, shared_data, movement_queue, point1['az'], point1['el'])
    if not point2 or (point2['timestamp'] - point1['timestamp'] < 0.2): return False
    acquired_points.append(point2)

    print("[ACQUIRE-DBG] Acquiring third point...")
    time.sleep(0.7)
    point3 = _refine_target(pi, shared_data, movement_queue, point2['az'], point2['el'])
    if not point3: return False
    acquired_points.append(point3)

    # --- Handoff to EKF ---
    print("\n[ACQUIRE-DBG] Success! Populating buffer with 3 points for EKF.")
    points_buffer = shared_data["points_buffer"]
    with shared_data["points_count"].get_lock():
        for i, point in enumerate(acquired_points):
            base_idx = i * 5
            points_buffer[base_idx + 0] = point['az']
            points_buffer[base_idx + 1] = point['el']
            points_buffer[base_idx + 2] = point['distance_m']
            points_buffer[base_idx + 3] = point['strength']
            points_buffer[base_idx + 4] = point['timestamp']
        shared_data["points_count"].value = len(acquired_points)

    return True