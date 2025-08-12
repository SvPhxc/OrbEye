import numpy as np
import time


def run_heatmap_tracker(shared_data):
    """
    Continuously scans a 3x3 grid around the predicted target position,
    generates a heatmap of LiDAR strength, and provides the hottest point
    as a measurement for the Kalman filter when in debug mode.
    """
    print("[HeatmapTracker] Starting...")
    grid_size = 3
    # Spacing in degrees between grid points
    grid_spacing_deg = 1.5
    heatmap = np.zeros((grid_size, grid_size))

    # Allow some time for other processes to initialize
    time.sleep(5)

    while not shared_data["shutdown"].value:
        try:
            if shared_data["debug_mode"].value:
                # Use the last predicted position from the EKF as the grid center
                center_az = shared_data["predicted_azimuth"].value
                center_el = shared_data["predicted_elevation"].value

                grid_points = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        az = center_az + (j - 1) * grid_spacing_deg
                        el = center_el + (i - 1) * grid_spacing_deg
                        grid_points.append({'az': az, 'el': el})

                for i in range(grid_size * grid_size):
                    target_az = grid_points[i]['az']
                    target_el = grid_points[i]['el']

                    shared_data["target_azimuth"].value = target_az
                    shared_data["target_elevation"].value = target_el
                    shared_data["go_to_target"].value = True

                    # Wait for the hardware to reach the target
                    start_time = time.time()
                    while time.time() - start_time < 1.0:  # 1 second timeout
                        if shared_data["target_reached"].value:
                            break
                        time.sleep(0.02)

                    time.sleep(0.05)  # Settle time for LiDAR reading
                    with shared_data["lidar_data"].get_lock():
                        dist_cm = shared_data["lidar_data"][0]
                        strength = shared_data["lidar_data"][1]

                    heatmap[i // grid_size, i % grid_size] = strength
                    grid_points[i]['dist'] = dist_cm / 100.0  # Convert to meters

                # Find the hottest point in the heatmap
                hot_spot_indices = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
                hottest_point_index = hot_spot_indices[0] * grid_size + hot_spot_indices[1]

                hottest_point = grid_points[hottest_point_index]

                # Pass the measurement to the EKF
                with shared_data["heatmap_measurement"].get_lock():
                    shared_data["heatmap_measurement"][0] = hottest_point['az']
                    shared_data["heatmap_measurement"][1] = hottest_point['el']
                    shared_data["heatmap_measurement"][2] = hottest_point['dist']
                    shared_data["heatmap_measurement_updated"].value = True
            else:
                time.sleep(0.5)

        except Exception as e:
            print(f"[HeatmapTracker] Error: {e}")
            time.sleep(1)

    print("[HeatmapTracker] Shutting down...")