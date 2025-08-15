# tle_generator.py

import numpy as np
import time
import traceback
from datetime import datetime, timezone
from sklearn.decomposition import PCA
import pyproj

try:
    # tle-tailor is used for the final conversion from a state vector to a TLE
    from tle_tailor import StateVector, TLE
except ImportError:
    print("[TLE Generator] Error: 'tle_tailor' library not found. Please install it using 'pip install tle-tailor'")
    TLE = None  # Set to None to handle import error gracefully


def spherical_to_cartesian(az, el, dist):
    """
    Converts spherical coordinates (azimuth, elevation, distance) to a local
    Cartesian coordinate system (East, North, Up).
    Azimuth is in degrees (0=North, 90=East), Elevation is in degrees (0=Horizon).
    Distance is in meters.
    Returns (x, y, z) in meters.
    """
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    x = dist * np.cos(el_rad) * np.sin(az_rad)  # East
    y = dist * np.cos(el_rad) * np.cos(az_rad)  # North
    z = dist * np.sin(el_rad)  # Up
    return x, y, z


def fit_3d_line_weighted(points, weights):
    """
    Fits a 3D line to a set of points using weighted Principal Component Analysis (PCA).
    The direction of the line is the first principal component.
    """
    if np.sum(weights) == 0:  # Avoid division by zero if all weights are zero
        weights = np.ones_like(weights)

    # Ensure weights are normalized and shaped correctly for broadcasting
    weights_normalized = weights / np.sum(weights)
    weights_col = weights_normalized[:, np.newaxis]

    # Calculate the weighted mean of the points
    weighted_mean = np.sum(points * weights_col, axis=0)

    # Center the points around the weighted mean
    centered_points = points - weighted_mean

    # Apply weights to the centered points for PCA
    weighted_centered_points = centered_points * np.sqrt(weights_col)

    pca = PCA(n_components=1)
    pca.fit(weighted_centered_points)

    direction_vector = pca.components_[0]
    return weighted_mean, direction_vector


def lla_to_ecef(lat, lon, alt):
    """Converts Latitude, Longitude, Altitude to Earth-Centered, Earth-Fixed (ECEF) coordinates."""
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return np.array([x, y, z])


def run_tle_generator(shared_data):
    """
    This function runs in a loop, waiting for the 'generate_tle' flag.
    When triggered, it processes the data from 'tracking_history' to create a TLE.
    """
    if TLE is None:
        print("[TLE Generator] Exiting due to missing 'tle_tailor' library.")
        return

    print("[TLE Generator] Process started.")
    while not shared_data["shutdown"].value:
        try:
            if shared_data["generate_tle"].value:
                print("[TLE Generator] TLE generation triggered.")

                # --- 1. Collect Data ---
                # Copy data from the shared list to prevent race conditions during calculation
                history_data = list(shared_data["tracking_history"])

                if len(history_data) < 2:
                    print("[TLE Generator] Warning: Not enough points in history (need at least 2).")
                    shared_data["generated_tle"].value = "Error: Not enough data points."
                    shared_data["generate_tle"].value = False
                    time.sleep(1)
                    continue

                # Unpack into a NumPy array: [az, el, dist, strength, timestamp]
                observations = np.array(history_data)

                # --- 2. Refine Trajectory ---
                points_enu = np.array([spherical_to_cartesian(az, el, dist) for az, el, dist, _, _ in observations])
                strengths = observations[:, 3]
                timestamps = observations[:, 4]

                point_on_line, direction_vector = fit_3d_line_weighted(points_enu, strengths)

                # --- 3. Calculate State Vector (Position & Velocity) in local ENU frame ---
                t0 = timestamps[0]
                t_end = timestamps[-1]
                time_diff = t_end - t0

                # Project the first and last points onto the line for a stable velocity
                start_point_local = point_on_line + np.dot(points_enu[0] - point_on_line,
                                                           direction_vector) * direction_vector
                end_point_local = point_on_line + np.dot(points_enu[-1] - point_on_line,
                                                         direction_vector) * direction_vector

                if time_diff <= 1e-6:  # Avoid division by zero for simultaneous points
                    velocity_vector_local = np.zeros(3)
                else:
                    velocity_vector_local = (end_point_local - start_point_local) / time_diff

                position_vector_local = start_point_local

                # --- 4. Convert State Vector to ECEF Frame ---
                observer_ecef = lla_to_ecef(
                    shared_data["observer_lat"].value,
                    shared_data["observer_lon"].value,
                    shared_data["observer_alt"].value
                )

                # Create rotation matrix for converting local ENU vectors to ECEF
                lat_rad = np.deg2rad(shared_data["observer_lat"].value)
                lon_rad = np.deg2rad(shared_data["observer_lon"].value)
                sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
                sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

                rotation_matrix = np.array([
                    [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                    [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                    [0, cos_lat, sin_lat]
                ])

                # Rotate and translate to get the final ECEF state vector
                position_vector_ecef = observer_ecef + rotation_matrix.dot(position_vector_local)
                velocity_vector_ecef = rotation_matrix.dot(velocity_vector_local)

                # --- 5. Generate TLE ---
                epoch = datetime.fromtimestamp(t0, tz=timezone.utc)

                state_vector = StateVector(
                    rx=position_vector_ecef[0], ry=position_vector_ecef[1], rz=position_vector_ecef[2],
                    vx=velocity_vector_ecef[0], vy=velocity_vector_ecef[1], vz=velocity_vector_ecef[2],
                    epoch=epoch
                )

                try:
                    # B-star (drag term) and satellite number are placeholders.
                    tle = TLE.from_state_vector(state_vector, b_star_drag=0.0001, sat_num=99999)
                    tle_string = str(tle)
                    print(f"[TLE Generator] TLE successfully generated for epoch {epoch}.")
                    print(tle_string)
                    shared_data["generated_tle"].value = tle_string
                except Exception as e:
                    error_msg = f"Error: TLE generation failed. The trajectory may not be a valid orbit. Details: {e}"
                    print(f"[TLE Generator] {error_msg}")
                    shared_data["generated_tle"].value = error_msg
                finally:
                    # Reset trigger and clear history for the next run
                    shared_data["generate_tle"].value = False
                    shared_data["tracking_history"][:] = []  # Clear the shared list
                    print("[TLE Generator] Ready for next trigger.")

            time.sleep(0.5)  # Check for the trigger flag every 500ms
        except Exception as e:
            print(f"[TLE Generator] An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            shared_data["generate_tle"].value = False  # Reset trigger on error
            time.sleep(2)

    print("[TLE Generator] Shutdown signal received. Terminating.")