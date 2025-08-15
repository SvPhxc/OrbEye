# tle_generator.py (Updated to remove tle-tailor dependency)

import numpy as np
import time
import traceback
from datetime import datetime, timezone

# --- New Dependencies: poliastro and astropy ---
# These libraries are well-maintained and standard for orbital mechanics in Python.
try:
    from astropy import units as u
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
except ImportError:
    print(
        "[TLE Generator] Error: 'poliastro' or 'astropy' not found. Please install them using 'pip install poliastro'")


    # Define dummy classes to prevent crashes if imports fail
    class u:
        pass


    class Earth:
        pass


    class Orbit:
        pass

# --- scikit-learn / sklearn ---
try:
    import sklearn
    from sklearn.decomposition import PCA
except ImportError:
    print("[TLE Generator] Error: 'scikit-learn' not found. Please install it using 'pip install scikit-learn'")


    class PCA:
        pass


    sklearn = None

import pyproj


# --- Helper functions for manual TLE string formatting ---

def tle_checksum(line):
    """Calculates the modulo-10 checksum for a TLE line."""
    total = 0
    for char in line:
        if char.isdigit():
            total += int(char)
        elif char == '-':
            total += 1
    return str(total % 10)


def format_tle(orbit, epoch, sat_num=99999, b_star=0.0001):
    """
    Manually formats an Orbit object from poliastro into a TLE string.

    Args:
        orbit (poliastro.twobody.Orbit): The orbit object.
        epoch (datetime): The epoch of the TLE.
        sat_num (int): The satellite catalog number.
        b_star (float): The B* drag term.

    Returns:
        str: A two-line string formatted as a TLE.
    """
    # --- Line 1 ---
    line1 = "1 "
    line1 += f"{sat_num:05d}U "
    line1 += "25001A   "  # Placeholder for International Designator

    # Epoch formatting (YYDDD.DDDDDDDD)
    year_short = epoch.strftime('%y')
    day_of_year = epoch.timetuple().tm_yday
    day_fraction = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second + epoch.microsecond / 1e6) / 86400.0
    epoch_str = f"{year_short}{day_of_year:03d}.{f'{day_fraction:.8f}'[2:]}"
    line1 += f"{epoch_str} "

    line1 += ".00000000 "  # First derivative of Mean Motion (placeholder)
    line1 += "00000-0 "  # Second derivative of Mean Motion (placeholder)

    # BSTAR Drag Term formatting
    b_star_sci = f"{b_star:.4e}".replace('e', '')
    base, exp = b_star_sci.split('-') if '-' in b_star_sci else b_star_sci.split('+')
    base_val = int(float(base) * 10000)
    exp_val = int(exp)
    line1 += f"{base_val:05d}{'-' if b_star < 0 else '+'}{exp_val} "

    line1 += "0 "  # Ephemeris type
    line1 += "999"  # Element set number

    line1 += tle_checksum(line1)

    # --- Line 2 ---
    line2 = "2 "
    line2 += f"{sat_num:05d} "

    # Orbital elements formatting
    inc = f"{orbit.inc.to_value(u.deg):8.4f}"
    raan = f"{orbit.raan.to_value(u.deg):8.4f}"
    ecc = f"{orbit.ecc.to_value(u.one):.7f}"[2:]  # Eccentricity without '0.'
    argp = f"{orbit.argp.to_value(u.deg):8.4f}"
    mean_anomaly = f"{orbit.M.to_value(u.deg):8.4f}"

    # Mean Motion in revolutions per day
    mean_motion_rev_day = orbit.n.to_value(u.rev / u.day)
    mean_motion = f"{mean_motion_rev_day:11.8f}"

    line2 += f"{inc} {raan} {ecc} {argp} {mean_anomaly} {mean_motion}"
    line2 += "00000"  # Revolution number at epoch (placeholder)

    line2 += tle_checksum(line2)

    return f"{line1}\n{line2}"


# --- Core Logic (mostly unchanged, except for the final TLE generation part) ---

def spherical_to_cartesian(az, el, dist):
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    x = dist * np.cos(el_rad) * np.sin(az_rad)
    y = dist * np.cos(el_rad) * np.cos(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def fit_3d_line_weighted(points, weights):
    if np.sum(weights) <= 1e-9:
        weights = np.ones(len(points))
    weights_normalized = weights / np.sum(weights)
    weighted_mean = np.average(points, axis=0, weights=weights_normalized)
    centered_points = points - weighted_mean
    weighted_centered_points = centered_points * np.sqrt(weights_normalized[:, np.newaxis])
    pca = PCA(n_components=1)
    pca.fit(weighted_centered_points)
    direction_vector = pca.components_[0]
    return weighted_mean, direction_vector


def lla_to_ecef(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return np.array([x, y, z])


def run_tle_generator(shared_data):
    """
    Main process loop for the TLE generator.
    """
    if 'Orbit' not in globals() or sklearn is None:
        print("[TLE Generator] Exiting due to missing critical libraries.")
        return

    print("[TLE Generator] Process started.")
    while not shared_data["shutdown"].value:
        try:
            if shared_data["generate_tle"].value:
                print("[TLE Generator] TLE generation triggered.")
                history_data = list(shared_data["tracking_history"])

                if len(history_data) < 2:
                    shared_data["generated_tle"].value = "Error: Not enough data points."
                    shared_data["generate_tle"].value = False
                    continue

                observations = np.array(history_data)
                points_enu = np.array([spherical_to_cartesian(az, el, dist) for az, el, dist, _, _ in observations])
                strengths = observations[:, 3]
                timestamps = observations[:, 4]
                point_on_line, direction_vector = fit_3d_line_weighted(points_enu, strengths)

                t0, t_end = timestamps[0], timestamps[-1]
                time_diff = t_end - t0
                start_point_local = point_on_line + np.dot(points_enu[0] - point_on_line,
                                                           direction_vector) * direction_vector
                end_point_local = point_on_line + np.dot(points_enu[-1] - point_on_line,
                                                         direction_vector) * direction_vector
                velocity_vector_local = (
                                                    end_point_local - start_point_local) / time_diff if time_diff > 1e-6 else np.zeros(
                    3)
                position_vector_local = start_point_local

                obs_lat, obs_lon, obs_alt = shared_data["observer_lat"].value, shared_data["observer_lon"].value, \
                shared_data["observer_alt"].value
                observer_ecef = lla_to_ecef(obs_lat, obs_lon, obs_alt)
                lat_rad, lon_rad = np.deg2rad(obs_lat), np.deg2rad(obs_lon)
                sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
                sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
                rotation_matrix = np.array([
                    [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                    [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                    [0, cos_lat, sin_lat]
                ])
                position_vector_ecef = observer_ecef + rotation_matrix.dot(position_vector_local)
                velocity_vector_ecef = rotation_matrix.dot(velocity_vector_local)

                # --- NEW TLE GENERATION LOGIC using poliastro ---
                epoch = datetime.fromtimestamp(t0, tz=timezone.utc)

                # poliastro requires inputs to have units from astropy
                r = position_vector_ecef * u.m
                v = velocity_vector_ecef * u.m / u.s

                try:
                    # Create the orbit object from the state vectors (position and velocity)
                    orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)

                    # Manually format the TLE string using our helper function
                    tle_string = format_tle(orbit, epoch)

                    print(f"[TLE Generator] TLE successfully generated for epoch {epoch.isoformat()}.")
                    print(tle_string)
                    shared_data["generated_tle"].value = tle_string

                except Exception as e:
                    error_msg = f"Error: TLE calculation failed. The trajectory may not be a valid orbit. Details: {e}"
                    print(f"[TLE Generator] {error_msg}")
                    shared_data["generated_tle"].value = error_msg
                finally:
                    shared_data["generate_tle"].value = False
                    shared_data["tracking_history"][:] = []
                    print("[TLE Generator] Ready for next trigger.")

            time.sleep(0.5)
        except Exception as e:
            print(f"[TLE Generator] An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            shared_data["generate_tle"].value = False
            time.sleep(2)

    print("[TLE Generator] Shutdown signal received. Terminating.")