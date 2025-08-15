# tle_generator.py (Updated to use only astropy and numpy, no poliastro)

import numpy as np
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace  # Used to create a simple object for the TLE formatter

# --- Primary Dependencies: astropy and numpy ---
try:
    from astropy import units as u
    from astropy.time import Time
    from astropy.constants import G, M_earth
except ImportError:
    print("[TLE Generator] Error: 'astropy' not found. Please install it using 'pip install astropy'")


    # Define dummy classes to prevent crashes if imports fail
    class u:
        pass


    class Time:
        pass


    G, M_earth = None, None

# --- scikit-learn for line fitting ---
try:
    from sklearn.decomposition import PCA
except ImportError:
    print("[TLE Generator] Error: 'scikit-learn' not found. Please install it using 'pip install scikit-learn'")


    class PCA:
        pass

import pyproj

# --- Constant: Earth's standard gravitational parameter (mu) ---
# Using astropy's constants for high precision. mu = G * M
MU_EARTH = (G * M_earth).to_value(u.m ** 3 / u.s ** 2) if G else 3.986004418e14


# --- New Function: Manual Orbital Element Calculation ---

def state_vectors_to_tle_elements(r_vec, v_vec):
    """
    Converts state vectors (position and velocity) into classical orbital elements.
    This function replaces the core functionality of poliastro.Orbit.from_vectors.

    Args:
        r_vec (np.ndarray): Position vector [x, y, z] in meters (ECEF).
        v_vec (np.ndarray): Velocity vector [vx, vy, vz] in m/s (ECEF).

    Returns:
        SimpleNamespace: An object containing the six orbital elements needed for a TLE.
                         Returns None if the orbit cannot be calculated.
    """
    # Magnitudes of the state vectors
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    # 1. Angular Momentum Vector (h) and its magnitude
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # 2. Node Vector (n) - points to the ascending node
    k_hat = np.array([0, 0, 1])
    n_vec = np.cross(k_hat, h_vec)
    n_mag = np.linalg.norm(n_vec)

    # 3. Eccentricity Vector (e_vec) and Eccentricity (e)
    # This vector points towards the periapsis
    e_vec = ((v_mag ** 2 - MU_EARTH / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU_EARTH
    ecc = np.linalg.norm(e_vec)

    # 4. Specific Orbital Energy (xi) and Semi-major Axis (a)
    xi = (v_mag ** 2 / 2) - (MU_EARTH / r_mag)
    if abs(xi) < 1e-9:  # Parabolic orbit, not supported by TLE
        return None
    a = -MU_EARTH / (2 * xi)
    if a < 0:  # Hyperbolic orbit, not supported by TLE
        return None

    # 5. Inclination (i)
    inc = np.arccos(h_vec[2] / h_mag)

    # 6. Right Ascension of the Ascending Node (RAAN or Omega)
    # Handle equatorial case where n_mag is zero
    if n_mag < 1e-9:
        raan = 0.0  # Conventionally set to 0 for equatorial orbits
    else:
        raan = np.arccos(n_vec[0] / n_mag)
        if n_vec[1] < 0:  # If n_y is negative, RAAN is in the 3rd or 4th quadrant
            raan = 2 * np.pi - raan

    # 7. Argument of Perigee (argp or omega)
    # Handle circular case where ecc is zero
    if ecc < 1e-9:
        argp = 0.0  # Undefined for circular orbits, conventionally set to 0
    else:
        argp = np.arccos(np.dot(n_vec, e_vec) / (n_mag * ecc))
        if e_vec[2] < 0:  # If e_z is negative, perigee is below the equatorial plane
            argp = 2 * np.pi - argp

    # 8. True Anomaly (nu)
    if ecc < 1e-9:
        # For circular orbits, use Argument of Latitude instead
        # This is the angle from the ascending node to the satellite
        cos_u = np.dot(n_vec, r_vec) / (n_mag * r_mag)
        true_anomaly = np.arccos(np.clip(cos_u, -1, 1))
        if r_vec[2] < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        # For eccentric orbits, it's the angle from perigee
        cos_nu = np.dot(e_vec, r_vec) / (ecc * r_mag)
        true_anomaly = np.arccos(np.clip(cos_nu, -1, 1))
    # Check if satellite is moving away from perigee (radial velocity is positive)
    if np.dot(r_vec, v_vec) < 0:
        true_anomaly = 2 * np.pi - true_anomaly

    # 9. Convert True Anomaly to Mean Anomaly (M) for TLE
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(true_anomaly / 2))  # Eccentric Anomaly
    mean_anomaly = E - ecc * np.sin(E)  # Mean Anomaly (from Kepler's equation)

    # 10. Mean Motion (n) in revolutions per day
    mean_motion_rad_s = np.sqrt(MU_EARTH / a ** 3)
    mean_motion_rev_day = mean_motion_rad_s * (86400 / (2 * np.pi))

    # Return results in a simple object, converting radians to degrees
    return SimpleNamespace(
        inc=np.rad2deg(inc),
        raan=np.rad2deg(raan),
        ecc=ecc,
        argp=np.rad2deg(argp),
        mean_anomaly=np.rad2deg(mean_anomaly % (2 * np.pi)),  # Ensure it's 0-360
        mean_motion=mean_motion_rev_day
    )


# --- TLE Formatting Helpers (Unchanged) ---
def tle_checksum(line):
    total = sum(int(c) for c in line if c.isdigit()) + line.count('-')
    return str(total % 10)


def format_tle(elements, epoch, sat_num=99999, b_star=0.0001):
    line1 = f"1 {sat_num:05d}U 25001A   "
    year_short = epoch.strftime('%y')
    day_of_year = epoch.timetuple().tm_yday
    day_fraction = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second + epoch.microsecond / 1e6) / 86400.0
    epoch_str = f"{year_short}{day_of_year:03d}.{f'{day_fraction:.8f}'[2:]}"
    line1 += f"{epoch_str} .00000000 00000-0 "
    b_star_sci = f"{b_star:.4e}".replace('e', '')
    base, exp = (b_star_sci.split('-') if '-' in b_star_sci else b_star_sci.split('+'))
    line1 += f"{int(float(base) * 10000):05d}{'-' if b_star < 0 else '+'}{int(exp)} 0 999"
    line1 += tle_checksum(line1)

    line2 = f"2 {sat_num:05d} "
    line2 += f"{elements.inc:8.4f} {elements.raan:8.4f} {f'{elements.ecc:.7f}'[2:]} "
    line2 += f"{elements.argp:8.4f} {elements.mean_anomaly:8.4f} {elements.mean_motion:11.8f}00000"
    line2 += tle_checksum(line2)
    return f"{line1}\n{line2}"


# --- Core Logic (Calls the new calculation function) ---
def spherical_to_cartesian(az, el, dist):
    az_rad, el_rad = np.deg2rad(az), np.deg2rad(el)
    x = dist * np.cos(el_rad) * np.sin(az_rad)
    y = dist * np.cos(el_rad) * np.cos(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def fit_3d_line_weighted(points, weights):
    if np.sum(weights) <= 1e-9: weights = np.ones(len(points))
    weights_normalized = weights / np.sum(weights)
    weighted_mean = np.average(points, axis=0, weights=weights_normalized)
    centered_points = points - weighted_mean
    weighted_centered_points = centered_points * np.sqrt(weights_normalized[:, np.newaxis])
    pca = PCA(n_components=1)
    pca.fit(weighted_centered_points)
    return weighted_mean, pca.components_[0]


def lla_to_ecef(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return np.array([x, y, z])


def run_tle_generator(shared_data):
    if G is None:
        print("[TLE Generator] Exiting due to missing astropy library.")
        return

    print("[TLE Generator] Process started. Using astropy for calculations.")
    while not shared_data["shutdown"].value:
        try:
            if shared_data["generate_tle"].value:
                print("[TLE Generator] TLE generation triggered.")
                history_data = list(shared_data["tracking_history"])

                if len(history_data) < 2:
                    shared_data["generated_tle"].value = "Error: Not enough data points."
                    shared_data["generate_tle"].value = False
                    continue

                # --- State vector calculation (same as before) ---
                observations = np.array(history_data)
                points_enu = np.array([spherical_to_cartesian(az, el, dist) for az, el, dist, _, _ in observations])
                strengths, timestamps = observations[:, 3], observations[:, 4]
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
                sin_lat, cos_lat, sin_lon, cos_lon = np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)
                rotation_matrix = np.array([
                    [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                    [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                    [0, cos_lat, sin_lat]
                ])
                position_vector_ecef = observer_ecef + rotation_matrix.dot(position_vector_local)
                velocity_vector_ecef = rotation_matrix.dot(velocity_vector_local)

                # --- NEW TLE GENERATION LOGIC using our manual function ---
                epoch = datetime.fromtimestamp(t0, tz=timezone.utc)

                try:
                    # Convert state vectors to orbital elements using our new function
                    orbital_elements = state_vectors_to_tle_elements(position_vector_ecef, velocity_vector_ecef)

                    if orbital_elements is None:
                        raise ValueError(
                            "The calculated trajectory is not a stable orbit (e.g., hyperbolic or parabolic).")

                    # Manually format the TLE string
                    tle_string = format_tle(orbital_elements, epoch)

                    print(f"[TLE Generator] TLE successfully generated for epoch {epoch.isoformat()}.")
                    print(tle_string)
                    shared_data["generated_tle"].value = tle_string

                except Exception as e:
                    error_msg = f"Error: TLE calculation failed. Details: {e}"
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