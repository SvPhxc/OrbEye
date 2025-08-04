from sgp4.api import Satrec, jday
from datetime import timedelta
import numpy as np
import requests

# SGP4: for orbit propagation
from sgp4.conveniences import sat_epoch_datetime

from astropy.coordinates import ITRS, GCRS, EarthLocation
from astropy.time import Time
from astropy import units as u


def parse_tle_file(file_path):
    """
    Reads a TLE file and returns a list of dicts with Keplerian elements for each satellite.

    Returns:
        List of dicts like:
        {
            'name': 'SAT_NAME',
            'inclination_deg': float,
            'raan_deg': float,
            'eccentricity': float,
            'arg_perigee_deg': float,
            'mean_anomaly_deg': float,
            'mean_motion_rev_per_day': float
        }
    """
    kepler_elements = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()

        # Parse values from line 2
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float('0.' + line2[26:33].strip())
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])

        kepler_elements.append({
            'name': name,
            'inclination_deg': inclination,
            'raan_deg': raan,
            'eccentricity': eccentricity,
            'arg_perigee_deg': arg_perigee,
            'mean_anomaly_deg': mean_anomaly,
            'mean_motion_rev_per_day': mean_motion
        })

    return kepler_elements


def generate_orbit_xyz(tle_filename=None, tle_lines=None, duration_minutes=90, step_seconds=60):
    if tle_lines:
        tle_line1, tle_line2 = tle_lines
    elif tle_filename:
        with open(tle_filename, 'r') as f:
            lines = f.readlines()
            tle_line1 = lines[1].strip()
            tle_line2 = lines[2].strip()
    else:
        raise ValueError("Either tle_filename or tle_lines must be provided.")

    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    start_time = sat_epoch_datetime(satellite)

    num_steps = int((duration_minutes * 60) / step_seconds)
    xyz_positions = np.zeros((num_steps, 3))

    for i in range(num_steps):
        t = start_time + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)

        e, r, _ = satellite.sgp4(jd, fr)
        if e == 0:
            xyz_positions[i] = r
        else:
            xyz_positions[i] = [np.nan, np.nan, np.nan]

    return xyz_positions

def fetch_tle_by_name(satellite_name):
    """
    Fetches TLE data for a satellite from Celestrak using its name.
    Returns a tuple of (line1, line2)
    """
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    response = requests.get(url)
    lines = response.text.strip().splitlines()

    for i in range(0, len(lines) - 2, 3):
        name = lines[i].strip()
        if satellite_name.lower() in name.lower():
            return lines[i + 1].strip(), lines[i + 2].strip()
    
    raise ValueError(f"Satellite '{satellite_name}' not found.")


def get_sofia_eci(lat=42.7, lon=23.3, alt=0, time_utc=None):
    """
    Convert Sofia's ECEF position to ECI (GCRS) at a given time.
    """
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)
    obstime = Time(time_utc or Time.now())

    itrs = location.get_itrs(obstime=obstime)
    gcrs = itrs.transform_to(GCRS(obstime=obstime))
    return np.array([
    gcrs.cartesian.x.value,
    gcrs.cartesian.y.value,
    gcrs.cartesian.z.value
    ])