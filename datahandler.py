from datetime import timedelta, datetime, timezone
import numpy as np
import requests
from math import sin, cos, sqrt

# SGP4: for orbit propagation
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime

from astropy.coordinates import TEME, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
import astropy.units as u
from math import sin, cos, sqrt

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


# --- WGS-84 constants
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)

def _rot_ecef_to_enu(lat_rad, lon_rad):
    slat, clat = sin(lat_rad), cos(lat_rad)
    slon, clon = sin(lon_rad), cos(lon_rad)
    return np.array([
        [-slon,          clon,           0.0],     # East
        [-slat*clon,    -slat*slon,      clat],    # North
        [ clat*clon,     clat*slon,      slat],    # Up
    ], dtype=float)


def _teme_to_itrs_m(r_teme_km, t_dt_utc):
    """TEME (km) -> ITRS/ECEF (meters) using Astropy, robust to units."""
    r = np.asarray(r_teme_km, dtype=float)  # ensure plain floats
    t = Time(t_dt_utc, scale="utc")
    cr = CartesianRepresentation(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km)
    itrs = TEME(cr, obstime=t).transform_to(ITRS(obstime=t))
    return itrs.cartesian.xyz.to_value(u.m)  # returns plain floats (meters)

# ---------- Main function ----------
def generate_orbit_xyz(
    tle_filename=None,
    tle_lines=None,
    duration_minutes=90,
    step_seconds=60,
    *,
    site_lat_deg=None,       # if provided -> return ENU [E,N,U]
    site_lon_deg=None,
    site_h_m=0.0,
    units="km"               # used for ENU only: "m" or "km"
):
    """
    If site_lat/lon are None: return TEME positions [km], shape (N,3).
    If site_lat/lon are given: return ENU positions [E,N,U] in meters or km, shape (N,3).
    """
    # --- TLE input (3-line block: name, L1, L2)
    if tle_lines:
        tle_line1, tle_line2 = tle_lines
    elif tle_filename:
        with open(tle_filename, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 3:
            raise ValueError("TLE file must contain: name, line1, line2.")
        tle_line1, tle_line2 = lines[1], lines[2]
    else:
        raise ValueError("Either tle_filename or tle_lines must be provided.")

    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    start_time = sat_epoch_datetime(sat)  # UTC datetime at TLE epoch

    num_steps = int((duration_minutes * 60) / step_seconds)
    xyz_km = np.full((num_steps, 3), np.nan, dtype=float)

    # ENU prep if requested
    want_enu = (site_lat_deg is not None and site_lon_deg is not None)
    if want_enu:
        site = EarthLocation.from_geodetic(lon=site_lon_deg * u.deg,
                                        lat=site_lat_deg * u.deg,
                                        height=site_h_m * u.m)
        site_ecef_m = np.array([
            site.x.to_value(u.m),
            site.y.to_value(u.m),
            site.z.to_value(u.m)
        ], dtype=float)

        lat_rad = np.deg2rad(site_lat_deg)
        lon_rad = np.deg2rad(site_lon_deg)
        R_e2n = _rot_ecef_to_enu(lat_rad, lon_rad)
        enu_m = np.full((num_steps, 3), np.nan, dtype=float)

    # Propagate and convert
    for i in range(num_steps):
        t = start_time + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond * 1e-6)

        e, r_km, _ = sat.sgp4(jd, fr)
        if e == 0:
            r_km = np.array(r_km, dtype=float)
            xyz_km[i] = r_km
            if want_enu:
                r_ecef_m = _teme_to_itrs_m(r_km, t)   # plain floats in meters
                rho_ecef = r_ecef_m - site_ecef_m
                enu_m[i] = R_e2n @ rho_ecef

    if want_enu:
        if units.lower() == "km":
            return enu_m / 1000.0
        elif units.lower() == "m":
            return enu_m
        else:
            raise ValueError('units must be "m" or "km".')
    else:
        return xyz_km

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