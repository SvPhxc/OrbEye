from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np


#test
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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


def generate_orbit_xyz(tle_filename='example.tle', duration_minutes=90, step_seconds=60):
    # Read TLE from file
    with open(tle_filename, 'r') as f:
        lines = f.readlines()
        tle_line1 = lines[1].strip()
        tle_line2 = lines[2].strip()
    
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    # Get the TLE epoch
    jd0 = satellite.jdsatepoch
    fr0 = satellite.jdsatepochF

    # How many points
    num_steps = int((duration_minutes * 60) / step_seconds)

    # Store XYZ positions
    xyz_positions = np.zeros((num_steps, 3))

    for i in range(num_steps):
        dt_minutes = (i * step_seconds) / 60.0
        jd = jd0
        fr = fr0 + dt_minutes / 1440.0  # 1 day = 1440 minutes

        e, r, _ = satellite.sgp4(jd, fr)
        xyz_positions[i] = r if e == 0 else [np.nan, np.nan, np.nan]

    return xyz_positions



def plot_orbit_with_earth(xyz):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot orbit path
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label="Satellite Orbit")

    # Draw Earth
    earth_radius = 6371  # km
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, cmap=cm.terrain, alpha=0.6)

    # Plot aesthetics
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit in ECI frame with Earth')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()