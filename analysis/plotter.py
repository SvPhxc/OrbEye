# File: analysis/plotter.py

import matplotlib.pyplot as plt
import numpy as np

def spherical_to_cartesian_plot(az_rad, el_rad, dist):
    """Helper to convert spherical coordinates to Cartesian for plotting."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z

def plot_ekf_vs_measured(measurements, estimates):
    """
    Generates a 3D plot comparing measured points to the EKF track.
    """
    if not measurements or not estimates:
        print("[Plotter] No data available to plot.")
        return

    meas_xyz = np.array([spherical_to_cartesian_plot(m['z'][0], m['z'][1], m['z'][2]) for m in measurements])
    est_xyz = np.array([e[0:3] for e in estimates])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(meas_xyz[:, 0], meas_xyz[:, 1], meas_xyz[:, 2], c='blue', marker='o', label='Measured Points', s=15, alpha=0.6)
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], c='red', linestyle='-', label='EKF Estimated Track', linewidth=2.5)
    ax.scatter(est_xyz[0, 0], est_xyz[0, 1], est_xyz[0, 2], c='lime', marker='^', s=120, label='Track Start', edgecolors='k')
    ax.scatter(est_xyz[-1, 0], est_xyz[-1, 1], est_xyz[-1, 2], c='purple', marker='s', s=120, label='Track End', edgecolors='k')

    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
    ax.set_title('EKF Estimated Track vs. LiDAR Measurements')
    ax.legend(); ax.grid(True)
    ax.view_init(elev=25., azim=-50)
    plt.tight_layout()
    plt.show()