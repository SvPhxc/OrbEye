# --- NEW FILE: analysis/plotter.py ---

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Helper function from your EKF file to convert spherical to cartesian
def spherical_to_cartesian(az_rad, el_rad, dist):
    """Converts spherical coordinates to Cartesian (x, y, z)."""
    x = dist * np.cos(el_rad) * np.cos(az_rad)
    y = dist * np.cos(el_rad) * np.sin(az_rad)
    z = dist * np.sin(el_rad)
    return x, y, z


def plot_ekf_vs_measured(history_measurements, history_estimates):
    """
    Generates a 3D plot comparing raw measurements against the EKF's filtered estimates.

    Args:
        history_measurements (list): A list of measurement dicts {'z': [az, el, dist], 'time': ts}.
        history_estimates (list): A list of the EKF state vectors [x, y, z, vx, vy, vz].
    """
    if not history_measurements or not history_estimates:
        print("[Plotter] No history to plot.")
        return

    print(
        f"[Plotter] Generating plot with {len(history_measurements)} measurements and {len(history_estimates)} EKF states.")

    # --- Process Data for Plotting ---

    # Convert raw measurements (spherical) to Cartesian coordinates
    measured_points = np.array([
        spherical_to_cartesian(m['z'][0], m['z'][1], m['z'][2]) for m in history_measurements
    ])

    # Extract just the position from the EKF state history
    estimated_points = np.array([state[0:3] for state in history_estimates])

    # --- Create the 3D Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot raw measurements as scattered blue dots
    ax.scatter(measured_points[:, 0], measured_points[:, 1], measured_points[:, 2],
               c='blue', marker='o', s=15, label='Raw Measurements', alpha=0.6)

    # Plot EKF filtered estimates as a smooth green line
    ax.plot(estimated_points[:, 0], estimated_points[:, 1], estimated_points[:, 2],
            color='green', linestyle='-', linewidth=2, label='EKF Estimated Path')

    # Plot start and end points for clarity
    ax.scatter(estimated_points[0, 0], estimated_points[0, 1], estimated_points[0, 2], c='cyan', s=100, label='Start',
               ec='black')
    ax.scatter(estimated_points[-1, 0], estimated_points[-1, 1], estimated_points[-1, 2], c='red', s=100, label='End',
               ec='black')

    ax.set_title('EKF Estimated Path vs. Raw LiDAR Measurements')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.grid(True)

    # Improve axis scaling to be more representative
    max_range = np.max(np.abs(estimated_points))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range * 1.5)  # Elevation is usually positive

    # Use a non-blocking show to avoid halting the main program on shutdown
    plt.show(block=False)
    print("[Plotter] Plot window opened. Close the window to continue program exit.")