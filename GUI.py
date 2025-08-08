import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time
import os
from astropy.time import Time

from motors.motor_controller import *

from datahandler import (
    parse_tle_file,
    generate_orbit_xyz,
    fetch_tle_by_name,
    get_sofia_eci
)


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data, movement_queue):
        super().__init__()
        self.setWindowTitle("LockedIn Martin")
        self.resize(1300, 600)
        self.debug_scale = 0.1
        self.orbit_items = []  # Store all orbit and vector items

        global _shared_data
        self.shared_data = _shared_data

        # Central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # === Left: 3D View ===
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=20)
        main_layout.addWidget(self.view, stretch=3)

        grid = gl.GLGridItem()
        self.view.addItem(grid)

        self.background_plot = gl.GLScatterPlotItem(size=5, color=(0.5, 0.5, 1, 0.5))
        self.view.addItem(self.background_plot)

        # Satellite object
        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.satellite)

        # Laser beam
        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(self.laser)

        # SLR Station marker
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5)
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded')
        self.view.addItem(self.station)

        # Orbit animation params
        self.radius = 8
        self.orbit_speed = 0.2
        self.elevation_deg = 30
        self.start_time = time.time()
        self.orbit_enabled = True

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_laser_from_pan_tilt)
        self.timer.start(30)  # ~33 fps is plenty

        # === Right: Controls ===
        controls = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls, stretch=1)

        target_controls = QtWidgets.QVBoxLayout()

        # Target Azimuth input
        target_controls.addWidget(QtWidgets.QLabel("Target Azimuth (°)"))
        self.az_input = QtWidgets.QLineEdit()
        self.az_input.setPlaceholderText("0–360")
        target_controls.addWidget(self.az_input)

        # Target Elevation input
        target_controls.addWidget(QtWidgets.QLabel("Target Elevation (°)"))
        self.el_input = QtWidgets.QLineEdit()
        self.el_input.setPlaceholderText("0–90")
        target_controls.addWidget(self.el_input)

        # Go Button
        self.btn_go = QtWidgets.QPushButton("Go")
        self.btn_go.clicked.connect(self.on_go_clicked)
        target_controls.addWidget(self.btn_go)

        # Add Background Button
        self.btn_show_background = QtWidgets.QPushButton("Show/Hide Background")
        self.btn_show_background.clicked.connect(self.toggle_background_plot)
        target_controls.addWidget(self.btn_show_background)

        self.btn_acquire = QtWidgets.QPushButton("Acquire (3 pts)")
        self.btn_acquire.clicked.connect(self.accuire_points)
        controls.addWidget(self.btn_acquire)

        self.btn_stop_ekf = QtWidgets.QPushButton("Stop EKF")
        self.btn_stop_ekf.clicked.connect(self.stopEKF)
        controls.addWidget(self.btn_stop_ekf)

        self.chk_track_pred = QtWidgets.QCheckBox("Track Prediction")
        self.chk_track_pred.setChecked(True)
        self.chk_track_pred.toggled.connect(self.on_track_pred_toggled)
        controls.addWidget(self.chk_track_pred)

        # Spacer to push widgets to the top
        target_controls.addStretch()

        # Add this new column to the main layout
        main_layout.addLayout(target_controls, stretch=1)

        # Satellite name input
        self.sat_name_input = QtWidgets.QLineEdit()
        self.sat_name_input.setPlaceholderText("ISS (ZARYA)")
        controls.addWidget(self.sat_name_input)

        # Fetch + Plot button
        self.btn_fetch_plot = QtWidgets.QPushButton("Fetch & Plot Satellite")
        self.btn_fetch_plot.clicked.connect(self.fetch_and_plot_satellite)
        controls.addWidget(self.btn_fetch_plot)

        # Remove orbit button
        self.btn_remove_orbit = QtWidgets.QPushButton("Remove Orbit")
        self.btn_remove_orbit.clicked.connect(self.remove_orbit)
        controls.addWidget(self.btn_remove_orbit)

        # Add Sphere Button
        self.btn_add_sphere = QtWidgets.QPushButton("Add Sphere at Drone")
        self.btn_add_sphere.clicked.connect(self.add_sphere)
        controls.addWidget(self.btn_add_sphere)

        # Toggle Background-Scan Button
        self.btn_background = QtWidgets.QPushButton("Background Scan")
        self.btn_background.clicked.connect(self.background_scan)
        controls.addWidget(self.btn_background)

        # Elevation angle slider
        controls.addWidget(QtWidgets.QLabel("Elevation Angle (°)"))
        self.slider_elevation = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_elevation.setMinimum(0)
        self.slider_elevation.setMaximum(90)
        self.slider_elevation.setValue(self.elevation_deg)
        self.slider_elevation.valueChanged.connect(self.set_elevation_angle)
        controls.addWidget(self.slider_elevation)


        # LCD for elevation angle
        self.lcd_elevation = QtWidgets.QLCDNumber()
        self.lcd_elevation.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_elevation.setDigitCount(4)
        self.lcd_elevation.display(self.elevation_deg)
        controls.addWidget(self.lcd_elevation)

        # LiDAR range
        controls.addWidget(QtWidgets.QLabel("LiDAR Range (cm)"))
        self.lcd_range = QtWidgets.QLCDNumber()
        self.lcd_range.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_range.setDigitCount(5)
        controls.addWidget(self.lcd_range)

        # LiDAR strength
        controls.addWidget(QtWidgets.QLabel("LiDAR Strength"))
        self.lcd_strength = QtWidgets.QLCDNumber()
        self.lcd_strength.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_strength.setDigitCount(5)
        controls.addWidget(self.lcd_strength)

        # Pan angle display
        controls.addWidget(QtWidgets.QLabel("Pan Angle (°)"))
        self.lcd_pan = QtWidgets.QLCDNumber()
        self.lcd_pan.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_pan.setDigitCount(6)
        controls.addWidget(self.lcd_pan)

        # Tilt angle display
        controls.addWidget(QtWidgets.QLabel("Tilt Angle (°)"))
        self.lcd_tilt = QtWidgets.QLCDNumber()
        self.lcd_tilt.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_tilt.setDigitCount(6)
        controls.addWidget(self.lcd_tilt)

        # Shutdown button
        self.btn_shutdown = QtWidgets.QPushButton("Shutdown")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        controls.addWidget(self.btn_shutdown)

        self.lidar_timer = QtCore.QTimer()
        self.lidar_timer.timeout.connect(self.update_lidar_display)
        self.lidar_timer.start(10)  

        self.angle_timer = QtCore.QTimer()
        self.angle_timer.timeout.connect(self.update_angle_display)
        self.angle_timer.start(100)  # Update every 100 ms

        # === Pan/Tilt Controls ===
        controls.addWidget(QtWidgets.QLabel("Pan/Tilt Controls"))
        dpad_layout = QtWidgets.QGridLayout()

        # Create buttons
        btn_up = QtWidgets.QPushButton("↑")
        btn_down = QtWidgets.QPushButton("↓")
        btn_left = QtWidgets.QPushButton("←")
        btn_right = QtWidgets.QPushButton("→")

        # Add buttons to grid (row, col)
        dpad_layout.addWidget(btn_up, 0, 1)
        dpad_layout.addWidget(btn_left, 1, 0)
        dpad_layout.addWidget(btn_right, 1, 2)
        dpad_layout.addWidget(btn_down, 2, 1)

        # Optional: Center empty widgets to keep spacing
        dpad_layout.addItem(QtWidgets.QSpacerItem(0, 0), 0, 0)
        dpad_layout.addItem(QtWidgets.QSpacerItem(0, 0), 0, 2)
        dpad_layout.addItem(QtWidgets.QSpacerItem(0, 0), 2, 0)
        dpad_layout.addItem(QtWidgets.QSpacerItem(0, 0), 2, 2)


        # Add to main control panel
        controls.addLayout(dpad_layout)

        btn_up.clicked.connect(self.set_tilt_up)
        btn_down.clicked.connect(self.set_tilt_down)
        btn_left.clicked.connect(self.set_pan_left)
        btn_right.clicked.connect(self.set_pan_right)

        controls.addStretch()

        
    

    def fetch_and_plot_satellite(self):
        name = self.sat_name_input.text().strip()
        if not name:
            print("No satellite name entered.")
            return

        try:
            # Fetch TLE and parse elements
            tle_lines = fetch_tle_by_name(name)

            with open("temp.tle", "w") as f:
                f.write(f"{name}\n{tle_lines[0]}\n{tle_lines[1]}\n")

            elements = parse_tle_file("temp.tle")[0]

            # Propagate orbit
            self.orbit_xyz = generate_orbit_xyz(tle_lines=tle_lines, duration_minutes=90)

            # Clear old visuals
            self.remove_orbit()

            # Plot orbit and reference vectors
            self.plot_orbit_line()
            self.plot_keplerian_reference(
                inclination_deg=elements['inclination_deg'],
                raan_deg=elements['raan_deg'],
                arg_perigee_deg=elements['arg_perigee_deg'],
                length=2000.0
            )
            print(f"Plotted orbit and frame for '{name}'")
        except Exception as e:
            print(f"Error: {e}")
        # Plot line to Sofia
        sofia_pos = get_sofia_eci(time_utc=Time.now())  # current ECI position
        sofia_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], sofia_pos * self.debug_scale]),
                                    color=(0, 1, 1, 1), width=2)
        self.view.addItem(sofia_line)
        self.orbit_items.append(sofia_line)

    def accuire_points(self):
        self.shared_data["acquire_points"].value = True

    def stopEKF(self):
        self.shared_data["ekf_running"].value = False

    def on_track_pred_toggled(self, enabled):
        # optional: mirror this into shared_data if you want to disable tracking without stopping EKF
        self.shared_data["ekf_running"].value = bool(enabled)

    def remove_orbit(self):
        for item in self.orbit_items:
            self.view.removeItem(item)
        self.orbit_items.clear()

    def plot_keplerian_reference(self, inclination_deg, raan_deg, arg_perigee_deg, length=5.0):
        inc = np.radians(inclination_deg)
        raan = np.radians(raan_deg)
        argp = np.radians(arg_perigee_deg)

        # Z-axis
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, length]]),
                                   color=(0, 0, 1, 1), width=2)
        self.view.addItem(z_axis)
        self.orbit_items.append(z_axis)

        # X and Y
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [length, 0, 0]]),
                                   color=(1, 0, 0, 1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, length, 0]]),
                                   color=(0, 1, 0, 1), width=2)
        self.view.addItem(x_axis)
        self.view.addItem(y_axis)
        self.orbit_items.extend([x_axis, y_axis])

        # RAAN vector
        raan_vec = np.array([np.cos(raan), np.sin(raan), 0]) * length
        raan_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], raan_vec]),
                                      color=(1, 1, 0, 1), width=2)
        self.view.addItem(raan_line)
        self.orbit_items.append(raan_line)

        # Perigee vector in orbital plane
        perigee_dir = np.array([np.cos(argp), np.sin(argp), 0])
        R_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan),  np.cos(raan), 0],
            [0, 0, 1]
        ])
        R_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc),  np.cos(inc)]
        ])
        perigee_world = R_raan @ (R_inc @ perigee_dir) * length
        perigee_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], perigee_world]),
                                         color=(1, 0, 1, 1), width=2)
        self.view.addItem(perigee_line)
        self.orbit_items.append(perigee_line)

    def set_elevation_angle(self, value):
        self.elevation_deg = value
        self.lcd_elevation.display(value)
    
    def on_go_clicked(self):
        try:
            az = float(self.az_input.text())
            el = float(self.el_input.text())

            self.shared_data["target_azimuth"].value = az
            self.shared_data["target_elevation"].value = el
            self.shared_data["go_to_target"].value = True  # trigger!

        except ValueError:
            print("Invalid input: please enter numeric values")


    def toggle_background_plot(self):
        """ --- NEW: Loads data from file and displays/hides the plot --- """
        if self.background_plot.visible():
            self.background_plot.hide()
            print("[GUI] Background visualization hidden.")
            return
            
        try:
            # Load the reshaped data [elevation, azimuth, [strength, range]]
            bg_data = np.load("background_data.npy")
            
            points = []
            # Iterate through the array to convert spherical to Cartesian
            for el_idx, az_row in enumerate(bg_data):
                for az_idx, reading in enumerate(az_row):
                    strength, dist_cm = reading
                    # Plot only valid points within a reasonable range
                    if 10 < dist_cm < 1200:
                        # Convert to radians for math
                        el_rad = np.radians(el_idx)
                        az_rad = np.radians(az_idx)
                        dist_m = dist_cm / 10.0  #Scale to meters for visualization

                        # Spherical to Cartesian conversion
                        x = dist_m * np.cos(el_rad) * np.cos(az_rad)
                        y = - dist_m * np.cos(el_rad) * np.sin(az_rad)
                        z = dist_m * np.sin(el_rad)
                        points.append([x, y, z])

            if points:
                print(f"[GUI] Plotting {len(points)} background points.")
                self.background_plot.setData(pos=np.array(points))
                self.background_plot.show()
            else:
                print("[GUI] No valid points found in background data file.")

        except FileNotFoundError:
            print("[GUI] Error: 'background_data.npy' not found. Please run a scan first.")
        except Exception as e:
            print(f"[GUI] Error loading background data: {e}")
            
    def update_lidar_display(self):
        lidar_data = self.shared_data.get("lidar_data")
        if lidar_data is not None:
            distance = lidar_data[0]
            strength = lidar_data[1]
            self.lcd_range.display(distance)
            self.lcd_strength.display(strength)
            #print(f"[GUI] LiDAR: {distance} cm | Strength: {strength}")

    def set_tilt_up(self):
        self.shared_data['tilt_up'].value = True

    def set_tilt_down(self):
        self.shared_data['tilt_down'].value = True

    def set_pan_left(self):
        self.shared_data['pan_left'].value = True

    def set_pan_right(self):
        self.shared_data['pan_right'].value = True

    def plot_orbit_line(self):
        valid_points = self.orbit_xyz[~np.isnan(self.orbit_xyz).any(axis=1)]
        scaled_points = valid_points * self.debug_scale
        orbit_line = gl.GLLinePlotItem(pos=scaled_points, color=(1, 1, 0, 1), width=2, antialias=True, mode='line_strip')
        self.view.addItem(orbit_line)
        self.orbit_items.append(orbit_line)

    def background_scan(self):
        print("[GUI] Triggering background scan")
        self.shared_data["scan_trigger"].value = True
    
    def update_angle_display(self):
        self.lcd_pan.display(self.shared_data['stepper_degrees'].value)
        self.lcd_tilt.display(self.shared_data['servo_degrees'].value)

    def update_laser_from_pan_tilt(self):
        """
        Point the laser along the current pan/tilt. 
        Length uses live LiDAR distance when plausible, else a default.
        """
        try:
            # Read current mount angles (degrees) and LiDAR in cm
            az = float(self.shared_data['stepper_degrees'].value)  # 0..360
            el = float(self.shared_data['servo_degrees'].value)    # 0..90

            lidar = self.shared_data.get('lidar_data')
            if lidar is not None:
                dist_cm = float(lidar[0])
            else:
                dist_cm = 0.0

            # Decide laser length in meters
            if 50.0 <= dist_cm <= 2000.0:     # sane-ish reading window (0.5–20 m)
                length_m = dist_cm / 100.0
            else:
                length_m = 10.0               # fallback if no LiDAR yet

            # Scale to your scene units
            length = max(5.0 * self.debug_scale, length_m * self.debug_scale) * 10


            # Convert angles to a unit direction vector
            az_rad = np.radians(az % 360.0)
            el_rad = np.radians(np.clip(el, 0.0, 90.0))
            x = np.cos(el_rad) * np.cos(az_rad)
            y = np.cos(el_rad) * np.sin(az_rad)
            z = np.sin(el_rad)

            tip = np.array([x * length, y * length, z * length], dtype=float)

            # Update the line and (optionally) place the 'satellite' at the tip
            self.laser.setData(pos=np.vstack((np.zeros(3), tip)))
            self.satellite.setData(pos=tip.reshape(1, 3))  # keeps add_sphere() working

        except Exception as e:
            print(f"[GUI] update_laser_from_pan_tilt error: {e}")

    def add_sphere(self):
        md = gl.MeshData.sphere(rows=5, cols=10, radius=1)
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 1), shader='shaded')
        x, y, z = self.satellite.pos[0]
        sphere.translate(x, y, z)
        self.view.addItem(sphere)
        print("Sphere added at", (x, y, z))

    def Pshutdown(self):
        print("Shutting down GUI...")
        try:
            az = float(0)
            el = float(0)

            self.shared_data["target_azimuth"].value = az
            self.shared_data["target_elevation"].value = el
            self.shared_data["go_to_target"].value = True  # trigger!

        except ValueError:
            print("Invalid input: please enter numeric values")
        self.shared_data["go_to_zero"].value = True
        QtWidgets.QApplication.quit()
        self.shared_data["shutdown"].value = True

if __name__ == "__main__":
    '''app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec_())'''



_shared_data = None  # global

def run_gui(shared, movement_queue):
    global _shared_data
    shared["movement_queue"] = movement_queue
    _shared_data = shared

    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow(_shared_data, movement_queue)
    window.show()
    sys.exit(app.exec_())

