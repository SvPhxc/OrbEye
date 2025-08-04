import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time
import serial

from astropy.time import Time

from datahandler import (
    parse_tle_file,
    generate_orbit_xyz,
    fetch_tle_by_name,
    get_sofia_eci
)


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LockedIn Martin")
        self.resize(1000, 600)
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
        self.timer.timeout.connect(self.update_orbit)
        self.timer.start(30)

        # === Right: Controls ===
        controls = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls, stretch=1)

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

        # Toggle Orbit Button
        self.btn_toggle_orbit = QtWidgets.QPushButton("Toggle Orbit")
        self.btn_toggle_orbit.setCheckable(True)
        self.btn_toggle_orbit.setChecked(True)
        self.btn_toggle_orbit.toggled.connect(self.toggle_orbit)
        controls.addWidget(self.btn_toggle_orbit)

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

        # Shutdown button
        self.btn_shutdown = QtWidgets.QPushButton("Shutdown")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        controls.addWidget(self.btn_shutdown)

        self.lidar_timer = QtCore.QTimer()
        self.lidar_timer.timeout.connect(self.update_lidar_display)
        self.lidar_timer.start(50)  

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

    def remove_orbit(self):
        for item in self.orbit_items:
            self.view.removeItem(item)
        self.orbit_items.clear()

    def plot_orbit_line(self):
        valid_points = self.orbit_xyz[~np.isnan(self.orbit_xyz).any(axis=1)]
        scaled_points = valid_points * self.debug_scale
        orbit_line = gl.GLLinePlotItem(pos=scaled_points, color=(1, 1, 0, 1), width=2, antialias=True, mode='line_strip')
        self.view.addItem(orbit_line)
        self.orbit_items.append(orbit_line)

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

    def update_orbit(self):
        if not self.orbit_enabled:
            return
        t = time.time() - self.start_time
        angle = self.orbit_speed * t
        elevation_rad = np.radians(self.elevation_deg)
        x = self.radius * np.cos(angle) * np.cos(elevation_rad)
        y = self.radius * np.sin(angle) * np.cos(elevation_rad)
        z = self.radius * np.sin(elevation_rad)
        self.satellite.setData(pos=np.array([[x, y, z]]))
        self.laser.setData(pos=np.array([[0, 0, 0], [x, y, z]]))

    def toggle_orbit(self, enabled):
        self.orbit_enabled = enabled
        if enabled:
            self.start_time = time.time()

    def set_elevation_angle(self, value):
        self.elevation_deg = value
        self.lcd_elevation.display(value)
    
    '''def update_lidar_display(self):
        lidar_data = self.shared_data.get("lidar_data")
        if lidar_data is not None:
            distance = lidar_data[0]
            strength = lidar_data[1]
            self.lcd_range.display(distance)
            self.lcd_strength.display(strength)
            print(f"[GUI] LiDAR: {distance} cm | Strength: {strength}")'''
    
    def update_lidar_display(self):
        distance, strength = self.read_tfmini_data()
        if distance is not None:
            self.lcd_range.display(distance)
            self.lcd_strength.display(strength)
            print(f"[GUI] LiDAR (direct): {distance} cm | Strength: {strength}")
    
    def poll_lidar_serial(self, port="/dev/serial0", baudrate=115200):
        try:
            self.lidar_serial = serial.Serial(port, baudrate, timeout=1)
            print("[GUI] Connected to LiDAR serial")
        except serial.SerialException as e:
            print(f"[GUI] Serial connection error: {e}")
            self.lidar_serial = None



    def add_sphere(self):
        md = gl.MeshData.sphere(rows=5, cols=10, radius=1000)
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 1), shader='shaded')
        x, y, z = self.satellite.pos[0]
        sphere.translate(x, y, z)
        self.view.addItem(sphere)
        print("Sphere added at", (x, y, z))

    def Pshutdown(self):
        print("Shutting down GUI...")
        QtWidgets.QApplication.quit()
        self.shared_data["shutdown"].value = True

if __name__ == "__main__":
    '''app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec_())'''



_shared_data = None  # global

def run_gui(shared):
    global _shared_data
    _shared_data = shared

    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec_())

def read_tfmini_data(self):
    if not self.lidar_serial or self.lidar_serial.in_waiting < 9:
        return None, None

    if self.lidar_serial.read() != b'\x59':
        return None, None
    if self.lidar_serial.read() != b'\x59':
        return None, None

    raw = self.lidar_serial.read(7)
    distance = raw[0] + raw[1] * 256
    strength = raw[2] + raw[3] * 256
    return distance, strength