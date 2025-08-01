import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time

from datahandler import parse_tle_file, generate_orbit_xyz, plot_orbit_with_earth




class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LockedIn Martin")
        self.resize(1000, 600)
        self.debug_scale = 0.01

        # Central widget and main layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # === Left: 3D View ===
        self.view = gl.GLViewWidget()
        self.view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.view.setCameraPosition(distance=20)
        main_layout.addWidget(self.view, stretch=3)

        # Grid
        grid = gl.GLGridItem()
        self.view.addItem(grid)

        # Satellite (drone)
        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.satellite)

        # Laser beam
        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(self.laser)

        # Station at origin
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5)
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded')
        self.station.translate(0, 0, 0)
        self.view.addItem(self.station)

        # Orbit parameters
        self.radius = 8
        self.orbit_speed = 0.2
        self.elevation_deg = 30  # initial elevation angle
        self.start_time = time.time()
        self.orbit_enabled = True

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_orbit)
        self.timer.start(30)

        # === Right: Control Panel ===
        controls = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls, stretch=1)

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

        # Elevation Angle Controls
        controls.addWidget(QtWidgets.QLabel("Elevation Angle (°)"))
        self.slider_elevation = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_elevation.setMinimum(0)
        self.slider_elevation.setMaximum(90)
        self.slider_elevation.setValue(self.elevation_deg)
        self.slider_elevation.valueChanged.connect(self.set_elevation_angle)
        controls.addWidget(self.slider_elevation)

        self.lcd_elevation = QtWidgets.QLCDNumber()
        self.lcd_elevation.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_elevation.setDigitCount(4)
        self.lcd_elevation.display(self.elevation_deg)
        controls.addWidget(self.lcd_elevation)

        controls.addStretch()

        self.orbit_xyz = generate_orbit_xyz('example.tle', duration_minutes=90, step_seconds=30)
        self.plot_orbit_spheres()
        # or
        # self.plot_orbit_points_fast()

    def keyPressEvent(self, event):
        key = event.key()
        if key == pg.QtCore.Qt.Key_A:
            self.add_sphere()

    def add_sphere(self):
        md = gl.MeshData.sphere(rows=5, cols=10, radius=0.1)
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 1), shader='shaded')
        x, y, z = self.satellite.pos[0]
        sphere.translate(x, y, z)
        self.view.addItem(sphere)
        print("Sphere added at", (x, y, z))

    def toggle_orbit(self, enabled):
        self.orbit_enabled = enabled
        if enabled:
            self.start_time = time.time()  # reset time

    def set_elevation_angle(self, value):
        self.elevation_deg = value
        self.lcd_elevation.display(value)

    def update_orbit(self):
        if not self.orbit_enabled:
            return

        # Time and orbit angle
        t = time.time() - self.start_time
        angle = self.orbit_speed * t

        # Convert elevation angle to radians
        elevation_rad = np.radians(self.elevation_deg)

        # Calculate position in spherical coordinates
        x = self.radius * np.cos(angle) * np.cos(elevation_rad)
        y = self.radius * np.sin(angle) * np.cos(elevation_rad)
        z = self.radius * np.sin(elevation_rad)

        # Update satellite position
        self.satellite.setData(pos=np.array([[x, y, z]]))

        # Update laser vector
        self.laser.setData(pos=np.array([[0, 0, 0], [x, y, z]]))

    def plot_orbit_spheres(self):
        # Create a small sphere at each XYZ point
        for point in self.orbit_xyz:
            if np.isnan(point).any():
                print("nothing added")
                continue  # skip invalid points
            md = gl.MeshData.sphere(rows=5, cols=10, radius=1)
            sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(1, 1, 0, 1), shader='shaded')
            scaled_point = np.array(point) * self.debug_scale
            sphere.translate(*scaled_point)
            self.view.addItem(sphere)  
            print("something added at", scaled_point)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec_())
    
