import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time

class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Tracker - Circular Orbit")
        self.resize(800, 600)

        # Central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # 3D View
        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)
        self.view.setCameraPosition(distance=20)

        # Grid
        grid = gl.GLGridItem()
        self.view.addItem(grid)

        # Satellite (drone)
        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.satellite)

        # Laser beam
        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(self.laser)

        # Parameters for circular orbit
        self.radius = 8
        self.orbit_speed = 0.2  # radians per second
        self.orbit_height = 3   # Z-axis level of the orbit
        self.start_time = time.time()

        # Timer for updates
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_orbit)
        self.timer.start(30)  # 30 ms ≈ 33 FPS

        # Create a small sphere using pyqtgraph's mesh data
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5)
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded')
        self.station.translate(0, 0, 0)  # Place at origin
        self.view.addItem(self.station)

    def update_orbit(self):
        # Time since start
        t = time.time() - self.start_time
        angle = self.orbit_speed * t

        # Circular orbit in XY plane at constant Z
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = self.orbit_height

        # Update satellite (drone) position
        self.satellite.setData(pos=np.array([[x, y, z]]))

        # Update laser pointing from origin to satellite
        self.laser.setData(pos=np.array([[0, 0, 0], [x, y, z]]))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec_())
