# File: GUI.py

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import opengl as gl


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data, movement_queue_ignored):
        super().__init__()
        self.setWindowTitle("LockedIn Martin Hardware Controller")
        self.resize(1300, 600)
        self.debug_scale = 0.1
        self.orbit_items = []
        self.shared_data = shared_data

        # --- UI Setup ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.view = gl.GLViewWidget();
        self.view.setCameraPosition(distance=20);
        main_layout.addWidget(self.view, stretch=3)
        grid = gl.GLGridItem();
        self.view.addItem(grid)
        self.background_plot = gl.GLScatterPlotItem(size=5, color=(0.5, 0.5, 1, 0.5));
        self.view.addItem(self.background_plot)
        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10);
        self.view.addItem(self.satellite)
        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2);
        self.view.addItem(self.laser)
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5);
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded');
        self.view.addItem(self.station)

        # --- Controls Panel ---
        controls = QtWidgets.QVBoxLayout();
        main_layout.addLayout(controls, stretch=1)

        # --- Manual Control Box ---
        dpad_controls = QtWidgets.QGroupBox("Manual Control")
        dpad_layout = QtWidgets.QGridLayout()
        btn_up = QtWidgets.QPushButton("↑ Tilt Up");
        btn_down = QtWidgets.QPushButton("↓ Tilt Down")
        btn_left = QtWidgets.QPushButton("← Pan Left");
        btn_right = QtWidgets.QPushButton("→ Pan Right")

        btn_up.pressed.connect(lambda: self.set_jog_flag('tilt_up', True));
        btn_up.released.connect(lambda: self.set_jog_flag('tilt_up', False))
        btn_down.pressed.connect(lambda: self.set_jog_flag('tilt_down', True));
        btn_down.released.connect(lambda: self.set_jog_flag('tilt_down', False))
        btn_left.pressed.connect(lambda: self.set_jog_flag('pan_left', True));
        btn_left.released.connect(lambda: self.set_jog_flag('pan_left', False))
        btn_right.pressed.connect(lambda: self.set_jog_flag('pan_right', True));
        btn_right.released.connect(lambda: self.set_jog_flag('pan_right', False))

        dpad_layout.addWidget(btn_up, 0, 1);
        dpad_layout.addWidget(btn_left, 1, 0)
        dpad_layout.addWidget(btn_right, 1, 2);
        dpad_layout.addWidget(btn_down, 2, 1)
        dpad_controls.setLayout(dpad_layout);
        controls.addWidget(dpad_controls)

        # (Other controls)
        self.btn_go = QtWidgets.QPushButton("Go to Target");
        self.btn_go.clicked.connect(self.on_go_clicked)
        self.az_input = QtWidgets.QLineEdit("180");
        self.el_input = QtWidgets.QLineEdit("45")
        controls.addWidget(self.az_input);
        controls.addWidget(self.el_input);
        controls.addWidget(self.btn_go)

        self.btn_acquire = QtWidgets.QPushButton("Acquire Target (Search)");
        self.btn_acquire.clicked.connect(self.acquire_points)
        controls.addWidget(self.btn_acquire)

        # --- Timers and LCDs ---
        lcd_layout = QtWidgets.QGridLayout()
        self.lcd_pan = QtWidgets.QLCDNumber(6);
        lcd_layout.addWidget(QtWidgets.QLabel("Pan Angle (°)"), 0, 0);
        lcd_layout.addWidget(self.lcd_pan, 0, 1)
        self.lcd_tilt = QtWidgets.QLCDNumber(6);
        lcd_layout.addWidget(QtWidgets.QLabel("Tilt Angle (°)"), 1, 0);
        lcd_layout.addWidget(self.lcd_tilt, 1, 1)
        self.lcd_range = QtWidgets.QLCDNumber(5);
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Range (cm)"), 2, 0);
        lcd_layout.addWidget(self.lcd_range, 2, 1)
        self.lcd_strength = QtWidgets.QLCDNumber(5);
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Strength"), 3, 0);
        lcd_layout.addWidget(self.lcd_strength, 3, 1)
        controls.addLayout(lcd_layout)

        controls.addStretch()
        self.btn_shutdown = QtWidgets.QPushButton("Shutdown System");
        self.btn_shutdown.setStyleSheet("background-color: #a83232; color: white;")
        self.btn_shutdown.clicked.connect(self.shutdown_system)
        controls.addWidget(self.btn_shutdown)

        self.timer = QtCore.QTimer();
        self.timer.timeout.connect(self.update_displays);
        self.timer.start(50)

    def update_displays(self):
        az = self.shared_data['stepper_degrees'].value;
        el = self.shared_data['servo_degrees'].value
        dist_cm = self.shared_data['lidar_data'][0]
        length_m = dist_cm / 100.0 if 50.0 <= dist_cm <= 2000.0 else 10.0
        length = length_m * self.debug_scale * 10
        az_rad, el_rad = np.radians(az), np.radians(el)
        x = np.cos(el_rad) * np.cos(az_rad);
        y = np.cos(el_rad) * np.sin(az_rad);
        z = np.sin(el_rad)
        tip = np.array([x * length, y * length, z * length])
        self.laser.setData(pos=np.vstack((np.zeros(3), tip)));
        self.satellite.setData(pos=tip.reshape(1, 3))
        self.lcd_pan.display(f"{az:.2f}");
        self.lcd_tilt.display(f"{el:.2f}")
        self.lcd_range.display(int(dist_cm));
        self.lcd_strength.display(int(self.shared_data['lidar_data'][1]))

    def set_jog_flag(self, flag_name, value):
        if flag_name in self.shared_data: self.shared_data[flag_name].value = value

    def on_go_clicked(self):
        try:
            self.shared_data["target_azimuth"].value = float(self.az_input.text())
            self.shared_data["target_elevation"].value = float(self.el_input.text())
            self.shared_data["go_to_target"].value = True
        except ValueError:
            print("[GUI] Invalid GoTo input.")

    def acquire_points(self):
        self.shared_data["acquire_points"].value = True

    def shutdown_system(self):
        self.shared_data["shutdown"].value = True
        QtWidgets.QApplication.quit()


def run_gui(shared_data, movement_queue_ignored):
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow(shared_data, movement_queue_ignored)
    window.show()
    sys.exit(app.exec_())