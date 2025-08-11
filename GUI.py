# GUI.py

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QCheckBox
import pyqtgraph.opengl as gl


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data, movement_queue):
        super().__init__()
        self.setWindowTitle("LockedIn Martin - Hardware Controller GUI")
        self.resize(1300, 700)
        self.debug_scale = 0.1
        self.orbit_items = []
        self.shared_data = shared_data
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30)  # Increased distance to see more
        main_layout.addWidget(self.view, stretch=3)

        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)  # Scale grid to be larger
        self.view.addItem(grid)

        self.background_plot = gl.GLScatterPlotItem(size=5, color=(0.5, 0.5, 1, 0.5))
        self.view.addItem(self.background_plot)

        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.satellite)

        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(self.laser)

        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5)
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded')
        self.view.addItem(self.station)

        controls_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)

        # --- Controls ---
        controls_layout.addWidget(QtWidgets.QLabel("CONTROLLER STATUS"))
        self.status_label = QtWidgets.QLabel("Status: IDLE")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #808080;")
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.create_separator())

        mode_box = QtWidgets.QGroupBox("Mode Control")
        mode_layout = QtWidgets.QVBoxLayout()
        self.btn_scan = QtWidgets.QPushButton("Start Background Scan")
        self.btn_scan.setCheckable(True)
        self.btn_scan.clicked.connect(self.on_scan_toggled)
        mode_layout.addWidget(self.btn_scan)
        self.btn_save_scan = QtWidgets.QPushButton("Save Scan Data")
        self.btn_save_scan.clicked.connect(self.on_save_scan_clicked)
        mode_layout.addWidget(self.btn_save_scan)
        self.btn_search = QtWidgets.QPushButton("Start Target Search")
        self.btn_search.setCheckable(True)
        self.btn_search.clicked.connect(self.on_search_toggled)
        mode_layout.addWidget(self.btn_search)
        self.chk_hf_tracking = QtWidgets.QCheckBox("Enable High-Frequency Tracking")
        self.chk_hf_tracking.toggled.connect(self.on_hf_track_toggled)
        mode_layout.addWidget(self.chk_hf_tracking)
        mode_box.setLayout(mode_layout)
        controls_layout.addWidget(mode_box)

        manual_box = QtWidgets.QGroupBox("Manual Control")
        manual_layout = QtWidgets.QVBoxLayout()
        manual_layout.addWidget(QtWidgets.QLabel("Target Azimuth (°)"))
        self.az_input = QtWidgets.QLineEdit("90")
        manual_layout.addWidget(self.az_input)
        manual_layout.addWidget(QtWidgets.QLabel("Target Elevation (°)"))
        self.el_input = QtWidgets.QLineEdit("45")
        manual_layout.addWidget(self.el_input)
        self.btn_go = QtWidgets.QPushButton("Go To Position")
        self.btn_go.clicked.connect(self.on_go_clicked)
        manual_layout.addWidget(self.btn_go)
        manual_box.setLayout(manual_layout)
        controls_layout.addWidget(manual_box)

        vis_box = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()
        self.btn_show_background = QtWidgets.QPushButton("Show/Hide Background Plot")
        self.btn_show_background.clicked.connect(self.toggle_background_plot)
        vis_layout.addWidget(self.btn_show_background)
        self.btn_add_sphere = QtWidgets.QPushButton("Add Sphere at Target")
        self.btn_add_sphere.clicked.connect(self.add_sphere)
        vis_layout.addWidget(self.btn_add_sphere)
        vis_box.setLayout(vis_layout)
        controls_layout.addWidget(vis_box)

        ekf_box = QtWidgets.QGroupBox("EKF Control")
        ekf_layout = QVBoxLayout()
        self.btn_acquire = QPushButton("Acquire (3 pts for EKF)")
        self.btn_acquire.clicked.connect(lambda: setattr(self.shared_data["acquire_points"], 'value', True))
        ekf_layout.addWidget(self.btn_acquire)
        self.btn_stop_ekf = QPushButton("Stop EKF & Generate Plot")
        self.btn_stop_ekf.clicked.connect(self.stop_and_plot_ekf)
        ekf_layout.addWidget(self.btn_stop_ekf)
        self.chk_debug_mode = QCheckBox("Debug Mode (Hand Tracking)")
        self.chk_debug_mode.toggled.connect(self.on_debug_mode_toggled)
        ekf_layout.addWidget(self.chk_debug_mode)
        ekf_box.setLayout(ekf_layout)
        controls_layout.addWidget(ekf_box)

        lcd_layout = QtWidgets.QGridLayout()
        lcd_layout.addWidget(QtWidgets.QLabel("Pan Angle"), 0, 0)
        lcd_layout.addWidget(QtWidgets.QLabel("Tilt Angle"), 0, 1)
        self.lcd_pan = self.create_lcd()
        lcd_layout.addWidget(self.lcd_pan, 1, 0)
        self.lcd_tilt = self.create_lcd()
        lcd_layout.addWidget(self.lcd_tilt, 1, 1)
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Range (cm)"), 2, 0)
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Strength"), 2, 1)
        self.lcd_range = self.create_lcd()
        lcd_layout.addWidget(self.lcd_range, 3, 0)
        self.lcd_strength = self.create_lcd()
        lcd_layout.addWidget(self.lcd_strength, 3, 1)
        controls_layout.addLayout(lcd_layout)
        controls_layout.addStretch()

        self.btn_shutdown = QtWidgets.QPushButton("Shutdown")
        self.btn_shutdown.setStyleSheet("background-color: #C41E3A; color: white; font-weight: bold;")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        controls_layout.addWidget(self.btn_shutdown)

        # --- Timers ---
        self.timer_3d = QtCore.QTimer(self)
        self.timer_3d.timeout.connect(self.update_laser_from_pan_tilt)  # This line was causing the error
        self.timer_3d.start(33)  # ~30 FPS

        self.timer_readouts = QtCore.QTimer(self)
        self.timer_readouts.timeout.connect(self.update_readouts)
        self.timer_readouts.start(100)  # 10 Hz

    def create_lcd(self):
        lcd = QtWidgets.QLCDNumber()
        lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        lcd.setDigitCount(6)
        return lcd

    def create_separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def on_scan_toggled(self, checked):
        if checked: self.shared_data["go_to_target"].value = False
        self.shared_data["background_scan_active"].value = checked
        if not checked: self.btn_scan.setChecked(False)

    def on_save_scan_clicked(self):
        self.shared_data["save_background_trigger"].value = True

    def on_search_toggled(self, checked):
        if checked: self.shared_data["go_to_target"].value = False
        self.shared_data["search_mode_active"].value = checked
        if not checked: self.btn_search.setChecked(False)

    def on_hf_track_toggled(self, checked):
        self.shared_data["lidar_track_mode_active"].value = checked

    def on_go_clicked(self):
        try:
            az, el = float(self.az_input.text()), float(self.el_input.text())
            self.shared_data["target_azimuth"].value, self.shared_data["target_elevation"].value = az, el
            self.shared_data["go_to_target"].value = True
        except ValueError:
            print("[GUI] Invalid input")

    def toggle_background_plot(self):
        if self.background_plot.visible():
            self.background_plot.hide()
            return
        try:
            bg_data = np.load(self.shared_data["background_path"])
            points = []
            for reading in bg_data:
                az, el, dist_cm, _ = reading
                if 10 < dist_cm < 16000:
                    az_rad, el_rad = np.radians(az), np.radians(el)
                    dist_m = dist_cm / 100.0

                    # --- CORRECTED COORDINATE CONVERSION ---
                    # Standard formula: Az=0 -> +X, Az=90 -> +Y
                    x = dist_m * np.cos(el_rad) * np.cos(az_rad)
                    y = dist_m * np.cos(el_rad) * np.sin(az_rad)
                    z = dist_m * np.sin(el_rad)
                    # ----------------------------------------
                    points.append([x, y, z])

            if points:
                self.background_plot.setData(pos=np.array(points))
                self.background_plot.show()
        except FileNotFoundError:
            print(f"[GUI] Error: '{self.shared_data['background_path']}' not found.")
        except Exception as e:
            print(f"[GUI] Error loading background data: {e}")

    def update_readouts(self):
        self.lcd_pan.display(self.shared_data['stepper_degrees'].value)
        self.lcd_tilt.display(self.shared_data['servo_degrees'].value)
        self.lcd_range.display(self.shared_data['lidar_data'][0])
        self.lcd_strength.display(self.shared_data['lidar_data'][1])
        if self.shared_data["lidar_track_mode_active"].value:
            self.status_label.setText("Status: HF TRACKING")
            self.status_label.setStyleSheet("color: #D22B2B;")
        elif self.shared_data["background_scan_active"].value:
            self.status_label.setText("Status: SCANNING")
            self.status_label.setStyleSheet("color: #FFBF00;")
            self.btn_scan.setChecked(True)
        elif self.shared_data["search_mode_active"].value:
            self.status_label.setText("Status: SEARCHING")
            self.status_label.setStyleSheet("color: #FFBF00;")
            self.btn_search.setChecked(True)
        elif self.shared_data["go_to_target"].value:
            if self.shared_data["target_reached"].value:
                self.status_label.setText("Status: HOLDING POSITION")
                self.status_label.setStyleSheet("color: #228B22;")
            else:
                self.status_label.setText("Status: MOVING")
                self.status_label.setStyleSheet("color: #33F;")
        else:
            self.status_label.setText("Status: IDLE")
            self.status_label.setStyleSheet("color: #808080;")
            if self.btn_scan.isChecked(): self.btn_scan.setChecked(False)
            if self.btn_search.isChecked(): self.btn_search.setChecked(False)

    def Pshutdown(self):
        self.shared_data["shutdown"].value = True
        # Use a short timer to allow the shutdown signal to propagate before quitting the app
        QtCore.QTimer.singleShot(100, QtWidgets.QApplication.instance().quit)

    def closeEvent(self, event):
        self.Pshutdown()
        super().closeEvent(event)

    def stop_and_plot_ekf(self):
        self.shared_data["generate_plot_on_stop"].value = True
        self.shared_data["ekf_running"].value = False

    def on_debug_mode_toggled(self, enabled):
        self.shared_data["debug_mode"].value = bool(enabled)
        min_range, max_range = (0.2, 2.0) if enabled else (3.0, 50.0)
        self.shared_data["lidar_acceptance_range"][0] = min_range
        self.shared_data["lidar_acceptance_range"][1] = max_range

    # --- THIS METHOD WAS MISSING ---
    def update_laser_from_pan_tilt(self):
        try:
            az, el = self.shared_data['stepper_degrees'].value, self.shared_data['servo_degrees'].value
            dist_cm = self.shared_data['lidar_data'][0]
            length_m = dist_cm / 100.0 if 10.0 <= dist_cm <= 16000.0 else 15.0
            length_m *= 100
            az_rad, el_rad = np.radians(az), np.radians(el)

            # --- CORRECTED COORDINATE CONVERSION ---
            # Use same formula as background plot for consistency
            x = length_m * np.cos(el_rad) * np.cos(az_rad)
            y = length_m * np.cos(el_rad) * np.sin(az_rad)
            z = length_m * np.sin(el_rad)
            # ----------------------------------------

            tip = np.array([x, y, z])
            self.laser.setData(pos=np.vstack((np.zeros(3), tip)))
            self.satellite.setData(pos=tip.reshape(1, 3))
        except Exception as e:
            # This can fire often on startup, so silence it
            # print(f"[GUI] Error updating laser: {e}")
            pass

    def add_sphere(self):
        md = gl.MeshData.sphere(rows=5, cols=10, radius=0.5)
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 1), shader='shaded')
        # Ensure self.satellite.pos is a valid array before trying to access it
        if self.satellite.pos is not None and self.satellite.pos.shape[0] > 0:
            x, y, z = self.satellite.pos[0]
            sphere.translate(x, y, z)
            self.view.addItem(sphere)


def run_gui(shared_data, movement_queue):
    # Ensure a QApplication instance exists
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    window = TrackerWindow(shared_data, movement_queue)
    window.show()
    sys.exit(app.exec_())