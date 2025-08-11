import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time
import os

# Note: The GUI no longer needs to import from motor_controller
# It only interacts with the shared data dictionary.
from datahandler import (
    parse_tle_file,
    generate_orbit_xyz,
    fetch_tle_by_name,
    get_sofia_eci
)


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data, movement_queue):
        super().__init__()
        self.setWindowTitle("LockedIn Martin - Hardware Controller GUI")
        self.resize(1300, 700)  # Increased height slightly for new controls
        self.debug_scale = 0.1
        self.orbit_items = []

        # --- Use the provided shared_data dictionary ---
        # The 'movement_queue' is no longer used by this version of the GUI
        self.shared_data = shared_data

        # --- Main Layout ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # === Left: 3D View ===
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=20)
        main_layout.addWidget(self.view, stretch=3)

        grid = gl.GLGridItem();
        self.view.addItem(grid)
        self.background_plot = gl.GLScatterPlotItem(size=5, color=(0.5, 0.5, 1, 0.5))
        self.view.addItem(self.background_plot)
        self.satellite = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.view.addItem(self.satellite)
        self.laser = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(self.laser)
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.5);
        self.station = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 0, 1, 1), shader='shaded');
        self.view.addItem(self.station)

        # === Right: Control Panel ===
        controls_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)

        # --- NEW: Status Display ---
        controls_layout.addWidget(QtWidgets.QLabel("CONTROLLER STATUS"))
        self.status_label = QtWidgets.QLabel("Status: IDLE")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #33F;")
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.create_separator())

        # --- Mode Control Group ---
        mode_box = QtWidgets.QGroupBox("Mode Control")
        mode_layout = QtWidgets.QVBoxLayout()

        self.btn_scan = QtWidgets.QPushButton("Start Background Scan")
        self.btn_scan.setCheckable(True)  # Make it a toggle button
        self.btn_scan.clicked.connect(self.on_scan_toggled)
        mode_layout.addWidget(self.btn_scan)

        self.btn_save_scan = QtWidgets.QPushButton("Save Scan Data")
        self.btn_save_scan.clicked.connect(self.on_save_scan_clicked)
        mode_layout.addWidget(self.btn_save_scan)

        self.btn_search = QtWidgets.QPushButton("Start Target Search")
        self.btn_search.setCheckable(True)  # Toggle button
        self.btn_search.clicked.connect(self.on_search_toggled)
        mode_layout.addWidget(self.btn_search)

        self.chk_hf_tracking = QtWidgets.QCheckBox("Enable High-Frequency Tracking")
        self.chk_hf_tracking.toggled.connect(self.on_hf_track_toggled)
        mode_layout.addWidget(self.chk_hf_tracking)

        mode_box.setLayout(mode_layout)
        controls_layout.addWidget(mode_box)

        # --- Manual Control Group ---
        manual_box = QtWidgets.QGroupBox("Manual Control")
        manual_layout = QtWidgets.QVBoxLayout()

        manual_layout.addWidget(QtWidgets.QLabel("Target Azimuth (°)"))
        self.az_input = QtWidgets.QLineEdit("90");
        manual_layout.addWidget(self.az_input)
        manual_layout.addWidget(QtWidgets.QLabel("Target Elevation (°)"))
        self.el_input = QtWidgets.QLineEdit("45");
        manual_layout.addWidget(self.el_input)
        self.btn_go = QtWidgets.QPushButton("Go To Position");
        self.btn_go.clicked.connect(self.on_go_clicked)
        manual_layout.addWidget(self.btn_go)

        manual_box.setLayout(manual_layout)
        controls_layout.addWidget(manual_box)

        # --- Visualization Group ---
        vis_box = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()
        self.btn_show_background = QtWidgets.QPushButton("Show/Hide Background Plot")
        self.btn_show_background.clicked.connect(self.toggle_background_plot)
        vis_layout.addWidget(self.btn_show_background)
        self.btn_add_sphere = QtWidgets.QPushButton("Add Sphere at Target");
        self.btn_add_sphere.clicked.connect(self.add_sphere)
        vis_layout.addWidget(self.btn_add_sphere)
        vis_box.setLayout(vis_layout)
        controls_layout.addWidget(vis_box)

        # --- EKF Control Group (for other processes) ---
        ekf_box = QtWidgets.QGroupBox("EKF Control")
        ekf_layout = QtWidgets.QVBoxLayout()  # <-- CORRECTED
        self.btn_acquire = QtWidgets.QPushButton("Acquire (3 pts for EKF)")  # <-- CORRECTED
        self.btn_acquire.clicked.connect(lambda: setattr(self.shared_data["acquire_points"], 'value', True))
        ekf_layout.addWidget(self.btn_acquire)
        self.btn_stop_ekf = QtWidgets.QPushButton("Stop EKF & Generate Plot")  # <-- CORRECTED
        self.btn_stop_ekf.clicked.connect(self.stop_and_plot_ekf)
        ekf_layout.addWidget(self.btn_stop_ekf)
        self.chk_debug_mode = QtWidgets.QCheckBox("Debug Mode (Hand Tracking)")  # <-- CORRECTED
        self.chk_debug_mode.toggled.connect(self.on_debug_mode_toggled)
        ekf_layout.addWidget(self.chk_debug_mode)
        ekf_box.setLayout(ekf_layout)
        controls_layout.addWidget(ekf_box)

        # --- LCD Displays ---
        lcd_layout = QtWidgets.QGridLayout()
        lcd_layout.addWidget(QtWidgets.QLabel("Pan Angle"), 0, 0);
        lcd_layout.addWidget(QtWidgets.QLabel("Tilt Angle"), 0, 1)
        self.lcd_pan = self.create_lcd();
        lcd_layout.addWidget(self.lcd_pan, 1, 0)
        self.lcd_tilt = self.create_lcd();
        lcd_layout.addWidget(self.lcd_tilt, 1, 1)
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Range (cm)"), 2, 0);
        lcd_layout.addWidget(QtWidgets.QLabel("LiDAR Strength"), 2, 1)
        self.lcd_range = self.create_lcd();
        lcd_layout.addWidget(self.lcd_range, 3, 0)
        self.lcd_strength = self.create_lcd();
        lcd_layout.addWidget(self.lcd_strength, 3, 1)
        controls_layout.addLayout(lcd_layout)

        controls_layout.addStretch()  # Pushes everything up

        # --- Shutdown Button ---
        self.btn_shutdown = QtWidgets.QPushButton("Shutdown")
        self.btn_shutdown.setStyleSheet("background-color: #C41E3A; color: white; font-weight: bold;")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        controls_layout.addWidget(self.btn_shutdown)

        # --- Timers for updating the display ---
        self.timer_3d = QtCore.QTimer(self);
        self.timer_3d.timeout.connect(self.update_laser_from_pan_tilt);
        self.timer_3d.start(33)  # ~30fps
        self.timer_readouts = QtCore.QTimer(self);
        self.timer_readouts.timeout.connect(self.update_readouts);
        self.timer_readouts.start(100)  # 10fps

    # --- NEW: Helper functions for creating UI elements ---
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

    # --- NEW: Click Handlers for New/Updated Buttons ---
    def on_scan_toggled(self, checked):
        print(f"[GUI] Background Scan toggled: {'ON' if checked else 'OFF'}")
        self.shared_data["background_scan_active"].value = checked
        if not checked:  # If user manually turns it off, ensure button reflects this
            self.btn_scan.setChecked(False)

    def on_save_scan_clicked(self):
        print("[GUI] Triggering save for background scan data.")
        self.shared_data["save_background_trigger"].value = True

    def on_search_toggled(self, checked):
        print(f"[GUI] Target Search toggled: {'ON' if checked else 'OFF'}")
        self.shared_data["search_mode_active"].value = checked
        if not checked:
            self.btn_search.setChecked(False)

    def on_hf_track_toggled(self, checked):
        print(f"[GUI] High-Frequency Tracking toggled: {'ON' if checked else 'OFF'}")
        self.shared_data["lidar_track_mode_active"].value = checked

    # --- UPDATED: Go button handler ---
    def on_go_clicked(self):
        try:
            az = float(self.az_input.text())
            el = float(self.el_input.text())
            print(f"[GUI] Commanding Go To Position: Az={az}, El={el}")
            self.shared_data["target_azimuth"].value = az
            self.shared_data["target_elevation"].value = el
            self.shared_data["go_to_target"].value = True
        except ValueError:
            print("[GUI] Invalid input: please enter numeric values")

    # --- UPDATED: Background plot loader for new data format ---
    def toggle_background_plot(self):
        if self.background_plot.visible():
            self.background_plot.hide();
            print("[GUI] Background visualization hidden.")
            return
        try:
            # The new format is a simple list of [az, el, dist, strength]
            bg_data = np.load(self.shared_data["background_path"])
            points = []
            for reading in bg_data:
                az, el, dist_cm, _ = reading[0], reading[1], reading[2], reading[3]
                if 10 < dist_cm < 16000:  # Generous range in cm
                    az_rad, el_rad = np.radians(az), np.radians(el)
                    dist_m = dist_cm / 100.0  # Scale to meters
                    x = dist_m * np.cos(el_rad) * np.cos(az_rad)
                    y = dist_m * np.cos(el_rad) * np.sin(az_rad)
                    z = dist_m * np.sin(el_rad)
                    points.append([x, y, z])
            if points:
                print(f"[GUI] Plotting {len(points)} background points.")
                self.background_plot.setData(pos=np.array(points));
                self.background_plot.show()
            else:
                print("[GUI] No valid points found in background data file.")
        except FileNotFoundError:
            print(f"[GUI] Error: '{self.shared_data['background_path']}' not found.")
        except Exception as e:
            print(f"[GUI] Error loading background data: {e}")

    # --- UPDATED: Combined readout update function ---
    def update_readouts(self):
        # Update LCDs for angles and LiDAR
        self.lcd_pan.display(self.shared_data['stepper_degrees'].value)
        self.lcd_tilt.display(self.shared_data['servo_degrees'].value)
        self.lcd_range.display(self.shared_data['lidar_data'][0])
        self.lcd_strength.display(self.shared_data['lidar_data'][1])

        # Update status label based on hardware controller state flags
        if self.shared_data["lidar_track_mode_active"].value:
            self.status_label.setText("Status: HF TRACKING")
            self.status_label.setStyleSheet("color: #D22B2B;")  # Red
        elif self.shared_data["background_scan_active"].value:
            self.status_label.setText("Status: SCANNING")
            self.status_label.setStyleSheet("color: #FFBF00;")  # Amber
            self.btn_scan.setChecked(True)  # Ensure button state matches
        elif self.shared_data["search_mode_active"].value:
            self.status_label.setText("Status: SEARCHING")
            self.status_label.setStyleSheet("color: #FFBF00;")  # Amber
            self.btn_search.setChecked(True)  # Ensure button state matches
        elif self.shared_data["go_to_target"].value:
            if self.shared_data["target_reached"].value:
                self.status_label.setText("Status: HOLDING POSITION")
                self.status_label.setStyleSheet("color: #228B22;")  # Green
            else:
                self.status_label.setText("Status: MOVING")
                self.status_label.setStyleSheet("color: #33F;")  # Blue
        else:
            self.status_label.setText("Status: IDLE")
            self.status_label.setStyleSheet("color: #808080;")  # Gray
            # Ensure toggle buttons are off if no activity is happening
            self.btn_scan.setChecked(False)
            self.btn_search.setChecked(False)

    # --- OBSOLETE: D-Pad methods removed ---
    # set_tilt_up, set_tilt_down, set_pan_left, set_pan_right are no longer needed.

    # --- UPDATED: Simplified shutdown ---
    def Pshutdown(self):
        print("[GUI] Shutdown requested. Signaling other processes.")
        self.shared_data["shutdown"].value = True
        # Allow a moment for other processes to see the flag before quitting the app
        QtCore.QTimer.singleShot(100, QtWidgets.QApplication.instance().quit)

    def closeEvent(self, event):
        """Ensure shutdown is called when window is closed."""
        self.Pshutdown()
        super().closeEvent(event)

    # --- Other methods (mostly unchanged) ---
    def stop_and_plot_ekf(self):
        print("[GUI] Requesting EKF stop and plot generation.")
        self.shared_data["generate_plot_on_stop"].value = True
        self.shared_data["ekf_running"].value = False

    def on_debug_mode_toggled(self, enabled):
        self.shared_data["debug_mode"].value = bool(enabled)
        # Configure LiDAR acceptance range based on mode
        min_range, max_range = (0.2, 2.0) if enabled else (3.0, 50.0)
        self.shared_data["lidar_acceptance_range"][0] = min_range
        self.shared_data["lidar_acceptance_range"][1] = max_range
        print(f"[GUI] Debug Mode {'ENABLED' if enabled else 'DISABLED'}. Range set to {min_range}m - {max_range}m.")

    def update_laser_from_pan_tilt(self):
        try:
            az = self.shared_data['stepper_degrees'].value
            el = self.shared_data['servo_degrees'].value
            dist_cm = self.shared_data['lidar_data'][0]
            length_m = dist_cm / 100.0 if 10.0 <= dist_cm <= 16000.0 else 15.0
            az_rad, el_rad = np.radians(az), np.radians(el)
            x = length_m * np.cos(el_rad) * np.cos(az_rad)
            y = length_m * np.cos(el_rad) * np.sin(az_rad)
            z = length_m * np.sin(el_rad)
            tip = np.array([x, y, z])
            self.laser.setData(pos=np.vstack((np.zeros(3), tip)))
            self.satellite.setData(pos=tip.reshape(1, 3))
        except Exception as e:
            # This can happen on startup if shared memory isn't populated yet
            pass

    def add_sphere(self):
        md = gl.MeshData.sphere(rows=5, cols=10, radius=0.5)
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 1), shader='shaded')
        x, y, z = self.satellite.pos[0]
        sphere.translate(x, y, z);
        self.view.addItem(sphere)
        print("Sphere added at", (x, y, z))


# --- Main entry point for the GUI process ---
def run_gui(shared_data, movement_queue):
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow(shared_data, movement_queue)  # movement_queue is ignored but kept for compatibility
    window.show()
    sys.exit(app.exec_())