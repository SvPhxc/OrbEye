# GUI.py

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.opengl as gl

class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data, movement_queue):
        super().__init__()
        self.setWindowTitle("LockedIn Martin Drone Tracker")
        self.resize(1300, 700)
        self.shared_data = shared_data
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30)
        main_layout.addWidget(self.view, stretch=3)
        self.view.addItem(gl.GLGridItem())

        # --- NEW: Scatter plot item for the background scan ---
        self.background_plot = gl.GLScatterPlotItem(size=2, color=(0.5, 0.5, 1, 0.5))
        self.view.addItem(self.background_plot)
        
        controls_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)
        
        sys_box = QtWidgets.QGroupBox("System Control")
        sys_layout = QtWidgets.QVBoxLayout()
        sys_box.setLayout(sys_layout)
        controls_layout.addWidget(sys_box)

        self.btn_background = QtWidgets.QPushButton("Start Background Scan")
        self.btn_background.clicked.connect(self.background_scan)
        sys_layout.addWidget(self.btn_background)

        # --- NEW: Button to toggle the background visualization ---
        self.btn_show_background = QtWidgets.QPushButton("Show/Hide Background")
        self.btn_show_background.clicked.connect(self.toggle_background_plot)
        sys_layout.addWidget(self.btn_show_background)

        self.btn_follow = QtWidgets.QPushButton("Follow Drone")
        self.btn_follow.setCheckable(True)
        self.btn_follow.toggled.connect(self.toggle_follow_mode)
        self.btn_follow.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        sys_layout.addWidget(self.btn_follow)

        self.btn_shutdown = QtWidgets.QPushButton("Shutdown All")
        self.btn_shutdown.setStyleSheet("background-color: #ff4d4d;")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        sys_layout.addWidget(self.btn_shutdown)

        # (The rest of the GUI layout remains the same as before)
        # --- Manual Targeting ---
        target_box = QtWidgets.QGroupBox("Manual Targeting")
        target_layout = QtWidgets.QGridLayout()
        target_box.setLayout(target_layout)
        controls_layout.addWidget(target_box)

        target_layout.addWidget(QtWidgets.QLabel("Target Azimuth (°):"), 0, 0)
        self.az_input = QtWidgets.QLineEdit("90.0")
        target_layout.addWidget(self.az_input, 0, 1)

        target_layout.addWidget(QtWidgets.QLabel("Target Elevation (°):"), 1, 0)
        self.el_input = QtWidgets.QLineEdit("45.0")
        target_layout.addWidget(self.el_input, 1, 1)

        self.btn_go = QtWidgets.QPushButton("Go to Target")
        self.btn_go.clicked.connect(self.on_go_clicked)
        target_layout.addWidget(self.btn_go, 2, 0, 1, 2)

        # --- Pan/Tilt Controls ---
        dpad_box = QtWidgets.QGroupBox("Manual Pan/Tilt")
        dpad_layout = QtWidgets.QGridLayout()
        dpad_box.setLayout(dpad_layout)
        controls_layout.addWidget(dpad_box)

        btn_up = QtWidgets.QPushButton("↑ Tilt Up"); btn_down = QtWidgets.QPushButton("↓ Tilt Down")
        btn_left = QtWidgets.QPushButton("← Pan Left"); btn_right = QtWidgets.QPushButton("→ Pan Right")
        
        dpad_layout.addWidget(btn_up, 0, 1); dpad_layout.addWidget(btn_left, 1, 0)
        dpad_layout.addWidget(btn_right, 1, 2); dpad_layout.addWidget(btn_down, 2, 1)

        btn_up.clicked.connect(self.set_tilt_up); btn_down.clicked.connect(self.set_tilt_down)
        btn_left.clicked.connect(self.set_pan_left); btn_right.clicked.connect(self.set_pan_right)

        # --- Status Displays ---
        status_box = QtWidgets.QGroupBox("Live Status")
        status_layout = QtWidgets.QGridLayout()
        status_box.setLayout(status_layout)
        controls_layout.addWidget(status_box)

        status_layout.addWidget(QtWidgets.QLabel("Current Pan (°):"), 0, 0); self.lcd_pan = self.create_lcd()
        status_layout.addWidget(self.lcd_pan, 0, 1)
        status_layout.addWidget(QtWidgets.QLabel("Current Tilt (°):"), 1, 0); self.lcd_tilt = self.create_lcd()
        status_layout.addWidget(self.lcd_tilt, 1, 1)
        status_layout.addWidget(QtWidgets.QLabel("LiDAR Range (cm):"), 2, 0); self.lcd_range = self.create_lcd()
        status_layout.addWidget(self.lcd_range, 2, 1)
        status_layout.addWidget(QtWidgets.QLabel("LiDAR Strength:"), 3, 0); self.lcd_strength = self.create_lcd()
        status_layout.addWidget(self.lcd_strength, 3, 1)

        controls_layout.addStretch()

        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_displays)
        self.status_timer.start(100)

    def create_lcd(self):
        lcd = QtWidgets.QLCDNumber()
        lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        lcd.setDigitCount(6)
        return lcd

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
                        dist_m = dist_cm / 100.0 # Scale to meters for visualization

                        # Spherical to Cartesian conversion
                        x = dist_m * np.cos(el_rad) * np.cos(az_rad)
                        y = dist_m * np.cos(el_rad) * np.sin(az_rad)
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

    def background_scan(self):
        print("[GUI] Triggering background scan")
        self.shared_data["scan_trigger"].value = True

    def toggle_follow_mode(self, checked):
        self.shared_data["follow_drone_enabled"].value = checked
        if checked:
            print("[GUI] Autonomous drone following ENABLED.")
            self.btn_go.setEnabled(False)
            self.btn_background.setEnabled(False)
        else:
            print("[GUI] Autonomous drone following DISABLED.")
            self.btn_go.setEnabled(True)
            self.btn_background.setEnabled(True)

    def on_go_clicked(self):
        try:
            az = float(self.az_input.text()); el = float(self.el_input.text())
            self.shared_data["target_azimuth"].value = az
            self.shared_data["target_elevation"].value = el
            self.shared_data["go_to_target"].value = True
        except ValueError: print("Invalid input: please enter numeric values")

    def set_tilt_up(self): self.shared_data['tilt_up'].value = True
    def set_tilt_down(self): self.shared_data['tilt_down'].value = True
    def set_pan_left(self): self.shared_data['pan_left'].value = True
    def set_pan_right(self): self.shared_data['pan_right'].value = True
    
    def update_displays(self):
        self.lcd_range.display(self.shared_data["lidar_data"][0])
        self.lcd_strength.display(self.shared_data["lidar_data"][1])
        self.lcd_pan.display(f"{self.shared_data['stepper_degrees'].value:.1f}")
        self.lcd_tilt.display(f"{self.shared_data['servo_degrees'].value:.1f}")

    def Pshutdown(self):
        print("Shutting down from GUI...")
        self.shared_data["shutdown"].value = True
        QtCore.QTimer.singleShot(500, QtWidgets.QApplication.instance().quit)

    def closeEvent(self, event):
        self.Pshutdown()
        event.accept()

_shared_data = None
def run_gui(shared, movement_queue):
    global _shared_data
    _shared_data = shared
    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow(_shared_data, movement_queue)
    window.show()
    sys.exit(app.exec_())
