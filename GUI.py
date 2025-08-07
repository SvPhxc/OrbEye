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
        
        # Central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # === Left: 3D View ===
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30)
        main_layout.addWidget(self.view, stretch=3)
        self.view.addItem(gl.GLGridItem())

        # === Right: Controls ===
        controls_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)
        
        # --- System Controls ---
        sys_box = QtWidgets.QGroupBox("System Control")
        sys_layout = QtWidgets.QVBoxLayout()
        sys_box.setLayout(sys_layout)
        controls_layout.addWidget(sys_box)

        self.btn_background = QtWidgets.QPushButton("Start Background Scan")
        self.btn_background.clicked.connect(self.background_scan)
        sys_layout.addWidget(self.btn_background)

        # --- NEW: Follow Drone Button ---
        self.btn_follow = QtWidgets.QPushButton("Follow Drone")
        self.btn_follow.setCheckable(True)
        self.btn_follow.toggled.connect(self.toggle_follow_mode)
        self.btn_follow.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        sys_layout.addWidget(self.btn_follow)

        self.btn_shutdown = QtWidgets.QPushButton("Shutdown All")
        self.btn_shutdown.setStyleSheet("background-color: #ff4d4d;")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        sys_layout.addWidget(self.btn_shutdown)

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

        btn_up = QtWidgets.QPushButton("↑ Tilt Up")
        btn_down = QtWidgets.QPushButton("↓ Tilt Down")
        btn_left = QtWidgets.QPushButton("← Pan Left")
        btn_right = QtWidgets.QPushButton("→ Pan Right")
        
        dpad_layout.addWidget(btn_up, 0, 1)
        dpad_layout.addWidget(btn_left, 1, 0)
        dpad_layout.addWidget(btn_right, 1, 2)
        dpad_layout.addWidget(btn_down, 2, 1)

        btn_up.clicked.connect(self.set_tilt_up)
        btn_down.clicked.connect(self.set_tilt_down)
        btn_left.clicked.connect(self.set_pan_left)
        btn_right.clicked.connect(self.set_pan_right)

        # --- Status Displays ---
        status_box = QtWidgets.QGroupBox("Live Status")
        status_layout = QtWidgets.QGridLayout()
        status_box.setLayout(status_layout)
        controls_layout.addWidget(status_box)

        # Pan/Tilt Angles
        status_layout.addWidget(QtWidgets.QLabel("Current Pan (°):"), 0, 0)
        self.lcd_pan = self.create_lcd()
        status_layout.addWidget(self.lcd_pan, 0, 1)
        
        status_layout.addWidget(QtWidgets.QLabel("Current Tilt (°):"), 1, 0)
        self.lcd_tilt = self.create_lcd()
        status_layout.addWidget(self.lcd_tilt, 1, 1)

        # LiDAR Data
        status_layout.addWidget(QtWidgets.QLabel("LiDAR Range (cm):"), 2, 0)
        self.lcd_range = self.create_lcd()
        status_layout.addWidget(self.lcd_range, 2, 1)

        status_layout.addWidget(QtWidgets.QLabel("LiDAR Strength:"), 3, 0)
        self.lcd_strength = self.create_lcd()
        status_layout.addWidget(self.lcd_strength, 3, 1)

        controls_layout.addStretch()

        # Timers for updating displays
        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_displays)
        self.status_timer.start(100)  # Update 10 times per second

    def create_lcd(self):
        lcd = QtWidgets.QLCDNumber()
        lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        lcd.setDigitCount(6)
        return lcd

    def background_scan(self):
        print("[GUI] Triggering background scan")
        self.shared_data["scan_trigger"].value = True

    def toggle_follow_mode(self, checked):
        """NEW: Toggles the autonomous drone following mode."""
        self.shared_data["follow_drone_enabled"].value = checked
        if checked:
            print("[GUI] Autonomous drone following ENABLED.")
            # Disable manual controls while following
            self.btn_go.setEnabled(False)
            self.btn_background.setEnabled(False)
        else:
            print("[GUI] Autonomous drone following DISABLED.")
            # Re-enable manual controls
            self.btn_go.setEnabled(True)
            self.btn_background.setEnabled(True)

    def on_go_clicked(self):
        try:
            az = float(self.az_input.text())
            el = float(self.el_input.text())
            self.shared_data["target_azimuth"].value = az
            self.shared_data["target_elevation"].value = el
            self.shared_data["go_to_target"].value = True
        except ValueError:
            print("Invalid input: please enter numeric values")

    def set_tilt_up(self): self.shared_data['tilt_up'].value = True
    def set_tilt_down(self): self.shared_data['tilt_down'].value = True
    def set_pan_left(self): self.shared_data['pan_left'].value = True
    def set_pan_right(self): self.shared_data['pan_right'].value = True
    
    def update_displays(self):
        # LiDAR data
        self.lcd_range.display(self.shared_data["lidar_data"][0])
        self.lcd_strength.display(self.shared_data["lidar_data"][1])
        # Motor angles
        self.lcd_pan.display(f"{self.shared_data['stepper_degrees'].value:.1f}")
        self.lcd_tilt.display(f"{self.shared_data['servo_degrees'].value:.1f}")

    def Pshutdown(self):
        print("Shutting down from GUI...")
        self.shared_data["shutdown"].value = True
        # A short delay can help ensure the flag propagates before the app quits
        QtCore.QTimer.singleShot(500, QtWidgets.QApplication.instance().quit)

    def closeEvent(self, event):
        """Ensure shutdown is triggered when the window is closed."""
        self.Pshutdown()
        event.accept()

# Global variable to hold shared data
_shared_data = None

def run_gui(shared, movement_queue):
    """Entry point for the GUI process."""
    global _shared_data
    _shared_data = shared

    app = QtWidgets.QApplication(sys.argv)
    window = TrackerWindow(_shared_data, movement_queue)
    window.show()
    sys.exit(app.exec_())
