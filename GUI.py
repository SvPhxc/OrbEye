# GUI.py

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QCheckBox
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class TrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, shared_data):
        super().__init__()
        self.setWindowTitle("Martin Systems - Lidar Tracker GUI")
        # Bigger window to fit heatmap under 3D
        self.resize(1500, 850)

        self.shared_data = shared_data
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # ===== Left panel: 3D view (top) + 2D heatmap (bottom) =====
        left_panel = QtWidgets.QWidget()
        left_vbox = QtWidgets.QVBoxLayout(left_panel)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(6)

        # 3D View Setup
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30)
        left_vbox.addWidget(self.view, stretch=3)

        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
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

        # === 2D Heatmap (Pan vs Tilt from shared 'satellite_points') ===
        # Heatmap bins: rows=tilt 0..90, cols=pan 0..359 (no transpose needed)
        self.hm_bins = np.zeros((91, 360), dtype=np.float32)
        self.hm_widget = pg.GraphicsLayoutWidget()
        left_vbox.addWidget(self.hm_widget, stretch=2)

        self.hm_plot = self.hm_widget.addPlot(title="Pan/Tilt Heatmap")
        self.hm_plot.setLabel('bottom', 'Pan (deg)')
        self.hm_plot.setLabel('left', 'Tilt (deg)')
        self.hm_plot.setLimits(xMin=0, xMax=360, yMin=0, yMax=90)
        self.hm_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hm_plot.setMouseEnabled(x=False, y=False)

        self.hm_img = pg.ImageItem()
        self.hm_plot.addItem(self.hm_img)

        # IMPORTANT: initialize image BEFORE setting rect so width/height are known
        # Flip vertically so tilt increases upward (row 0 at bottom visually)
        self.hm_img.setImage(self.hm_bins[::-1, :], autoLevels=False)
        # Map image pixels directly to degree space
        try:
            self.hm_img.setRect(QtCore.QRectF(0, 0, 360, 91))
        except Exception:
            # Fallback for older pyqtgraph: use transform if setRect is problematic
            self.hm_img.resetTransform()
            sx = 360.0 / self.hm_bins.shape[1]  # 360 / 360
            sy = 91.0 / self.hm_bins.shape[0]   # 91 / 91
            self.hm_img.scale(sx, sy)
            self.hm_img.setPos(0, 0)

        # Colormap (fallback to grayscale if not available)
        try:
            cm = pg.colormap.get('inferno')
            self.hm_img.setLookupTable(cm.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        self.hm_img.setLevels((0.0, 1.0))
        self._hm_last_ts = -1.0  # to avoid double-counting same sample

        # Add left panel to main layout
        main_layout.addWidget(left_panel, stretch=3)

        # ===== Right: Controls Layout =====
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)

        controls_layout.addWidget(QtWidgets.QLabel("CONTROLLER STATUS"))
        self.status_label = QtWidgets.QLabel("Status: IDLE")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #808080;")
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.create_separator())

        mode_box = QtWidgets.QGroupBox("Mode Control")
        mode_layout = QVBoxLayout()
        self.btn_scan = QPushButton("Start/Stop Background Scan")
        self.btn_scan.setCheckable(True)
        self.btn_scan.clicked.connect(self.on_scan_toggled)
        mode_layout.addWidget(self.btn_scan)

        # Show/Hide Background button
        self.btn_show_background = QPushButton("Show/Hide Background Data")
        self.btn_show_background.clicked.connect(self.toggle_background_plot)
        mode_layout.addWidget(self.btn_show_background)

        self.chk_hf_tracking = QCheckBox("Enable High-Frequency Tracking")
        self.chk_hf_tracking.toggled.connect(self.on_hf_track_toggled)
        mode_layout.addWidget(self.chk_hf_tracking)

        # Clear Heatmap button
        self.btn_clear_heat = QPushButton("Clear Heatmap")
        self.btn_clear_heat.clicked.connect(self.clear_heatmap)
        mode_layout.addWidget(self.btn_clear_heat)

        mode_box.setLayout(mode_layout)
        controls_layout.addWidget(mode_box)

        manual_box = QtWidgets.QGroupBox("Manual Control")
        manual_layout = QVBoxLayout()
        manual_layout.addWidget(QtWidgets.QLabel("Target Azimuth (°)"))
        self.az_input = QtWidgets.QLineEdit("90")
        manual_layout.addWidget(self.az_input)
        manual_layout.addWidget(QtWidgets.QLabel("Target Elevation (°)"))
        self.el_input = QtWidgets.QLineEdit("45")
        manual_layout.addWidget(self.el_input)
        self.btn_go = QPushButton("Go To Position")
        self.btn_go.clicked.connect(self.on_go_clicked)
        manual_layout.addWidget(self.btn_go)

        # D-pad layout for arrow controls
        dpad_layout = QtWidgets.QGridLayout()
        btn_up = QtWidgets.QPushButton("↑")
        btn_down = QtWidgets.QPushButton("↓")
        btn_left = QtWidgets.QPushButton("←")
        btn_right = QtWidgets.QPushButton("→")
        dpad_layout.addWidget(btn_up, 0, 1)
        dpad_layout.addWidget(btn_left, 1, 0)
        dpad_layout.addWidget(btn_right, 1, 2)
        dpad_layout.addWidget(btn_down, 2, 1)
        manual_layout.addLayout(dpad_layout)

        # Connect new buttons
        btn_up.clicked.connect(self.set_tilt_up)
        btn_down.clicked.connect(self.set_tilt_down)
        btn_left.clicked.connect(self.set_pan_left)
        btn_right.clicked.connect(self.set_pan_right)

        manual_box.setLayout(manual_layout)
        controls_layout.addWidget(manual_box)

        ekf_box = QtWidgets.QGroupBox("EKF Control")
        ekf_layout = QVBoxLayout()
        self.btn_acquire = QPushButton("Acquire Target (3 pts)")
        self.btn_acquire.clicked.connect(self.on_acquire_clicked)
        ekf_layout.addWidget(self.btn_acquire)
        self.btn_stop_ekf = QPushButton("Stop Tracking & Generate Plot")
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

        self.chk_reactive_mode = QCheckBox("Enable Reactive Mode (No Prediction)")
        self.chk_reactive_mode.toggled.connect(self.on_reactive_mode_toggled)
        controls_layout.addWidget(self.chk_reactive_mode)

        controls_layout.addStretch()

        self.btn_shutdown = QPushButton("Shutdown")
        self.btn_shutdown.setStyleSheet("background-color: #C41E3A; color: white; font-weight: bold;")
        self.btn_shutdown.clicked.connect(self.Pshutdown)
        controls_layout.addWidget(self.btn_shutdown)

        # Timers
        self.timer_ui = QtCore.QTimer(self)
        self.timer_ui.timeout.connect(self.update_ui)
        self.timer_ui.start(100)

    # ===== Heatmap helpers =====
    def clear_heatmap(self):
        self.hm_bins[...] = 0.0
        # Refresh display to show empty map
        self.hm_img.setImage(self.hm_bins[::-1, :], autoLevels=True)
        self._hm_last_ts = -1.0

    def update_pan_tilt_heatmap(self):
        """
        Read [az, el, dist_cm, strength, ts] from shared_data['satellite_points']
        and accumulate into a 2D heatmap: X=pan°, Y=tilt°.
        """
        sp = self.shared_data.get("satellite_points")
        if sp is None:
            return

        try:
            az_deg = float(sp[0])
            el_deg = float(sp[1])
            dist_cm = float(sp[2])
            strength = float(sp[3])
            ts = float(sp[4])
        except Exception:
            return

        # Skip duplicate sample (same timestamp)
        if ts <= self._hm_last_ts:
            return

        # Validity checks
        if not (0.0 <= az_deg < 360.0):
            self._hm_last_ts = ts
            return
        if not (0.0 <= el_deg <= 90.0):
            self._hm_last_ts = ts
            return
        if not (10.0 < dist_cm < 16000.0):  # ignore garbage distances
            self._hm_last_ts = ts
            return

        # Bin by integer degrees (change to finer bins if needed)
        pan_bin = int(round(az_deg)) % 360
        tilt_bin = int(round(el_deg))

        # Weight by strength lightly (or set to 1.0)
        w = max(1.0, strength / 100.0)
        self.hm_bins[tilt_bin, pan_bin] += w

        # Optional gentle time decay so old hits fade (uncomment to enable)
        # self.hm_bins *= 0.9995

        # Normalize and update image; flip vertically so tilt increases upward
        mx = float(self.hm_bins.max())
        if mx <= 0.0:
            mx = 1.0
        img = (self.hm_bins / mx)[::-1, :]  # shape: (91, 360); x=pan, y=tilt
        self.hm_img.setImage(img, autoLevels=False)
        self.hm_img.setLevels((0.0, 1.0))

        self._hm_last_ts = ts

    # ===== Controls handlers =====
    def on_reactive_mode_toggled(self, checked):
        self.shared_data["demo"].value = checked

    def toggle_background_plot(self):
        """Loads data from file and displays/hides the plot."""
        if self.background_plot.visible():
            self.background_plot.hide()
            print("[GUI] Background visualization hidden.")
            return

        try:
            bg_data_path_obj = self.shared_data.get("background_path", None)
            if bg_data_path_obj is None:
                bg_data_path = "background_data.npy"
            else:
                bg_data_path = bg_data_path_obj.value

            bg_data = np.load(bg_data_path)

            points = []
            # Expected rows: [azimuth, elevation, distance_cm, strength]
            for az, el, dist_cm, strength in bg_data:
                if 10 < dist_cm < 16000:
                    az_rad = np.radians(az)
                    el_rad = np.radians(el)
                    dist_m = dist_cm / 100.0  # cm -> m

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
            print(f"[GUI] Error: '{bg_data_path}' not found. Please run a background scan first.")
        except Exception as e:
            print(f"[GUI] An error occurred while loading or processing background data: {e}")

    def set_tilt_up(self):
        self.shared_data['tilt_up'].value = True

    def set_tilt_down(self):
        self.shared_data['tilt_down'].value = True

    def set_pan_left(self):
        self.shared_data['pan_left'].value = True

    def set_pan_right(self):
        self.shared_data['pan_right'].value = True

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

    def on_acquire_clicked(self):
        self.shared_data["acquire_points"].value = True

    def on_scan_toggled(self, checked):
        self.shared_data["background_scan_active"].value = checked

    def on_hf_track_toggled(self, checked):
        self.shared_data["lidar_track_mode_active"].value = checked

    def stop_and_plot_ekf(self):
        self.shared_data["generate_plot_on_stop"].value = True
        self.shared_data["lidar_track_mode_active"].value = False

    def on_debug_mode_toggled(self, en):
        self.shared_data["debug_mode"].value = bool(en)
        self.shared_data["lidar_acceptance_range"][:] = [0.2, 2.0] if en else [3.0, 50.0]

    def on_go_clicked(self):
        try:
            self.shared_data["target_azimuth"].value = float(self.az_input.text())
            self.shared_data["target_elevation"].value = float(self.el_input.text())
            self.shared_data["go_to_target"].value = True
        except ValueError:
            print("[GUI] Invalid go-to coordinates.")

    def update_ui(self):
        # Update LCDs
        try:
            self.lcd_pan.display(f"{self.shared_data['stepper_degrees'].value:.2f}")
            self.lcd_tilt.display(f"{self.shared_data['servo_degrees'].value:.2f}")
            self.lcd_range.display(self.shared_data['lidar_data'][0])
            self.lcd_strength.display(self.shared_data['lidar_data'][1])
        except Exception:
            pass

        # Update Status Label
        try:
            if self.shared_data["acquirer_status"].value == 1:
                self.status_label.setText("Status: ACQUIRING...")
                self.status_label.setStyleSheet("color: #FFA500;")
            elif self.shared_data["lidar_track_mode_active"].value:
                self.status_label.setText("Status: TRACKING")
                self.status_label.setStyleSheet("color: #D22B2B;")
            elif self.shared_data["background_scan_active"].value:
                self.status_label.setText("Status: SCANNING...")
                self.status_label.setStyleSheet("color: #007FFF;")
            elif self.shared_data["go_to_target"].value:
                status_text = "MOVING" if not self.shared_data["target_reached"].value else "HOLDING"
                self.status_label.setText(f"Status: {status_text}")
                self.status_label.setStyleSheet("color: #33F;")
            else:
                self.status_label.setText("Status: IDLE")
                self.status_label.setStyleSheet("color: #808080;")
        except Exception:
            pass

        # Update 3D view (laser & satellite tip)
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
        except Exception:
            pass

        # Update 2D heatmap from shared_data['satellite_points']
        self.update_pan_tilt_heatmap()

    def Pshutdown(self):
        print("[GUI] Shutdown requested.")
        self.shared_data["shutdown"].value = True
        self.timer_ui.stop()
        # Give main a moment to process the shutdown flag before quitting the app
        QtCore.QTimer.singleShot(250, QtWidgets.QApplication.instance().quit)

    def closeEvent(self, event):
        self.Pshutdown()
        super().closeEvent(event)


def run_gui(shared_data):
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    window = TrackerWindow(shared_data)
    window.show()
    sys.exit(app.exec_())
