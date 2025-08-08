# GUI integration code - add this to your GUI module

import tkinter as tk
from tkinter import ttk
import threading
import time


class EnhancedAcquisitionGUI:
    def __init__(self, parent_frame, shared_data):
        self.shared_data = shared_data
        self.acquisition_frame = ttk.LabelFrame(parent_frame, text="Enhanced Drone Acquisition", padding="10")
        self.acquisition_frame.pack(fill="x", padx=5, pady=5)

        # Status display
        self.status_var = tk.StringVar(value="Ready for acquisition")
        self.status_label = ttk.Label(self.acquisition_frame, textvariable=self.status_var,
                                      font=("Arial", 10, "bold"))
        self.status_label.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.acquisition_frame, mode='determinate', length=300)
        self.progress.pack(pady=5)

        # Control buttons
        button_frame = ttk.Frame(self.acquisition_frame)
        button_frame.pack(fill="x", pady=5)

        self.start_btn = ttk.Button(button_frame, text="Start 3-Point Acquisition",
                                    command=self.start_acquisition)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Acquisition",
                                   command=self.stop_acquisition, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        # Settings frame
        settings_frame = ttk.LabelFrame(self.acquisition_frame, text="Acquisition Settings")
        settings_frame.pack(fill="x", pady=5)

        # Distance range settings
        ttk.Label(settings_frame, text="Distance Range (m):").grid(row=0, column=0, sticky="w")
        self.min_distance = tk.DoubleVar(value=6.0)
        self.max_distance = tk.DoubleVar(value=12.0)

        ttk.Scale(settings_frame, from_=3.0, to=15.0, variable=self.min_distance,
                  orient="horizontal", length=100).grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.min_distance).grid(row=0, column=2)

        ttk.Scale(settings_frame, from_=6.0, to=20.0, variable=self.max_distance,
                  orient="horizontal", length=100).grid(row=0, column=3, padx=5)
        ttk.Label(settings_frame, textvariable=self.max_distance).grid(row=0, column=4)

        # Strength threshold
        ttk.Label(settings_frame, text="Min Strength:").grid(row=1, column=0, sticky="w")
        self.min_strength = tk.IntVar(value=5000)
        ttk.Scale(settings_frame, from_=3000, to=15000, variable=self.min_strength,
                  orient="horizontal", length=200).grid(row=1, column=1, columnspan=3, padx=5)
        ttk.Label(settings_frame, textvariable=self.min_strength).grid(row=1, column=4)

        # Points display
        self.points_frame = ttk.LabelFrame(self.acquisition_frame, text="Acquired Points")
        self.points_frame.pack(fill="x", pady=5)

        self.points_text = tk.Text(self.points_frame, height=6, width=50)
        self.points_text.pack(padx=5, pady=5)

        # EKF status
        ekf_frame = ttk.LabelFrame(self.acquisition_frame, text="EKF Status")
        ekf_frame.pack(fill="x", pady=5)

        self.ekf_status = tk.StringVar(value="EKF Not Started")
        ttk.Label(ekf_frame, textvariable=self.ekf_status).pack(side="left", padx=5)

        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        ttk.Label(ekf_frame, textvariable=self.confidence_var).pack(side="right", padx=5)

        # Start status monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_acquisition, daemon=True)
        self.monitor_thread.start()

    def start_acquisition(self):
        """Start the enhanced 3-point acquisition process"""
        # Update distance range in shared data
        self.shared_data["lidar_acceptance_range"][0] = self.min_distance.get()
        self.shared_data["lidar_acceptance_range"][1] = self.max_distance.get()

        # Reset acquisition state
        self.shared_data["points_count"].value = 0
        self.shared_data["ekf_start"].value = False
        self.shared_data["ekf_running"].value = False

        # Trigger acquisition
        self.shared_data["acquire_points"].value = True

        # Update UI
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Starting acquisition...")
        self.progress.config(value=0)
        self.points_text.delete(1.0, tk.END)

        print("🎯 3-Point acquisition started from GUI")

    def stop_acquisition(self):
        """Stop the acquisition process"""
        self.shared_data["acquire_points"].value = False
        self.shared_data["ekf_running"].value = False

        # Update UI
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Acquisition stopped")
        self.progress.config(value=0)

        print("🛑 Acquisition stopped from GUI")

    def monitor_acquisition(self):
        """Monitor acquisition progress and update GUI"""
        while self.monitoring:
            try:
                # Check acquisition status
                acquiring = self.shared_data["acquire_points"].value
                points_count = self.shared_data["points_count"].value
                ekf_running = self.shared_data["ekf_running"].value
                ekf_initialized = self.shared_data["ekf_initialized"].value

                # Update progress
                if acquiring:
                    if points_count == 0:
                        self.status_var.set("Phase 1: Finding initial detection...")
                        self.progress.config(value=10)
                    elif points_count == 1:
                        self.status_var.set("Phase 2: Estimating motion vector...")
                        self.progress.config(value=40)
                    elif points_count == 2:
                        self.status_var.set("Phase 3: Acquiring predictive point...")
                        self.progress.config(value=70)
                    else:
                        self.status_var.set("Finalizing acquisition...")
                        self.progress.config(value=90)
                elif points_count >= 3 and not ekf_running:
                    self.status_var.set("3 points acquired - initializing EKF...")
                    self.progress.config(value=95)
                elif ekf_running:
                    self.status_var.set("EKF tracking active")
                    self.progress.config(value=100)
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                elif not acquiring and points_count == 0:
                    self.status_var.set("Ready for acquisition")
                    self.progress.config(value=0)
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")

                # Update points display
                if points_count > 0:
                    self.update_points_display()

                # Update EKF status
                if ekf_initialized:
                    confidence = self.shared_data["ekf_confidence"].value
                    self.ekf_status.set("EKF Running" if ekf_running else "EKF Initialized")
                    self.confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
                else:
                    self.ekf_status.set("EKF Not Started")
                    self.confidence_var.set("Confidence: 0%")

                time.sleep(0.5)  # Update every 0.5 seconds

            except Exception as e:
                print(f"GUI monitoring error: {e}")
                time.sleep(1.0)

    def update_points_display(self):
        """Update the points display with current acquired points"""
        try:
            points_buffer = self.shared_data["points_buffer"]
            points_count = self.shared_data["points_count"].value

            self.points_text.delete(1.0, tk.END)
            self.points_text.insert(tk.END, f"Acquired Points ({points_count}/3):\n\n")

            for i in range(min(points_count, 3)):
                base_idx = i * 4
                az = points_buffer[base_idx + 0]
                el = points_buffer[base_idx + 1]
                dist = points_buffer[base_idx + 2]
                strength = points_buffer[base_idx + 3]

                self.points_text.insert(tk.END,
                                        f"Point {i + 1}:\n"
                                        f"  Azimuth: {az:.2f}°\n"
                                        f"  Elevation: {el:.2f}°\n"
                                        f"  Distance: {dist:.2f}m\n"
                                        f"  Strength: {strength:.0f}\n\n")

        except Exception as e:
            print(f"Error updating points display: {e}")


class DroneTrackingGUI:
    """Complete GUI for drone tracking system"""

    def __init__(self, root, shared_data):
        self.root = root
        self.shared_data = shared_data
        self.root.title("Enhanced Drone Tracking System")
        self.root.geometry("800x900")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Acquisition tab
        self.acq_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.acq_frame, text="Acquisition")
        self.acquisition_gui = EnhancedAcquisitionGUI(self.acq_frame, shared_data)

        # Tracking tab
        self.track_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.track_frame, text="Tracking")
        self.create_tracking_tab()

        # System tab
        self.system_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.system_frame, text="System")
        self.create_system_tab()

    def create_tracking_tab(self):
        """Create tracking control and display tab"""
        # Current position display
        pos_frame = ttk.LabelFrame(self.track_frame, text="Current Position", padding="10")
        pos_frame.pack(fill="x", padx=5, pady=5)

        self.current_az = tk.StringVar(value="0.0°")
        self.current_el = tk.StringVar(value="90.0°")

        ttk.Label(pos_frame, text="Azimuth:").grid(row=0, column=0, sticky="w")
        ttk.Label(pos_frame, textvariable=self.current_az).grid(row=0, column=1, sticky="w")
        ttk.Label(pos_frame, text="Elevation:").grid(row=1, column=0, sticky="w")
        ttk.Label(pos_frame, textvariable=self.current_el).grid(row=1, column=1, sticky="w")

        # Manual control
        manual_frame = ttk.LabelFrame(self.track_frame, text="Manual Control", padding="10")
        manual_frame.pack(fill="x", padx=5, pady=5)

        # Direction buttons
        btn_frame = ttk.Frame(manual_frame)
        btn_frame.pack()

        ttk.Button(btn_frame, text="↑", command=self.tilt_up).grid(row=0, column=1)
        ttk.Button(btn_frame, text="←", command=self.pan_left).grid(row=1, column=0)
        ttk.Button(btn_frame, text="→", command=self.pan_right).grid(row=1, column=2)
        ttk.Button(btn_frame, text="↓", command=self.tilt_down).grid(row=2, column=1)

        # Go to target
        target_frame = ttk.Frame(manual_frame)
        target_frame.pack(fill="x", pady=10)

        ttk.Label(target_frame, text="Target Az:").grid(row=0, column=0)
        self.target_az_var = tk.DoubleVar(value=0.0)
        ttk.Entry(target_frame, textvariable=self.target_az_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(target_frame, text="Target El:").grid(row=0, column=2)
        self.target_el_var = tk.DoubleVar(value=90.0)
        ttk.Entry(target_frame, textvariable=self.target_el_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Button(target_frame, text="Go To Target", command=self.go_to_target).grid(row=0, column=4, padx=10)

        # EKF predictions display
        pred_frame = ttk.LabelFrame(self.track_frame, text="EKF Predictions", padding="10")
        pred_frame.pack(fill="x", padx=5, pady=5)

        self.pred_az = tk.StringVar(value="N/A")
        self.pred_el = tk.StringVar(value="N/A")
        self.est_az = tk.StringVar(value="N/A")
        self.est_el = tk.StringVar(value="N/A")

        ttk.Label(pred_frame, text="Predicted Az:").grid(row=0, column=0, sticky="w")
        ttk.Label(pred_frame, textvariable=self.pred_az).grid(row=0, column=1, sticky="w")
        ttk.Label(pred_frame, text="Predicted El:").grid(row=1, column=0, sticky="w")
        ttk.Label(pred_frame, textvariable=self.pred_el).grid(row=1, column=1, sticky="w")

        ttk.Label(pred_frame, text="Estimated Az:").grid(row=0, column=2, sticky="w", padx=20)
        ttk.Label(pred_frame, textvariable=self.est_az).grid(row=0, column=3, sticky="w")
        ttk.Label(pred_frame, text="Estimated El:").grid(row=1, column=2, sticky="w", padx=20)
        ttk.Label(pred_frame, textvariable=self.est_el).grid(row=1, column=3, sticky="w")

        # Start tracking monitoring
        self.start_tracking_monitor()

    def create_system_tab(self):
        """Create system status and control tab"""
        # Background scan control
        scan_frame = ttk.LabelFrame(self.system_frame, text="Background Scan", padding="10")
        scan_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(scan_frame, text="Start Background Scan",
                   command=self.start_background_scan).pack(side="left", padx=5)

        # LiDAR data display
        lidar_frame = ttk.LabelFrame(self.system_frame, text="LiDAR Data", padding="10")
        lidar_frame.pack(fill="x", padx=5, pady=5)

        self.lidar_distance = tk.StringVar(value="N/A")
        self.lidar_strength = tk.StringVar(value="N/A")

        ttk.Label(lidar_frame, text="Distance:").grid(row=0, column=0, sticky="w")
        ttk.Label(lidar_frame, textvariable=self.lidar_distance).grid(row=0, column=1, sticky="w")
        ttk.Label(lidar_frame, text="Strength:").grid(row=1, column=0, sticky="w")
        ttk.Label(lidar_frame, textvariable=self.lidar_strength).grid(row=1, column=1, sticky="w")

        # System control
        control_frame = ttk.LabelFrame(self.system_frame, text="System Control", padding="10")
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Shutdown System",
                   command=self.shutdown_system).pack(pady=10)

    def start_tracking_monitor(self):
        """Monitor tracking data and update display"""

        def monitor():
            while True:
                try:
                    # Update current position
                    az = self.shared_data["stepper_degrees"].value
                    el = self.shared_data["servo_degrees"].value
                    self.current_az.set(f"{az:.2f}°")
                    self.current_el.set(f"{el:.2f}°")

                    # Update EKF predictions
                    if self.shared_data["ekf_initialized"].value:
                        pred_az = self.shared_data["predicted_azimuth"].value
                        pred_el = self.shared_data["predicted_elevation"].value
                        est_az = self.shared_data["estimated_azimuth"].value
                        est_el = self.shared_data["estimated_elevation"].value

                        self.pred_az.set(f"{pred_az:.2f}°")
                        self.pred_el.set(f"{pred_el:.2f}°")
                        self.est_az.set(f"{est_az:.2f}°")
                        self.est_el.set(f"{est_el:.2f}°")

                    # Update LiDAR data
                    with self.shared_data["lidar_data"].get_lock():
                        distance = self.shared_data["lidar_data"][0]
                        strength = self.shared_data["lidar_data"][1]

                    self.lidar_distance.set(f"{distance:.1f} cm")
                    self.lidar_strength.set(f"{strength:.0f}")

                    time.sleep(0.2)  # Update 5 times per second

                except Exception as e:
                    print(f"Tracking monitor error: {e}")
                    time.sleep(1.0)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    # Control methods
    def tilt_up(self):
        self.shared_data["tilt_up"].value = True

    def tilt_down(self):
        self.shared_data["tilt_down"].value = True

    def pan_left(self):
        self.shared_data["pan_left"].value = True

    def pan_right(self):
        self.shared_data["pan_right"].value = True

    def go_to_target(self):
        self.shared_data["target_azimuth"].value = self.target_az_var.get()
        self.shared_data["target_elevation"].value = self.target_el_var.get()
        self.shared_data["go_to_target"].value = True

    def start_background_scan(self):
        self.shared_data["scan_trigger"].value = True

    def shutdown_system(self):
        self.shared_data["shutdown"].value = True
        self.root.quit()


# Main GUI function to replace your existing run_gui
def run_gui(shared_data, movement_queue):
    """Enhanced GUI with 3-point acquisition system"""
    print("[GUI] Starting enhanced GUI...")

    root = tk.Tk()
    app = DroneTrackingGUI(root, shared_data)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        shared_data["shutdown"].value = True

    print("[GUI] GUI shutting down...")