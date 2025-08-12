import math
import time
import threading
import numpy as np
from enum import Enum
from collections import deque


class TrackingState(Enum):
    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"
    ACQUIRING = "acquiring"


class IntegratedAdaptiveTracker:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.min_target_distance = 20 # cm (e.g., ignore targets closer than 20cm)
        self.max_target_distance = 150 # cm (e.g., ignore targets farther than 1.5m)
        # Target tracking variables
        self.state = TrackingState.SEARCHING
        self.target_history = deque(maxlen=10)  # Store last 10 target positions
        self.last_detection_time = 0
        self.tracking_confidence = 0
        self.search_radius = 30.0  # Initial search radius in degrees
        self.max_search_radius = 60.0
        self.shared_data["go_to_target"].value = True
        # Motion prediction
        self.angular_velocity_pan = 0.0  # degrees/second
        self.angular_velocity_tilt = 0.0
        self.prediction_horizon = 0.15  # seconds ahead to predict
        
        # Detection parameters
        self.detection_threshold = 9000  # LiDAR strength threshold
        self.min_detections_for_track = 3
        self.max_time_without_detection = 1.5  # seconds
        
        # Search pattern variables
        self.search_center_pan = 0
        self.search_center_tilt = 45
        self.search_angle = 0  # Current angle in spiral search
        self.search_start_time = 0
        
        # Tracking thread control
        self.tracking_thread = None
        self.tracking_active = threading.Event()
        self.shutdown_tracker = threading.Event()
        
        # Statistics
        self.total_detections = 0
        self.track_start_time = 0
        
        # Coordination state - tracks what we've requested vs what's happening
        self.last_target_pan = None
        self.last_target_tilt = None
        self.last_position_read_time = 0
        self.mode_request_time = 0
        self.waiting_for_mode_change = False
        
        print("[AdaptiveTracker] Initialized with race-condition-free coordination")
        
    def _is_system_ready_for_tracking(self):
        """Check if the system is in a state where we can safely request tracking mode."""
        # Don't interfere if hardware is busy with critical operations
        if self.shared_data["stepper_busy"].value:
            return False
            
        # Don't interfere if other high-priority modes are active
        if self.shared_data["background_scan_active"].value:
            return False
            
        if self.shared_data["go_to_target"].value:
            return False
            
        # System shutdown requested
        if self.shared_data["shutdown"].value:
            return False
            
        return True
    
    def _request_tracking_mode(self):
        """Safely request tracking mode through the proper flag hierarchy."""
        if not self._is_system_ready_for_tracking():
            return False
            
        # Request adaptive tracking mode (main process will coordinate)
        if not self.shared_data["adaptive_tracking_active"].value:
            print("[AdaptiveTracker] Requesting adaptive tracking mode activation")
            self.shared_data["adaptive_tracking_active"].value = True
            self.mode_request_time = time.time()
            self.waiting_for_mode_change = True
        
        # Wait for hardware controller to acknowledge by setting lidar_track_mode_active
        # This creates a handshake: we request -> main sets adaptive_tracking_active -> HW sets lidar_track_mode_active
        if self.waiting_for_mode_change:
            # Give the system time to respond (up to 1 second)
            if time.time() - self.mode_request_time > 1.0:
                print("[AdaptiveTracker] Mode request timeout - system may be busy")
                self.waiting_for_mode_change = False
                return False
                
            # Check if hardware controller has acknowledged
            if self.shared_data["lidar_track_mode_active"].value:
                print("[AdaptiveTracker] Tracking mode confirmed by hardware controller")
                self.waiting_for_mode_change = False
                return True
            else:
                return False  # Still waiting
        
        # Already in tracking mode
        return self.shared_data["lidar_track_mode_active"].value
    
    def _release_tracking_mode(self):
        """Safely release tracking mode."""
        print("[AdaptiveTracker] Releasing tracking mode")
        # Signal that we no longer need adaptive tracking
        self.shared_data["adaptive_tracking_active"].value = False
        # Hardware controller will see this and disable lidar_track_mode_active
        self.waiting_for_mode_change = False
        
    def start_tracking(self):
        """Start the adaptive tracking process."""
        if self.tracking_thread and self.tracking_thread.is_alive():
            print("[AdaptiveTracker] Already running")
            return
            
        print("[AdaptiveTracker] Starting adaptive tracking")
        self.shutdown_tracker.clear()
        self.tracking_active.set()
        
        # Reset tracking state
        self.state = TrackingState.SEARCHING
        self.target_history.clear()
        self.tracking_confidence = 0
        self.search_angle = 0
        self.search_start_time = time.time()
        self.track_start_time = time.time()
        self.total_detections = 0
        self.waiting_for_mode_change = False
        
        # Get initial search center safely
        self._safe_get_initial_position()
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
    def _safe_get_initial_position(self):
        """Safely get initial position for search center."""
        # Wait for hardware to not be busy before reading position
        timeout = time.time() + 2.0
        while self.shared_data["stepper_busy"].value and time.time() < timeout:
            time.sleep(0.1)
            
        # Set initial search center to current position (if available)
        if not self.shared_data["stepper_busy"].value:
            self.search_center_pan = self.shared_data["stepper_degrees"].value
            self.search_center_tilt = max(30, min(60, self.shared_data["servo_degrees"].value))
        else:
            # Fallback to safe default if we can't read position
            print("[AdaptiveTracker] Warning: Could not read initial position, using defaults")
            self.search_center_pan = 0
            self.search_center_tilt = 45
        
    def stop_tracking(self):
        """Stop the adaptive tracking process."""
        print("[AdaptiveTracker] Stopping adaptive tracking")
        self.tracking_active.clear()
        self.shutdown_tracker.set()
        
        # Release tracking mode through proper channels
        self._release_tracking_mode()
        
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)
            
    def _get_current_lidar_data(self):
        """Get current LiDAR data from shared memory."""
        with self.shared_data["lidar_data"].get_lock():
            return list(self.shared_data["lidar_data"][:])
            
    def _safe_get_current_position(self):
        """Safely get current motor positions - respects hardware state."""
        # Don't read position if hardware is busy to avoid inconsistent reads
        if self.shared_data["stepper_busy"].value:
            # Return last known position if hardware is busy
            if hasattr(self, '_last_known_pan') and hasattr(self, '_last_known_tilt'):
                return self._last_known_pan, self._last_known_tilt
            else:
                return 0, 45  # Safe fallback
                
        # Read position and cache it
        current_pan = self.shared_data["stepper_degrees"].value
        current_tilt = self.shared_data["servo_degrees"].value
        
        # Cache for use when hardware is busy
        self._last_known_pan = current_pan
        self._last_known_tilt = current_tilt
        self.last_position_read_time = time.time()
        
        return current_pan, current_tilt
        
    def _safe_set_target_position(self, pan_deg, tilt_deg):
        """Safely set target position - respects hardware state and coordination."""
        # Don't send commands if we don't have tracking mode
        if not self.shared_data["lidar_track_mode_active"].value:
            return False
            
        # Don't send commands if hardware is busy
        if self.shared_data["stepper_busy"].value:
            return False
            
        # Don't send same target repeatedly (reduces command spam)
        if (self.last_target_pan == pan_deg and 
            self.last_target_tilt == tilt_deg):
            return True  # Already sent this target
            
        # Clamp tilt to valid servo range
        tilt_deg = max(0, min(90, tilt_deg))
        
        # Set predicted positions for hardware controller
        self.shared_data["target_azimuth"].value = pan_deg
        self.shared_data["target_elevation"].value = tilt_deg
        self.shared_data["go_to_target"].value = True
        
        # Cache what we sent
        self.last_target_pan = pan_deg
        self.last_target_tilt = tilt_deg
        
        return True
        
    def _update_target_position(self, lidar_data, current_time):
        """Update target position from LiDAR data."""
        if len(lidar_data) < 3:
            return
            
        distance, strength, lidar_timestamp = lidar_data
        current_pan, current_tilt = self._safe_get_current_position()
        
        if strength >= self.detection_threshold and self.min_target_distance <= distance <= self.max_target_distance:
            # Valid target detection
            target_pos = {
                'pan': current_pan,
                'tilt': current_tilt,
                'distance': distance,
                'timestamp': current_time,
                'strength': strength
            }
            
            self.target_history.append(target_pos)
            self.last_detection_time = current_time
            self.total_detections += 1
            
            # Calculate angular velocity if we have enough history
            if len(self.target_history) >= 2:
                self._calculate_angular_velocity()
            
            # Update tracking confidence
            self.tracking_confidence = min(100, self.tracking_confidence + 15)
            
            # Update state based on detections
            if self.state == TrackingState.SEARCHING:
                if len(self.target_history) >= self.min_detections_for_track:
                    self.state = TrackingState.TRACKING
                    print(f"[AdaptiveTracker] Target acquired! Switching to tracking mode (confidence: {self.tracking_confidence}%)")
                else:
                    self.state = TrackingState.ACQUIRING
                    print(f"[AdaptiveTracker] Target detected, acquiring... ({len(self.target_history)}/{self.min_detections_for_track})")
            elif self.state == TrackingState.LOST:
                self.state = TrackingState.ACQUIRING
                print(f"[AdaptiveTracker] Target reacquired!")
                
        else:
            # No target detected - handle loss
            time_since_detection = current_time - self.last_detection_time
            
            if time_since_detection > self.max_time_without_detection:
                if self.state == TrackingState.TRACKING:
                    print(f"[AdaptiveTracker] Target lost after {time_since_detection:.1f}s! Switching to search mode.")
                    self.state = TrackingState.LOST
                    # Set search center to last known position
                    if self.target_history:
                        last_pos = self.target_history[-1]
                        self.search_center_pan = last_pos['pan']
                        self.search_center_tilt = last_pos['tilt']
                    self.search_angle = 0
                    
            # Decrease confidence gradually
            self.tracking_confidence = max(0, self.tracking_confidence - 3)
            
    def _calculate_angular_velocity(self):
        """Calculate target angular velocity from position history."""
        if len(self.target_history) < 2:
            return
            
        # Use multiple recent positions for better velocity estimation
        recent_positions = list(self.target_history)[-5:]  # Last 5 positions
        
        if len(recent_positions) >= 2:
            # Calculate average velocity over recent history
            total_time = recent_positions[-1]['timestamp'] - recent_positions[0]['timestamp']
            if total_time > 0:
                total_pan_change = recent_positions[-1]['pan'] - recent_positions[0]['pan']
                total_tilt_change = recent_positions[-1]['tilt'] - recent_positions[0]['tilt']
                
                # Handle angle wraparound for pan - be more careful with modulo issues
                if abs(total_pan_change) > 180:
                    if total_pan_change > 0:
                        total_pan_change -= 360
                    else:
                        total_pan_change += 360
                
                self.angular_velocity_pan = total_pan_change / total_time
                self.angular_velocity_tilt = total_tilt_change / total_time
                
                # Limit velocities to reasonable ranges
                self.angular_velocity_pan = max(-200, min(200, self.angular_velocity_pan))
                self.angular_velocity_tilt = max(-100, min(100, self.angular_velocity_tilt))
                
    def _predict_target_position(self, future_time):
        """Predict where the target will be at a future time."""
        if not self.target_history:
            current_pan, current_tilt = self._safe_get_current_position()
            return current_pan, current_tilt
            
        last_pos = self.target_history[-1]
        
        # Predict position based on angular velocity with damping
        velocity_damping = 0.8  # Reduce prediction aggressiveness
        predicted_pan = last_pos['pan'] + (self.angular_velocity_pan * velocity_damping * future_time)
        predicted_tilt = last_pos['tilt'] + (self.angular_velocity_tilt * velocity_damping * future_time)
        
        # Clamp tilt to valid range
        predicted_tilt = max(0, min(90, predicted_tilt))
        
        return predicted_pan, predicted_tilt
        
    def _generate_next_waypoint(self, current_time):
        """Generate the next pan/tilt waypoint based on current tracking state."""
        if self.state == TrackingState.TRACKING:
            return self._tracking_waypoint(current_time)
        elif self.state == TrackingState.ACQUIRING:
            return self._acquisition_waypoint(current_time)
        elif self.state in [TrackingState.LOST, TrackingState.SEARCHING]:
            return self._search_waypoint(current_time)
            
    def _tracking_waypoint(self, current_time):
        """Generate waypoint for active tracking mode."""
        # Predict where target will be
        pred_pan, pred_tilt = self._predict_target_position(self.prediction_horizon)
        
        # Add small adaptive offset based on tracking confidence
        confidence_factor = self.tracking_confidence / 100.0
        uncertainty_factor = (1.0 - confidence_factor) * 2.0
        
        # Use time-based pseudo-random offsets for smooth uncertainty
        time_seed = int(current_time * 10) % 100
        uncertainty_offset_pan = uncertainty_factor * math.sin(time_seed * 0.1) * 1.0
        uncertainty_offset_tilt = uncertainty_factor * math.cos(time_seed * 0.15) * 0.5
        
        next_pan = pred_pan + uncertainty_offset_pan
        next_tilt = max(0, min(90, pred_tilt + uncertainty_offset_tilt))
        
        return next_pan, next_tilt, "tracking"
        
    def _acquisition_waypoint(self, current_time):
        """Generate waypoint for target acquisition mode."""
        if self.target_history:
            # Small search pattern around last known position
            last_pos = self.target_history[-1]
            search_radius = 5.0  # degrees
            
            # Figure-8 pattern for acquisition
            time_factor = (current_time * 3) % (2 * math.pi)
            offset_pan = search_radius * math.sin(time_factor)
            offset_tilt = search_radius * 0.3 * math.sin(2 * time_factor)
            
            next_pan = last_pos['pan'] + offset_pan
            next_tilt = max(0, min(90, last_pos['tilt'] + offset_tilt))
            
            return next_pan, next_tilt, "acquiring"
        else:
            return self._search_waypoint(current_time)
            
    def _search_waypoint(self, current_time):
        """Generate waypoint for search mode using expanding spiral."""
        # Expanding spiral search
        time_since_search_start = current_time - self.search_start_time
        
        # Increase search radius over time
        radius_expansion = min(time_since_search_start * 5, self.max_search_radius)
        current_search_radius = min(self.search_radius + radius_expansion, self.max_search_radius)
        
        # Spiral parameters
        self.search_angle += 0.2  # Increment search angle
        spiral_factor = (self.search_angle / (4 * math.pi)) % 1.0  # Normalize to 0-1
        radius = current_search_radius * spiral_factor
        
        # Generate spiral coordinates
        offset_pan = radius * math.cos(self.search_angle)
        offset_tilt = radius * 0.4 * math.sin(self.search_angle)  # Limit vertical movement
        
        next_pan = self.search_center_pan + offset_pan
        next_tilt = max(10, min(80, self.search_center_tilt + offset_tilt))  # Keep reasonable tilt range
        
        # Reset spiral when we complete a revolution
        if self.search_angle > 8 * math.pi:
            self.search_angle = 0
            # Move search center slightly
            self.search_center_pan = (self.search_center_pan + 30) % 360
            
        return next_pan, next_tilt, "searching"
        
    def _tracking_loop(self):
        """Main tracking loop running in separate thread."""
        print("[AdaptiveTracker] Tracking loop started")
        loop_hz = 15  # 15 Hz update rate
        loop_period = 1.0 / loop_hz
        last_status_time = 0
        status_period = 2.0  # Print status every 2 seconds
        
        # Track mode request state
        mode_requested = False
        last_mode_check = 0
        mode_check_period = 0.5  # Check mode every 500ms
        
        while self.tracking_active.is_set() and not self.shutdown_tracker.is_set():
            loop_start = time.time()
            
            try:
                # Periodically check and request tracking mode
                if time.time() - last_mode_check > mode_check_period:
                    if not mode_requested or not self.shared_data["lidar_track_mode_active"].value:
                        mode_requested = self._request_tracking_mode()
                        #self.shared_data["lidar_track_mode_active"].value = True
                    last_mode_check = time.time()
                
                # Only process if we have tracking mode
                if self.shared_data["lidar_track_mode_active"].value:
                    # Get current LiDAR data
                    lidar_data = self._get_current_lidar_data()
                    
                    

                    current_time = time.time()
                    
                    # Update target position from LiDAR data
                    self._update_target_position(lidar_data, current_time)
                    
                    # Generate next waypoint based on current state
                    next_pan, next_tilt, movement_type = self._generate_next_waypoint(current_time)
                    
                    # Send target to hardware controller (if successful)
                    target_sent = self._safe_set_target_position(next_pan, next_tilt)
                    
                    # Print status periodically
                    if current_time - last_status_time > status_period:
                        self._print_status(lidar_data, next_pan, next_tilt, movement_type, target_sent)
                        last_status_time = current_time
                        
                else:
                    # Waiting for tracking mode - reduce CPU usage
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[AdaptiveTracker] Error in tracking loop: {e}")
                
            # Maintain loop timing
            loop_time = time.time() - loop_start
            sleep_time = max(0, loop_period - loop_time)
            time.sleep(sleep_time)
            
        # Cleanup on exit
        self._release_tracking_mode()
        print("[AdaptiveTracker] Tracking loop stopped")
        
    def _print_status(self, lidar_data, target_pan, target_tilt, movement_type, target_sent):
        """Print current tracking status."""
        current_pan, current_tilt = self._safe_get_current_position()
        runtime = time.time() - self.track_start_time
        
        if len(lidar_data) >= 2:
            distance, strength = lidar_data[0], lidar_data[1]
        else:
            distance, strength = 0, 0
            
        # Show if we're actually sending commands or just planning
        status_suffix = "✓" if target_sent else "⚠"
        mode_status = "ACTIVE" if self.shared_data["lidar_track_mode_active"].value else "WAITING"
            
        print(f"[AdaptiveTracker] {self.state.value.upper()} ({mode_status}) {status_suffix} | "
              f"Conf: {self.tracking_confidence:3.0f}% | "
              f"Pos: {current_pan:6.1f}°/{current_tilt:4.1f}° | "
              f"Tgt: {target_pan:6.1f}°/{target_tilt:4.1f}° | "
              f"LiDAR: {distance:4.0f}cm/{strength:5.0f} | "
              f"Vel: {self.angular_velocity_pan:4.1f}°/s | "
              f"Detections: {self.total_detections} | "
              f"Runtime: {runtime:4.0f}s")
              
    def get_status(self):
        """Get current tracking status information."""
        runtime = time.time() - self.track_start_time if hasattr(self, 'track_start_time') else 0
        current_pan, current_tilt = self._safe_get_current_position()
        
        return {
            'state': self.state.value,
            'confidence': self.tracking_confidence,
            'angular_velocity_pan': self.angular_velocity_pan,
            'angular_velocity_tilt': self.angular_velocity_tilt,
            'detections_count': len(self.target_history),
            'total_detections': self.total_detections,
            'current_position': (current_pan, current_tilt),
            'runtime_seconds': runtime,
            'is_active': self.tracking_active.is_set(),
            'has_tracking_mode': self.shared_data["lidar_track_mode_active"].value,
            'system_ready': self._is_system_ready_for_tracking()
        }
        
    def is_tracking_active(self):
        """Check if tracker is currently active."""
        return self.tracking_active.is_set()


# ==============================================================================
# Integration Functions
# ==============================================================================

def run_adaptive_tracking_mode(shared_data):
    """
    Run adaptive tracking only when adaptive_tracking_active flag is True.
    
    Args:
        shared_data: Shared data dictionary for hardware communication
    """
    print("[AdaptiveTracking] Starting adaptive tracking process")
    
    tracker = IntegratedAdaptiveTracker(shared_data)
    last_status = 0
    
    try:
        # Main monitoring loop - runs continuously but only tracks when flag is True
        while not shared_data["shutdown"].value:
            
            # Check if tracking should be active
            if shared_data["adaptive_tracking_active"].value:
                # Start tracking if not already active
                if not tracker.is_tracking_active():
                    print("[AdaptiveTracking] Flag enabled - starting tracking")
                    tracker.start_tracking()
                
                # Print status every 10 seconds while tracking
                current_time = time.time()
                if current_time - last_status > 10:
                    status = tracker.get_status()
                    print(f"[AdaptiveTracking] Status - State: {status['state']}, "
                          f"Confidence: {status['confidence']}%, "
                          f"Total Detections: {status['total_detections']}, "
                          f"Mode Active: {status['has_tracking_mode']}")
                    last_status = current_time
                    
            else:
                # Stop tracking if flag is disabled
                if tracker.is_tracking_active():
                    print("[AdaptiveTracking] Flag disabled - stopping tracking")
                    tracker.stop_tracking()
            
            time.sleep(0.1)  # Check flag every 100ms
            
    except KeyboardInterrupt:
        print("[AdaptiveTracking] Interrupted by user")
    except Exception as e:
        print(f"[AdaptiveTracking] Error: {e}")
    finally:
        if tracker.is_tracking_active():
            tracker.stop_tracking()
        print("[AdaptiveTracking] Adaptive tracking process stopped")


def integrate_with_main_system(shared_data):
    """
    Example of how to integrate adaptive tracking with main system.
    This function shows the pattern for switching between modes.
    """
    print("=== Race-Condition-Free Adaptive Tracker Integration Example ===")
    
    # Create tracker instance
    tracker = IntegratedAdaptiveTracker(shared_data)
    
    print("Available commands:")
    print("  'track' - Start adaptive tracking")
    print("  'stop' - Stop tracking")
    print("  'status' - Show current status")
    print("  'scan' - Run background scan")
    print("  'goto X Y' - Go to position X degrees pan, Y degrees tilt")
    print("  'quit' - Exit")
    
    try:
        while not shared_data["shutdown"].value:
            try:
                cmd = input("\nCommand: ").strip().lower()
                
                if cmd == 'track':
                    if not tracker.is_tracking_active():
                        tracker.start_tracking()
                    else:
                        print("Tracking already active")
                        
                elif cmd == 'stop':
                    if tracker.is_tracking_active():
                        tracker.stop_tracking()
                    else:
                        print("Tracking not active")
                        
                elif cmd == 'status':
                    status = tracker.get_status()
                    print(f"Tracking Status:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                        
                elif cmd == 'scan':
                    if tracker.is_tracking_active():
                        tracker.stop_tracking()
                        time.sleep(1)
                    print("Starting background scan...")
                    shared_data["background_scan_active"].value = True
                    
                elif cmd.startswith('goto'):
                    parts = cmd.split()
                    if len(parts) >= 3:
                        try:
                            pan = float(parts[1])
                            tilt = float(parts[2])
                            if tracker.is_tracking_active():
                                tracker.stop_tracking()
                                time.sleep(0.5)
                            print(f"Going to pan={pan}°, tilt={tilt}°")
                            shared_data["target_azimuth"].value = pan
                            shared_data["target_elevation"].value = tilt
                            shared_data["go_to_target"].value = True
                        except ValueError:
                            print("Invalid coordinates")
                    else:
                        print("Usage: goto <pan_degrees> <tilt_degrees>")
                        
                elif cmd in ['quit', 'exit', 'q']:
                    break
                    
                else:
                    print("Unknown command")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if tracker.is_tracking_active():
            tracker.stop_tracking()
