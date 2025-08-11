import math
import time
from enum import Enum
from collections import deque
from motors.motor_controller import move

class TrackingState(Enum):
    SEARCHING = "searching"
    TRACKING = "tracking" 
    LOST = "lost"
    ACQUIRING = "acquiring"

class AdaptiveTracker:
    def __init__(self, shared_dict, motor_response_time=0.1):
        self.shared_dict = shared_dict
        self.current_pan = shared_dict['stepper_degrees'].value  # Pan from stepper
        self.current_tilt = shared_dict['servo_degrees'].value  # Tilt from servo
        self.motor_response_time = motor_response_time
        
        # Target tracking variables
        self.state = TrackingState.SEARCHING
        self.target_history = deque(maxlen=5)  # Store last 5 target positions
        self.last_detection_time = 0
        self.tracking_confidence = 0
        self.search_radius = 5.0  # Initial search radius in degrees
        self.max_search_radius = 30.0
        
        # Motion prediction
        self.angular_velocity_pan = 0.0  # degrees/second
        self.angular_velocity_tilt = 0.0
        self.prediction_horizon = 0.2  # seconds ahead to predict
        
        # Detection parameters
        self.detection_threshold = 10000
        self.min_detections_for_track = 3
        self.max_time_without_detection = 2.0  # seconds
        
        # Search pattern variables
        self.search_center_pan = shared_dict["stepper_degrees"].value
        self.search_center_tilt = shared_dict["servo_degrees"].value
        self.search_angle = 0  # Current angle in spiral search
        
    def update_target_position(self, lidar_data, current_time):
        """
        Update target position from lidar data.
        
        Args:
            lidar_data: Array [distance, strength, timestamp]
            current_time: Current system timestamp
        """
        distance, strength, lidar_timestamp = lidar_data
        
        if strength >= self.detection_threshold:
            # Valid target detection
            target_pos = {
                'pan': self.current_pan,
                'tilt': self.current_tilt,
                'distance': distance,
                'timestamp': current_time,
                'strength': strength
            }
            
            self.target_history.append(target_pos)
            self.last_detection_time = current_time
            
            # Calculate angular velocity if we have enough history
            if len(self.target_history) >= 2:
                self._calculate_angular_velocity()
            
            # Update tracking confidence
            self.tracking_confidence = min(100, self.tracking_confidence + 20)
            
            # Update state based on detections
            if self.state == TrackingState.SEARCHING:
                if len(self.target_history) >= self.min_detections_for_track:
                    self.state = TrackingState.TRACKING
                    print(f"Target acquired! Switching to tracking mode.")
                else:
                    self.state = TrackingState.ACQUIRING
            elif self.state == TrackingState.LOST:
                self.state = TrackingState.ACQUIRING
                
        else:
            # No target detected
            time_since_detection = current_time - self.last_detection_time
            
            if time_since_detection > self.max_time_without_detection:
                if self.state == TrackingState.TRACKING:
                    print("Target lost! Switching to search mode.")
                    self.state = TrackingState.LOST
                    self.search_center_pan = self.current_pan
                    self.search_center_tilt = self.current_tilt
            
            # Decrease confidence
            self.tracking_confidence = max(0, self.tracking_confidence - 5)
    
    def _calculate_angular_velocity(self):
        """Calculate target angular velocity from position history."""
        if len(self.target_history) < 2:
            return
            
        recent = self.target_history[-1]
        previous = self.target_history[-2]
        
        dt = recent['timestamp'] - previous['timestamp']
        if dt > 0:
            dpan = recent['pan'] - previous['pan']
            dtilt = recent['tilt'] - previous['tilt']
            
            # Handle angle wraparound
            if abs(dpan) > 180:
                dpan = dpan - 360 * (1 if dpan > 0 else -1)
            
            self.angular_velocity_pan = dpan / dt
            self.angular_velocity_tilt = dtilt / dt
    
    def predict_target_position(self, future_time):
        """
        Predict where the target will be at a future time.
        
        Args:
            future_time (float): Time in seconds from now
            
        Returns:
            tuple: (predicted_pan, predicted_tilt)
        """
        if not self.target_history:
            return self.current_pan, self.current_tilt
            
        last_pos = self.target_history[-1]
        
        # Predict position based on angular velocity
        predicted_pan = last_pos['pan'] + self.angular_velocity_pan * future_time
        predicted_tilt = last_pos['tilt'] + self.angular_velocity_tilt * future_time
        
        return predicted_pan, predicted_tilt
    
    def generate_next_waypoint(self, current_time):
        """
        Generate the next pan/tilt waypoint based on current tracking state.
        
        Args:
            current_time (float): Current system timestamp
            
        Returns:
            tuple: (next_pan, next_tilt, movement_type)
        """
        if self.state == TrackingState.TRACKING:
            return self._tracking_waypoint(current_time)
        elif self.state == TrackingState.ACQUIRING:
            return self._acquisition_waypoint(current_time)
        elif self.state == TrackingState.LOST:
            return self._search_waypoint(current_time)
        else:  # SEARCHING
            return self._search_waypoint(current_time)
    
    def _tracking_waypoint(self, current_time):
        """Generate waypoint for active tracking mode."""
        # Predict where target will be when motor reaches position
        future_time = self.motor_response_time + self.prediction_horizon
        pred_pan, pred_tilt = self.predict_target_position(future_time)
        
        # Add small uncertainty offset based on confidence
        uncertainty_factor = (100 - self.tracking_confidence) / 100.0
        uncertainty_offset_pan = uncertainty_factor * 2.0 * (0.5 - hash(str(current_time)) % 100 / 100.0)
        uncertainty_offset_tilt = uncertainty_factor * 1.0 * (0.5 - hash(str(current_time + 1)) % 100 / 100.0)
        
        next_pan = pred_pan + uncertainty_offset_pan
        next_tilt = pred_tilt + uncertainty_offset_tilt
        
        return next_pan, next_tilt, "tracking"
    
    def _acquisition_waypoint(self, current_time):
        """Generate waypoint for target acquisition mode."""
        if self.target_history:
            # Search around last known position with small pattern
            last_pos = self.target_history[-1]
            search_offset = 3.0  # degrees
            
            # Simple oscillating pattern around last position
            time_factor = (current_time * 2) % (2 * math.pi)
            offset_pan = search_offset * math.cos(time_factor)
            offset_tilt = search_offset * 0.5 * math.sin(time_factor)
            
            next_pan = last_pos['pan'] + offset_pan
            next_tilt = last_pos['tilt'] + offset_tilt
            
            return next_pan, next_tilt, "acquiring"
        else:
            return self._search_waypoint(current_time)
    
    def _search_waypoint(self, current_time):
        """Generate waypoint for search mode."""
        # Expanding spiral search around last known or current position
        self.search_angle += 0.3  # Increment search angle
        
        # Expand search radius over time if target not found
        time_since_detection = current_time - self.last_detection_time
        radius_expansion = min(time_since_detection * 2, self.max_search_radius)
        current_search_radius = min(self.search_radius + radius_expansion, self.max_search_radius)
        
        # Spiral pattern
        spiral_factor = self.search_angle / (2 * math.pi)
        radius = current_search_radius * spiral_factor
        
        offset_pan = radius * math.cos(self.search_angle)
        offset_tilt = radius * 0.6 * math.sin(self.search_angle)  # Limit tilt movement
        
        next_pan = self.search_center_pan + offset_pan
        next_tilt = self.search_center_tilt + offset_tilt
        
        # Reset spiral if we've gone too far
        if radius > current_search_radius:
            self.search_angle = 0
        
        return next_pan, next_tilt, "searching"
    
    def update_current_position(self):
        """Update current pan/tilt position from shared dictionary."""
        self.current_pan = self.shared_dict['stepper_degrees'].value
        self.current_tilt = self.shared_dict['servo_degrees'].value
        
    
    def get_motor_commands(self, target_pan, target_tilt):
        """
        Convert target position to motor movement commands compatible with your move() function.
        
        Args:
            target_pan (float): Target pan angle in degrees (stepper)
            target_tilt (float): Target tilt angle in degrees (servo)
            
        Returns:
            list: List of (direction, degrees) commands for motors
        """
        # Update current position from shared data first
        self.update_current_position()
        
        commands = []
        
        # Calculate pan movement (stepper motor)
        pan_change = target_pan - self.current_pan
        # Handle wraparound for stepper degrees (0-360)
        if pan_change > 180:
            pan_change -= 360
        elif pan_change < -180:
            pan_change += 360
            
        if abs(pan_change) > 0.5:  # Only move if change is significant
            if pan_change > 0:
                commands.append(("right", round(abs(pan_change))))
            else:
                commands.append(("left", round(abs(pan_change))))
        
        # Calculate tilt movement (servo motor, 0-180 range)
        tilt_change = target_tilt - self.current_tilt
        target_tilt_clamped = max(0, min(180, target_tilt))  # Clamp to servo range
        tilt_change = target_tilt_clamped - self.current_tilt
        
        if abs(tilt_change) > 0.5:  # Only move if change is significant
            if tilt_change > 0:
                commands.append(("up", round(abs(tilt_change))))
            else:
                commands.append(("down", round(abs(tilt_change))))
        
        return commands
    
    def get_status(self):
        """Get current tracking status information."""
        return {
            'state': self.state.value,
            'confidence': self.tracking_confidence,
            'angular_velocity_pan': self.angular_velocity_pan,
            'angular_velocity_tilt': self.angular_velocity_tilt,
            'detections_count': len(self.target_history),
            'current_position': (self.current_pan, self.current_tilt)
        }


def process_tracking_step(tracker, lidar_data, current_time):
    """
    Process one step of the tracking algorithm.
    
    Args:
        tracker (AdaptiveTracker): The tracking system instance
        lidar_data: Array [distance, strength, timestamp]
        current_time (float): Current system timestamp
        
    Returns:
        tuple: (motor_commands, movement_type, status)
    """
    # Update target position from lidar data
    tracker.update_target_position(lidar_data, current_time)
    
    # Generate next waypoint
    next_pan, next_tilt, movement_type = tracker.generate_next_waypoint(current_time)
    
    # Convert to motor commands
    motor_commands = tracker.get_motor_commands(next_pan, next_tilt)
    
    # Get current status
    status = tracker.get_status()
    
    return motor_commands, movement_type, status


def execute_adaptive_tracking_commands(pi, motor_commands, movement_queue, shared_data):
    """
    Execute motor commands using your existing motor control functions.
    
    Args:
        pi: pigpio instance
        motor_commands: List of (direction, degrees) tuples
        movement_queue: Your movement queue for stepper commands
        shared_data: Your shared data dictionary
    """
    for direction, degrees in motor_commands:
        if shared_data['shutdown'].value == False:
            break
            
        # Use your existing move function
        move(pi, direction, degrees, 0.0001, movement_queue, shared_data)
        
        # Wait for stepper movements to complete
        if direction in ['left', 'right']:
            while shared_data['stepper_busy'].value == False:
                time.sleep(0.01)
                if shared_data['shutdown'].value== True:
                    break


def adaptive_tracking_loop(shared_data, pi, movement_queue, lidar_data):
    """
    Main adaptive tracking loop that replaces your arc search when target detected.
    
    Args:
        shared_data: Your shared data dictionary
        pi: pigpio instance  
        movement_queue: Your movement queue
        lidar_data: Your lidar data array [distance, strength, timestamp]
    """
    
    # Initialize tracker with current motor positions
    tracker = AdaptiveTracker(shared_data)
    
    print("=== Starting Adaptive Tracking ===")
    
    # Main tracking loop
    while (shared_data['acquire_points'].value == True and shared_data['shutdown'].value == False):
        
        current_time = time.time()
        
        # Process tracking step
        motor_commands, movement_type, status = process_tracking_step(
            tracker, lidar_data, current_time
        )
        
        # Execute motor commands
        if motor_commands:
            execute_adaptive_tracking_commands(pi, motor_commands, movement_queue, shared_data)
        
        # Optional: Print status for debugging
        if status['state'] != 'searching':  # Only print when interesting
            print(f"Tracking: {status['state']}, Confidence: {status['confidence']}%, "
                  f"Position: pan={status['current_position'][0]:.1f}°, tilt={status['current_position'][1]:.1f}°")
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.05)  # 20Hz update rate
    
    print("=== Adaptive Tracking Stopped ===")



# Example usage function
def simulate_tracking_sequence():
    """Simulate a tracking sequence with sample data."""
    
    # Mock shared data dictionary like yours
    mock_shared_data = {
        'stepper_degrees': type('obj', (object,), {'value': 0.0})(),
        'servo_degrees': type('obj', (object,), {'value': 90.0})(),
        'shutdown': type('obj', (object,), {'value': False})(),
        'acquire_points': type('obj', (object,), {'value': True})()
    }
    
    # Initialize tracker
    tracker = AdaptiveTracker(mock_shared_data)
    
    print("=== Adaptive Tracking Simulation ===")
    print("Target detection threshold: 10,000 strength")
    print(f"Initial position: pan={tracker.current_pan}°, tilt={tracker.current_tilt}°")
    print()
    
    # Simulate some tracking steps
    simulation_steps = [
        ([5.2, 5000, 1.001], "No target detected"),
        ([4.8, 12000, 1.002], "Target detected!"),
        ([4.9, 11500, 1.003], "Target tracking"),
        ([5.1, 10800, 1.004], "Target moving"),
        ([5.3, 9500, 1.005], "Target lost"),
        ([6.1, 15000, 1.006], "Target reacquired"),
    ]
    
    current_time = time.time()
    
    for step, (lidar_data, description) in enumerate(simulation_steps):
        print(f"Step {step + 1}: {description}")
        print(f"  Lidar: distance={lidar_data[0]}m, strength={lidar_data[1]}")
        
        # Process tracking step
        motor_commands, movement_type, status = process_tracking_step(
            tracker, lidar_data, current_time + step * 0.1
        )
        
        print(f"  Motor commands: {motor_commands}")
        print(f"  Movement type: {movement_type}")
        print(f"  State: {status['state']}, Confidence: {status['confidence']}%")
        print(f"  Current position: pan={status['current_position'][0]:.1f}°, tilt={status['current_position'][1]:.1f}°")
        
        # Simulate motor movement by updating mock shared data
        for direction, degrees in motor_commands:
            if direction == "right":
                mock_shared_data['stepper_degrees'].value = (mock_shared_data['stepper_degrees'].value + degrees) % 360
            elif direction == "left":
                mock_shared_data['stepper_degrees'].value = (mock_shared_data['stepper_degrees'].value - degrees) % 360
            elif direction == "up":
                mock_shared_data['servo_degrees'].value = min(180, mock_shared_data['servo_degrees'].value + degrees)
            elif direction == "down":
                mock_shared_data['servo_degrees'].value = max(0, mock_shared_data['servo_degrees'].value - degrees)
        
        print()


# Example of how to integrate with your existing motor control system
def example_motor_integration():
    """
    Example showing how to integrate with your existing motor control system.
    """
    
    print("=== Motor Integration Example ===")
    print("This shows how to integrate the adaptive tracker with your existing code:")
    print()
    
    code_example = '''
    
# In main control loop, replace or modify your start_arc_search call:

def main_tracking_loop(shared_data, pi, movement_queue):
    while not shared_data['shutdown'].value:
        # Get current lidar data (you already have this)
        lidar_data = [
            shared_data['lidar_distance'].value,   # Your distance reading
            shared_data['lidar_strength'].value,   # Your strength reading  
            time.time()                           # Current timestamp
        ]
        
        # Check if we should track or search
        if lidar_data[1] >= 10000:  # Strong target detected
            print("Target detected - switching to adaptive tracking")
            adaptive_tracking_loop(shared_data, pi, movement_queue, lidar_data)
        else:
            print("No target - running arc search")
            start_arc_search(shared_data, pi, movement_queue,
                           delta_azimuth=50, distance_meters=2)
        
        time.sleep(0.1)  # Small delay between mode checks
'''
    
    print(code_example)

if __name__ == "__main__":
    simulate_tracking_sequence()
    print("\n" + "="*50 + "\n")
    example_motor_integration()