#!/usr/bin/env python3
"""
Precision Spherical Grid-Based Target Tracking System
Implements systematic spherical coordinate grid search with velocity decomposition
and predictive 3D tracking for objects with 18°/s total angular velocity.
"""

import time
import math
import numpy as np
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import json
from datetime import datetime

# System Constants
LIDAR_FOV = 3.2  # degrees
MOTOR_PRECISION = 1.0  # degrees (minimum servo movement)
TOTAL_ANGULAR_VELOCITY = 18.0  # degrees/second (known constraint)
ELEVATION_RANGE = (0.0, 90.0)  # degrees
AZIMUTH_RANGE = (0.0, 360.0)  # degrees
STRENGTH_THRESHOLD = 9000  # LiDAR detection threshold
DISTANCE_RANGE = (0.2, 1.5)  # meters (20cm to 150cm)


class TrackingState(Enum):
    """State machine states for tracking system"""
    IDLE = 0
    SPHERICAL_COARSE_SEARCH = 1
    SPHERICAL_FINE_POSITIONING = 2
    VELOCITY_DECOMPOSITION = 3
    PREDICTIVE_3D_TRACKING = 4
    LOST = 5
    ERROR = 6


@dataclass
class SphericalPosition:
    """Represents a position in spherical coordinates"""
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float  # meters
    strength: float  # LiDAR signal strength
    timestamp: float  # Unix timestamp
    coverage_area: float = 0.0  # Calculated coverage area in sq degrees

    def __str__(self):
        return f"Az:{self.azimuth:.1f}° El:{self.elevation:.1f}° Dist:{self.distance:.2f}m Str:{self.strength:.0f}"


@dataclass
class VelocityComponents:
    """3D velocity decomposition"""
    azimuth_velocity: float  # degrees/second
    elevation_velocity: float  # degrees/second
    total_velocity: float  # degrees/second (should equal 18°/s)
    confidence: float  # 0.0 to 1.0
    trajectory_phase: str  # "ascending", "descending", "horizontal"

    def validate_constraint(self) -> bool:
        """Validate that velocity components satisfy the 18°/s constraint"""
        calculated_total = math.sqrt(self.azimuth_velocity ** 2 + self.elevation_velocity ** 2)
        return abs(calculated_total - TOTAL_ANGULAR_VELOCITY) < 0.5  # Allow 0.5°/s tolerance


class IntegratedAdaptiveTracker:
    """
    Comprehensive spherical grid-based target acquisition and tracking system.
    Uses systematic search with known velocity constraints for optimal efficiency.
    """

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.state = TrackingState.IDLE
        self.running = False

        # Position tracking
        self.current_position = None
        self.target_history: List[SphericalPosition] = []
        self.grid_search_results: List[SphericalPosition] = []

        # Velocity tracking
        self.velocity_components: Optional[VelocityComponents] = None
        self.trajectory_parameters: Dict = {}

        # Search parameters
        self.raan_azimuth = 0.0  # Starting azimuth from RAAN
        self.search_start_time = None
        self.acquisition_time = None

        # Threading
        self.tracker_thread = None
        self.state_lock = threading.Lock()

        # Performance metrics
        self.metrics = {
            "grid_positions_tested": 0,
            "acquisition_attempts": 0,
            "tracking_accuracy": 0.0,
            "prediction_errors": [],
            "last_update": time.time()
        }

        print("[Tracker] IntegratedAdaptiveTracker initialized with spherical grid search")

    # ==================== PHASE 1: SPHERICAL COARSE GRID SEARCH ====================

    def calculate_spherical_grid_positions(self) -> List[Tuple[float, float]]:
        """
        Calculate spherical grid positions with proper geometry corrections.
        Returns list of (azimuth, elevation) tuples for coarse search.
        """
        positions = []

        # Calculate elevation positions (28 positions for 90° range with 3.2° FOV)
        num_elevation_positions = int(90.0 / LIDAR_FOV) + 1  # 29 positions to cover full range

        for elev_idx in range(num_elevation_positions):
            elevation = min(elev_idx * LIDAR_FOV, 90.0)

            # Calculate azimuth spacing based on elevation (spherical geometry correction)
            if elevation < 85.0:  # Normal calculation
                azimuth_spacing = LIDAR_FOV / max(math.cos(math.radians(elevation)), 0.1)
            else:  # Near pole, use larger spacing
                azimuth_spacing = 30.0  # Reasonable spacing near pole

            # Calculate number of azimuth positions for this elevation
            num_azimuth_positions = max(int(360.0 / azimuth_spacing), 12)  # At least 12 positions

            for az_idx in range(num_azimuth_positions):
                azimuth = (self.raan_azimuth + az_idx * azimuth_spacing) % 360.0
                positions.append((azimuth, elevation))

        print(f"[Tracker] Generated {len(positions)} spherical grid positions for coarse search")
        return positions

    def perform_coarse_grid_search(self) -> Optional[SphericalPosition]:
        """
        Phase 1: Perform systematic spherical coordinate coarse grid search.
        Returns the position with highest LiDAR strength.
        """
        print("[Tracker] Starting PHASE 1: Spherical Coarse Grid Search")
        self.search_start_time = time.time()
        self.grid_search_results.clear()

        grid_positions = self.calculate_spherical_grid_positions()
        best_position = None
        max_strength = 0

        for idx, (azimuth, elevation) in enumerate(grid_positions):
            if self.shared_data["shutdown"].value:
                break

            # Calculate coverage area for this position
            coverage_area = self._calculate_coverage_area(elevation)

            # Move to grid position
            self._move_to_position(azimuth, elevation)
            time.sleep(0.05)  # Allow hardware to settle

            # Get LiDAR measurement
            measurement = self._get_lidar_measurement()
            if measurement:
                position = SphericalPosition(
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=measurement[0],
                    strength=measurement[1],
                    timestamp=measurement[2],
                    coverage_area=coverage_area
                )
                self.grid_search_results.append(position)

                # Update best position
                if position.strength > max_strength:
                    max_strength = position.strength
                    best_position = position
                    print(f"[Tracker] New best position: {position}")

            # Update metrics
            self.metrics["grid_positions_tested"] += 1

            # Progress update every 10 positions
            if idx % 10 == 0:
                progress = (idx / len(grid_positions)) * 100
                print(f"[Tracker] Coarse search progress: {progress:.1f}% ({idx}/{len(grid_positions)})")

        if best_position:
            search_time = time.time() - self.search_start_time
            print(f"[Tracker] Coarse search completed in {search_time:.1f}s")
            print(f"[Tracker] Best position: {best_position}")
            return best_position
        else:
            print("[Tracker] ERROR: Coarse search failed - no valid positions found")
            return None

    # ==================== PHASE 2: FINE POSITIONING ====================

    def perform_fine_positioning(self, coarse_position: SphericalPosition) -> Optional[SphericalPosition]:
        """
        Phase 2: Perform fine positioning around the best coarse position.
        Uses 1-degree precision (motor minimum increment).
        """
        print("[Tracker] Starting PHASE 2: Fine Positioning")
        print(
            f"[Tracker] Centering on coarse position: Az={coarse_position.azimuth:.1f}° El={coarse_position.elevation:.1f}°")

        fine_positions = []
        search_radius = LIDAR_FOV / 2  # ±1.6 degrees

        # Calculate fine grid with spherical corrections
        for elev_offset in np.arange(-search_radius, search_radius + MOTOR_PRECISION, MOTOR_PRECISION):
            elevation = coarse_position.elevation + elev_offset
            if not (ELEVATION_RANGE[0] <= elevation <= ELEVATION_RANGE[1]):
                continue

            # Spherical correction for azimuth spacing
            azimuth_step = MOTOR_PRECISION / max(math.cos(math.radians(elevation)), 0.1)

            for az_offset in np.arange(-search_radius, search_radius + azimuth_step, azimuth_step):
                azimuth = (coarse_position.azimuth + az_offset) % 360.0

                # Move to fine position
                self._move_to_position(azimuth, elevation)
                time.sleep(0.03)  # Shorter settle time for fine movements

                # Get measurement
                measurement = self._get_lidar_measurement()
                if measurement and measurement[1] > STRENGTH_THRESHOLD:
                    position = SphericalPosition(
                        azimuth=azimuth,
                        elevation=elevation,
                        distance=measurement[0],
                        strength=measurement[1],
                        timestamp=measurement[2],
                        coverage_area=self._calculate_coverage_area(elevation)
                    )
                    fine_positions.append(position)

        # Find optimal position
        if fine_positions:
            best_fine = max(fine_positions, key=lambda p: p.strength)
            print(f"[Tracker] Fine positioning complete. Optimal position: {best_fine}")
            self.current_position = best_fine
            self.target_history.append(best_fine)
            return best_fine
        else:
            print("[Tracker] WARNING: Fine positioning failed, using coarse position")
            return coarse_position

    # ==================== PHASE 3: VELOCITY DECOMPOSITION ====================

    def perform_velocity_analysis(self, current_pos: SphericalPosition) -> Optional[VelocityComponents]:
        """
        Phase 3: Analyze 3D velocity components through systematic testing.
        Decomposes the known 18°/s total velocity into azimuth and elevation components.
        """
        print("[Tracker] Starting PHASE 3: Velocity Decomposition Analysis")

        # Test elevation movement
        elevation_gradient = self._test_elevation_gradient(current_pos)

        # Test azimuth movement (accounting for rightward motion constraint)
        azimuth_gradient = self._test_azimuth_gradient(current_pos)

        # Determine trajectory phase
        if elevation_gradient > 0.1:
            trajectory_phase = "ascending"
        elif elevation_gradient < -0.1:
            trajectory_phase = "descending"
        else:
            trajectory_phase = "horizontal"

        # Calculate velocity components using constraint
        # We know: sqrt(az_vel^2 + el_vel^2) = 18°/s
        # And we have the ratio from gradients

        if abs(elevation_gradient) < 0.01:  # Nearly horizontal
            azimuth_velocity = TOTAL_ANGULAR_VELOCITY
            elevation_velocity = 0.0
        else:
            # Use gradient ratio to decompose velocity
            gradient_magnitude = math.sqrt(azimuth_gradient ** 2 + elevation_gradient ** 2)
            if gradient_magnitude > 0:
                azimuth_velocity = TOTAL_ANGULAR_VELOCITY * (azimuth_gradient / gradient_magnitude)
                elevation_velocity = TOTAL_ANGULAR_VELOCITY * (elevation_gradient / gradient_magnitude)
            else:
                # Default to horizontal if no gradient detected
                azimuth_velocity = TOTAL_ANGULAR_VELOCITY
                elevation_velocity = 0.0

        # Ensure rightward (positive azimuth) constraint
        azimuth_velocity = abs(azimuth_velocity)

        # Create velocity components
        velocity = VelocityComponents(
            azimuth_velocity=azimuth_velocity,
            elevation_velocity=elevation_velocity,
            total_velocity=math.sqrt(azimuth_velocity ** 2 + elevation_velocity ** 2),
            confidence=0.8,  # Initial confidence
            trajectory_phase=trajectory_phase
        )

        # Validate constraint
        if velocity.validate_constraint():
            print(f"[Tracker] Velocity decomposition successful:")
            print(f"  - Azimuth velocity: {velocity.azimuth_velocity:.2f}°/s")
            print(f"  - Elevation velocity: {velocity.elevation_velocity:.2f}°/s")
            print(f"  - Total velocity: {velocity.total_velocity:.2f}°/s (constraint: 18°/s)")
            print(f"  - Trajectory phase: {velocity.trajectory_phase}")
            self.velocity_components = velocity
            return velocity
        else:
            print(f"[Tracker] WARNING: Velocity constraint validation failed!")
            print(f"  - Calculated total: {velocity.total_velocity:.2f}°/s (expected: 18°/s)")
            # Attempt to correct by scaling
            scale_factor = TOTAL_ANGULAR_VELOCITY / velocity.total_velocity
            velocity.azimuth_velocity *= scale_factor
            velocity.elevation_velocity *= scale_factor
            velocity.total_velocity = TOTAL_ANGULAR_VELOCITY
            self.velocity_components = velocity
            return velocity

    def _test_elevation_gradient(self, current_pos: SphericalPosition) -> float:
        """Test positions above and below to determine elevation velocity component"""
        measurements = {}

        # Test at current, +1°, and -1° elevation
        for offset in [0, 1.0, -1.0]:
            test_elevation = current_pos.elevation + offset
            if ELEVATION_RANGE[0] <= test_elevation <= ELEVATION_RANGE[1]:
                self._move_to_position(current_pos.azimuth, test_elevation)
                time.sleep(0.05)
                measurement = self._get_lidar_measurement()
                if measurement:
                    measurements[offset] = measurement[1]  # strength

        # Calculate gradient
        if 1.0 in measurements and -1.0 in measurements:
            gradient = (measurements[1.0] - measurements[-1.0]) / 2.0
            return gradient / 1000.0  # Normalize
        return 0.0

    def _test_azimuth_gradient(self, current_pos: SphericalPosition) -> float:
        """Test positions left and right to determine azimuth velocity component"""
        measurements = {}

        # Account for expected movement during test (0.1 second test interval)
        expected_movement = TOTAL_ANGULAR_VELOCITY * 0.1

        # Test at predicted future positions
        for offset in [0, expected_movement, -expected_movement]:
            test_azimuth = (current_pos.azimuth + offset) % 360.0
            self._move_to_position(test_azimuth, current_pos.elevation)
            time.sleep(0.05)
            measurement = self._get_lidar_measurement()
            if measurement:
                measurements[offset] = measurement[1]  # strength

        # Calculate gradient (we expect positive since object moves rightward)
        if expected_movement in measurements:
            gradient = measurements[expected_movement] - measurements.get(0, 0)
            return max(gradient / 1000.0, 0.1)  # Ensure positive (rightward motion)
        return 1.0  # Default to rightward

    # ==================== PHASE 4: PREDICTIVE TRACKING ====================

    def perform_predictive_tracking(self):
        """
        Phase 4: Continuous predictive tracking using discovered velocity components.
        Implements proactive positioning based on predicted 3D trajectory.
        """
        if not self.velocity_components or not self.current_position:
            print("[Tracker] ERROR: Cannot start predictive tracking - missing velocity or position data")
            return

        print("[Tracker] Starting PHASE 4: Predictive 3D Tracking")
        print(f"[Tracker] Using velocity: Az={self.velocity_components.azimuth_velocity:.2f}°/s, "
              f"El={self.velocity_components.elevation_velocity:.2f}°/s")

        prediction_horizon = 0.5  # seconds ahead
        last_update_time = self.current_position.timestamp
        tracking_start = time.time()
        consecutive_misses = 0
        max_consecutive_misses = 5

        while self.running and self.state == TrackingState.PREDICTIVE_3D_TRACKING:
            current_time = time.time()
            dt = current_time - last_update_time

            # Predict next position
            predicted_azimuth = (self.current_position.azimuth +
                                 self.velocity_components.azimuth_velocity * (dt + prediction_horizon)) % 360.0
            predicted_elevation = self.current_position.elevation + \
                                  self.velocity_components.elevation_velocity * (dt + prediction_horizon)

            # Handle elevation bounds and trajectory phase transitions
            if predicted_elevation <= ELEVATION_RANGE[0]:
                predicted_elevation = ELEVATION_RANGE[0]
                self.velocity_components.elevation_velocity = abs(self.velocity_components.elevation_velocity)
                self.velocity_components.trajectory_phase = "ascending"
                self._recalculate_velocity_components()
            elif predicted_elevation >= ELEVATION_RANGE[1]:
                predicted_elevation = ELEVATION_RANGE[1]
                self.velocity_components.elevation_velocity = -abs(self.velocity_components.elevation_velocity)
                self.velocity_components.trajectory_phase = "descending"
                self._recalculate_velocity_components()

            # Move to predicted position
            self._move_to_position(predicted_azimuth, predicted_elevation)

            # Update shared data for GUI
            self.shared_data["predicted_azimuth"].value = predicted_azimuth
            self.shared_data["predicted_elevation"].value = predicted_elevation

            # Wait and get measurement
            time.sleep(prediction_horizon)
            measurement = self._get_lidar_measurement()

            if measurement and measurement[1] > STRENGTH_THRESHOLD:
                # Successful tracking
                consecutive_misses = 0

                # Create new position record
                new_position = SphericalPosition(
                    azimuth=self.shared_data["stepper_degrees"].value,
                    elevation=self.shared_data["servo_degrees"].value,
                    distance=measurement[0],
                    strength=measurement[1],
                    timestamp=measurement[2]
                )

                # Update trajectory based on actual vs predicted
                self._update_trajectory_model(new_position, predicted_azimuth, predicted_elevation)

                # Store position
                self.current_position = new_position
                self.target_history.append(new_position)

                # Update shared data for other processes
                self._update_shared_tracking_data(new_position)

                # Update timing
                last_update_time = current_time

                # Calculate and store prediction error
                az_error = abs(new_position.azimuth - predicted_azimuth)
                el_error = abs(new_position.elevation - predicted_elevation)
                total_error = math.sqrt(az_error ** 2 + el_error ** 2)
                self.metrics["prediction_errors"].append(total_error)

                # Print status every 10 updates
                if len(self.target_history) % 10 == 0:
                    avg_error = np.mean(self.metrics["prediction_errors"][-10:])
                    tracking_duration = time.time() - tracking_start
                    print(f"[Tracker] Tracking update #{len(self.target_history)}")
                    print(f"  - Position: {new_position}")
                    print(f"  - Avg prediction error: {avg_error:.2f}°")
                    print(f"  - Tracking duration: {tracking_duration:.1f}s")
                    print(f"  - Velocity confidence: {self.velocity_components.confidence:.2%}")
            else:
                # Missed detection
                consecutive_misses += 1
                print(f"[Tracker] Missed detection {consecutive_misses}/{max_consecutive_misses}")

                if consecutive_misses >= max_consecutive_misses:
                    print("[Tracker] Target lost! Too many consecutive misses.")
                    self.state = TrackingState.LOST
                    break

                # Try recovery by searching nearby
                self._attempt_recovery(predicted_azimuth, predicted_elevation)

            # Check for shutdown
            if self.shared_data["shutdown"].value:
                break

        # Tracking ended
        if self.state == TrackingState.LOST:
            print("[Tracker] Predictive tracking failed - target lost")
        else:
            print("[Tracker] Predictive tracking ended")

    def _recalculate_velocity_components(self):
        """Recalculate velocity components maintaining 18°/s constraint"""
        # Ensure total velocity remains 18°/s after phase change
        az_vel = self.velocity_components.azimuth_velocity
        el_vel = self.velocity_components.elevation_velocity

        current_total = math.sqrt(az_vel ** 2 + el_vel ** 2)
        if abs(current_total - TOTAL_ANGULAR_VELOCITY) > 0.5:
            scale = TOTAL_ANGULAR_VELOCITY / current_total
            self.velocity_components.azimuth_velocity *= scale
            self.velocity_components.elevation_velocity *= scale
            self.velocity_components.total_velocity = TOTAL_ANGULAR_VELOCITY

    def _update_trajectory_model(self, actual_pos: SphericalPosition,
                                 predicted_az: float, predicted_el: float):
        """Update trajectory model based on prediction errors"""
        # Calculate errors
        az_error = actual_pos.azimuth - predicted_az
        el_error = actual_pos.elevation - predicted_el

        # Handle azimuth wraparound
        if az_error > 180:
            az_error -= 360
        elif az_error < -180:
            az_error += 360

        # Apply small corrections to velocity estimates (0.1 learning rate)
        learning_rate = 0.1
        self.velocity_components.azimuth_velocity += az_error * learning_rate
        self.velocity_components.elevation_velocity += el_error * learning_rate

        # Maintain constraint
        self._recalculate_velocity_components()

        # Update confidence based on prediction accuracy
        error_magnitude = math.sqrt(az_error ** 2 + el_error ** 2)
        if error_magnitude < 1.0:  # Very accurate
            self.velocity_components.confidence = min(1.0, self.velocity_components.confidence + 0.01)
        elif error_magnitude > 3.0:  # Poor accuracy
            self.velocity_components.confidence = max(0.3, self.velocity_components.confidence - 0.05)

    def _attempt_recovery(self, last_az: float, last_el: float):
        """Attempt to recover lost target by searching nearby positions"""
        print("[Tracker] Attempting recovery search...")
        search_offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]

        for az_offset, el_offset in search_offsets:
            test_az = (last_az + az_offset) % 360.0
            test_el = max(ELEVATION_RANGE[0], min(ELEVATION_RANGE[1], last_el + el_offset))

            self._move_to_position(test_az, test_el)
            time.sleep(0.03)

            measurement = self._get_lidar_measurement()
            if measurement and measurement[1] > STRENGTH_THRESHOLD:
                print(f"[Tracker] Recovery successful at offset ({az_offset}, {el_offset})")
                return True

        return False

    # ==================== UTILITY FUNCTIONS ====================

    def _calculate_coverage_area(self, elevation: float) -> float:
        """Calculate the coverage area at a given elevation (spherical geometry)"""
        # Area decreases with cos(elevation) due to spherical projection
        base_area = LIDAR_FOV ** 2  # Square degrees at equator
        area = base_area * max(math.cos(math.radians(elevation)), 0.1)
        return area

    def _move_to_position(self, azimuth: float, elevation: float):
        """Command hardware to move to specified position"""
        # Ensure angles are within bounds
        azimuth = azimuth % 360.0
        elevation = max(ELEVATION_RANGE[0], min(ELEVATION_RANGE[1], elevation))

        # Set targets in shared data
        self.shared_data["target_azimuth"].value = azimuth
        self.shared_data["target_elevation"].value = elevation
        self.shared_data["go_to_target"].value = True

        # Wait for movement to complete (with timeout)
        timeout = 2.0
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.shared_data["target_reached"].value:
                self.shared_data["target_reached"].value = False
                break
            time.sleep(0.001)

    def _get_lidar_measurement(self) -> Optional[Tuple[float, float, float]]:
        """Get current LiDAR measurement"""
        # Get data from shared memory
        with self.shared_data["lidar_data"].get_lock():
            dist_cm = self.shared_data["lidar_data"][0]
            strength = self.shared_data["lidar_data"][1]
            timestamp = self.shared_data["lidar_data"][2]

        # Convert distance to meters and validate
        dist_m = dist_cm / 100.0
        if DISTANCE_RANGE[0] <= dist_m <= DISTANCE_RANGE[1] and strength > 0:
            return (dist_m, strength, timestamp)
        return None

    def _update_shared_tracking_data(self, position: SphericalPosition):
        """Update shared data for GUI and other processes"""
        self.shared_data["estimated_azimuth"].value = position.azimuth
        self.shared_data["estimated_elevation"].value = position.elevation
        self.shared_data["satellite_detected"].value = True
        self.shared_data["system_status"].value = 1  # Tracking

        # Update satellite points array
        with self.shared_data["satellite_points"].get_lock():
            self.shared_data["satellite_points"][0] = position.azimuth
            self.shared_data["satellite_points"][1] = position.elevation
            self.shared_data["satellite_points"][2] = position.distance * 100  # Convert to cm
            self.shared_data["satellite_points"][3] = position.strength
            self.shared_data["satellite_points"][4] = position.timestamp

        # Add to tracking history for TLE generation
        self.shared_data["tracking_history"].append(
            (position.azimuth, position.elevation, position.distance,
             position.strength, position.timestamp)
        )

    # ==================== STATE MACHINE & CONTROL ====================

    def start_tracking(self, raan_azimuth: float = 0.0):
        """Start the complete tracking sequence"""
        if self.running:
            print("[Tracker] Already running")
            return

        self.raan_azimuth = raan_azimuth
        self.running = True
        self.state = TrackingState.SPHERICAL_COARSE_SEARCH

        # Start tracker thread
        self.tracker_thread = threading.Thread(target=self._tracking_loop)
        self.tracker_thread.daemon = True
        self.tracker_thread.start()

        print(f"[Tracker] Started tracking with RAAN azimuth: {raan_azimuth}°")

    def _tracking_loop(self):
        """Main tracking loop implementing the state machine"""
        print("[Tracker] Tracking loop started")

        try:
            while self.running and not self.shared_data["shutdown"].value:
                with self.state_lock:
                    current_state = self.state

                if current_state == TrackingState.SPHERICAL_COARSE_SEARCH:
                    # Phase 1: Coarse grid search
                    coarse_position = self.perform_coarse_grid_search()
                    if coarse_position:
                        self.state = TrackingState.SPHERICAL_FINE_POSITIONING
                    else:
                        print("[Tracker] Coarse search failed - entering ERROR state")
                        self.state = TrackingState.ERROR

                elif current_state == TrackingState.SPHERICAL_FINE_POSITIONING:
                    # Phase 2: Fine positioning
                    if self.current_position or self.grid_search_results:
                        start_pos = self.current_position or max(self.grid_search_results, key=lambda p: p.strength)
                        fine_position = self.perform_fine_positioning(start_pos)
                        if fine_position:
                            self.acquisition_time = time.time() - self.search_start_time
                            print(f"[Tracker] Target acquired in {self.acquisition_time:.1f} seconds")
                            self.state = TrackingState.VELOCITY_DECOMPOSITION
                        else:
                            self.state = TrackingState.ERROR
                    else:
                        self.state = TrackingState.ERROR

                elif current_state == TrackingState.VELOCITY_DECOMPOSITION:
                    # Phase 3: Velocity analysis
                    if self.current_position:
                        velocity = self.perform_velocity_analysis(self.current_position)
                        if velocity and velocity.validate_constraint():
                            self.state = TrackingState.PREDICTIVE_3D_TRACKING
                        else:
                            print("[Tracker] Velocity analysis failed - attempting retry")
                            time.sleep(0.5)
                            # Retry once
                            velocity = self.perform_velocity_analysis(self.current_position)
                            if velocity:
                                self.state = TrackingState.PREDICTIVE_3D_TRACKING
                            else:
                                self.state = TrackingState.ERROR
                    else:
                        self.state = TrackingState.ERROR

                elif current_state == TrackingState.PREDICTIVE_3D_TRACKING:
                    # Phase 4: Predictive tracking
                    self.perform_predictive_tracking()
                    # State change handled within perform_predictive_tracking

                elif current_state == TrackingState.LOST:
                    print("[Tracker] Target lost - attempting reacquisition")
                    # Try to reacquire using last known velocity
                    time.sleep(1.0)
                    self.state = TrackingState.SPHERICAL_COARSE_SEARCH

                elif current_state == TrackingState.ERROR:
                    print("[Tracker] Error state - stopping tracker")
                    self.running = False
                    break

                elif current_state == TrackingState.IDLE:
                    time.sleep(0.1)

                # Small delay between state transitions
                time.sleep(0.01)

        except Exception as e:
            print(f"[Tracker] ERROR in tracking loop: {e}")
            import traceback
            traceback.print_exc()
            self.state = TrackingState.ERROR
            self.shared_data["system_status"].value = 3  # Error

        finally:
            self.running = False
            print("[Tracker] Tracking loop ended")

    def stop_tracking(self):
        """Stop tracking and generate report"""
        print("[Tracker] Stopping tracker...")
        self.running = False

        if self.tracker_thread:
            self.tracker_thread.join(timeout=2.0)

        # Generate and save tracking report
        self.generate_tracking_report()

        # Reset shared data
        self.shared_data["satellite_detected"].value = False
        self.shared_data["system_status"].value = 0  # Idle

        print("[Tracker] Tracker stopped")

    def generate_tracking_report(self):
        """Generate comprehensive tracking report with all collected data"""
        if not self.target_history:
            print("[Tracker] No tracking data to report")
            return

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "raan_azimuth": self.raan_azimuth,
                "acquisition_time": self.acquisition_time,
                "total_tracking_points": len(self.target_history),
                "grid_positions_tested": self.metrics["grid_positions_tested"],
                "tracking_duration": self.target_history[-1].timestamp - self.target_history[0].timestamp if len(
                    self.target_history) > 1 else 0
            },
            "velocity_decomposition": {
                "azimuth_velocity": self.velocity_components.azimuth_velocity if self.velocity_components else None,
                "elevation_velocity": self.velocity_components.elevation_velocity if self.velocity_components else None,
                "total_velocity": self.velocity_components.total_velocity if self.velocity_components else None,
                "confidence": self.velocity_components.confidence if self.velocity_components else None,
                "trajectory_phase": self.velocity_components.trajectory_phase if self.velocity_components else None,
                "constraint_validated": self.velocity_components.validate_constraint() if self.velocity_components else False
            },
            "trajectory_characterization": {
                "start_position": {
                    "azimuth": self.target_history[0].azimuth,
                    "elevation": self.target_history[0].elevation
                } if self.target_history else None,
                "end_position": {
                    "azimuth": self.target_history[-1].azimuth,
                    "elevation": self.target_history[-1].elevation
                } if self.target_history else None,
                "elevation_range": {
                    "min": min(p.elevation for p in self.target_history),
                    "max": max(p.elevation for p in self.target_history)
                } if self.target_history else None,
                "azimuth_range": {
                    "min": min(p.azimuth for p in self.target_history),
                    "max": max(p.azimuth for p in self.target_history)
                } if self.target_history else None
            },
            "precision_metrics": {
                "positioning_accuracy": "±1.0 degrees",
                "average_lidar_strength": np.mean(
                    [p.strength for p in self.target_history]) if self.target_history else 0,
                "average_prediction_error": np.mean(self.metrics["prediction_errors"]) if self.metrics[
                    "prediction_errors"] else None,
                "tracking_consistency": len([p for p in self.target_history if p.strength > STRENGTH_THRESHOLD]) / len(
                    self.target_history) * 100 if self.target_history else 0
            },
            "grid_search_statistics": {
                "coarse_positions_tested": len(self.grid_search_results),
                "best_coarse_strength": max(
                    p.strength for p in self.grid_search_results) if self.grid_search_results else 0,
                "average_coverage_area": np.mean(
                    [p.coverage_area for p in self.grid_search_results]) if self.grid_search_results else 0
            }
        }

        # Save report to file
        report_filename = f"tracking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[Tracker] Report saved to {report_filename}")

            # Print summary to console
            print("\n" + "=" * 60)
            print("TRACKING REPORT SUMMARY")
            print("=" * 60)
            print(f"RAAN Azimuth: {report['summary']['raan_azimuth']:.1f}°")
            print(f"Acquisition Time: {report['summary']['acquisition_time']:.1f} seconds")
            print(f"Tracking Duration: {report['summary']['tracking_duration']:.1f} seconds")
            print(f"Total Points Tracked: {report['summary']['total_tracking_points']}")
            print(f"\nVELOCITY DECOMPOSITION:")
            print(f"  Azimuth: {report['velocity_decomposition']['azimuth_velocity']:.2f}°/s")
            print(f"  Elevation: {report['velocity_decomposition']['elevation_velocity']:.2f}°/s")
            print(f"  Total: {report['velocity_decomposition']['total_velocity']:.2f}°/s")
            print(f"  Confidence: {report['velocity_decomposition']['confidence']:.1%}")
            print(f"\nPRECISION METRICS:")
            print(f"  Average Prediction Error: {report['precision_metrics']['average_prediction_error']:.2f}°")
            print(f"  Tracking Consistency: {report['precision_metrics']['tracking_consistency']:.1f}%")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"[Tracker] Error saving report: {e}")


# ==================== MAIN PROCESS FUNCTION ====================

def run_tracker_process(shared_data):
    """Main entry point for the tracking logic process"""
    print("[Tracker] Tracking logic process started")
    print("[Tracker] Initializing IntegratedAdaptiveTracker with spherical grid search...")

    tracker = IntegratedAdaptiveTracker(shared_data)

    # Wait for system initialization
    time.sleep(2.0)

    # Main control loop
    try:
        while not shared_data["shutdown"].value:
            # Check for tracking start command
            if shared_data["lidar_track_mode_active"].value and not tracker.running:
                # Get RAAN from shared data if available
                raan = shared_data.get("initial_heading", {}).value if "initial_heading" in shared_data else 0.0
                tracker.start_tracking(raan_azimuth=raan)

            # Check for tracking stop command
            elif not shared_data["lidar_track_mode_active"].value and tracker.running:
                tracker.stop_tracking()

            # Update system status
            if tracker.running:
                shared_data["tracking_logic_ready"].value = True

                # Update metrics periodically
                if time.time() - tracker.metrics["last_update"] > 1.0:
                    if tracker.velocity_components:
                        shared_data["heading_deviation"].value = tracker.velocity_components.azimuth_velocity
                        shared_data["inclination_deviation"].value = tracker.velocity_components.elevation_velocity
                    tracker.metrics["last_update"] = time.time()
            else:
                shared_data["tracking_logic_ready"].value = False

            time.sleep(0.01)  # 100 Hz main loop

    except Exception as e:
        print(f"[Tracker] Fatal error in tracker process: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if tracker.running:
            tracker.stop_tracking()
        print("[Tracker] Tracking logic process ended")


if __name__ == "__main__":
    # Test mode - create dummy shared data for testing
    from multiprocessing import Manager, Value, Array

    manager = Manager()
    test_shared_data = {
        "shutdown": Value('b', False),
        "lidar_track_mode_active": Value('b', False),
        "target_azimuth": Value('d', 0.0),
        "target_elevation": Value('d', 45.0),
        "go_to_target": Value('b', False),
        "target_reached": Value('b', False),
        "stepper_degrees": Value('d', 0.0),
        "servo_degrees": Value('d', 45.0),
        "lidar_data": Array('d', [50.0, 15000.0, time.time()]),
        "satellite_detected": Value('b', False),
        "system_status": Value('i', 0),
        "satellite_points": Array('d', [0.0, 0.0, 0.0, 0.0, 0.0]),
        "tracking_history": manager.list(),
        "estimated_azimuth": Value('d', 0.0),
        "estimated_elevation": Value('d', 0.0),
        "predicted_azimuth": Value('d', 0.0),
        "predicted_elevation": Value('d', 0.0),
        "heading_deviation": Value('d', 0.0),
        "inclination_deviation": Value('d', 0.0),
        "tracking_logic_ready": Value('b', False),
        "initial_heading": Value('d', 90.0),  # Test RAAN
    }

    print("[Test] Starting tracker in test mode...")
    print("[Test] Commands: 's' to start tracking, 't' to stop, 'q' to quit")

    # Start tracker process
    import threading

    tracker_thread = threading.Thread(target=run_tracker_process, args=(test_shared_data,))
    tracker_thread.daemon = True
    tracker_thread.start()

    # Simple command interface
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 's':
                test_shared_data["lidar_track_mode_active"].value = True
                print("[Test] Started tracking")
            elif cmd == 't':
                test_shared_data["lidar_track_mode_active"].value = False
                print("[Test] Stopped tracking")
            elif cmd == 'q':
                print("[Test] Shutting down...")
                test_shared_data["shutdown"].value = True
                break
            else:
                print("[Test] Unknown command")
    except KeyboardInterrupt:
        print("\n[Test] Interrupted")
        test_shared_data["shutdown"].value = True

    tracker_thread.join(timeout=2.0)
    print("[Test] Test completed")