#!/usr/bin/env python3
"""
Fast and Responsive Tracking Logic
Minimal overhead, reduced overshooting, simple control
"""

import time
import math
import numpy as np
from collections import deque
from enum import Enum


class TrackingMode(Enum):
    IDLE = 0
    ORBITAL = 1
    REACTIVE = 2


class FastClutterFilter:
    """Simple fast clutter filter"""

    def __init__(self):
        self.background = {}
        self.loaded = False

        try:
            data = np.load("background_scan.npy")
            # Build simple grid
            for point in data:
                if len(point) >= 4:
                    az, el, dist, _ = point[:4]
                    key = (int(az / 5) * 5, int(el / 5) * 5)
                    if key not in self.background:
                        self.background[key] = []
                    self.background[key].append(dist)

            # Average distances
            for key in self.background:
                self.background[key] = np.mean(self.background[key])

            self.loaded = True
        except:
            pass

    def is_valid(self, az, el, dist, strength):
        """Quick validity check"""
        if strength < 100 or dist < 50 or dist > 1000:
            return False

        if not self.loaded:
            return 100 < dist < 500

        key = (int(az / 5) * 5, int(el / 5) * 5)
        if key in self.background:
            return dist < (self.background[key] - 30)

        return 100 < dist < 500


class OrbitalTracker:
    """Simple predictive tracker for circular orbit"""

    def __init__(self):
        self.tracking = False
        self.last_time = 0
        self.last_angle = 0
        self.angular_velocity = 18.0  # degrees/second (20s orbit)
        self.history = deque(maxlen=20)
        self.lost_count = 0

    def update(self, az, el, dist, strength):
        """Update with new measurement"""
        current_time = time.time()

        if not self.tracking:
            self.tracking = True
            self.last_angle = az
            self.last_time = current_time
            self.lost_count = 0
            return az, el

        # Store measurement
        self.history.append((az, el, dist, current_time))

        # Estimate velocity from recent history
        if len(self.history) >= 3:
            old = self.history[0]
            dt = current_time - old[3]
            if dt > 0:
                d_angle = az - old[0]
                if d_angle > 180:
                    d_angle -= 360
                elif d_angle < -180:
                    d_angle += 360

                # Smooth velocity update
                new_vel = d_angle / dt
                self.angular_velocity = 0.7 * self.angular_velocity + 0.3 * new_vel

        self.last_angle = az
        self.last_time = current_time
        self.lost_count = 0

        # Predict ahead slightly (50ms)
        predicted_az = (az + self.angular_velocity * 0.05) % 360

        return predicted_az, el

    def predict_lost(self):
        """Predict when target is lost"""
        if not self.tracking:
            return None

        self.lost_count += 1
        if self.lost_count > 100:  # Lost for too long
            self.tracking = False
            return None

        # Continue prediction based on last known velocity
        dt = time.time() - self.last_time
        predicted_az = (self.last_angle + self.angular_velocity * dt) % 360

        # Simple elevation model
        predicted_el = 45 + 15 * math.sin(math.radians(predicted_az * 2))
        predicted_el = max(10, min(80, predicted_el))

        return predicted_az, predicted_el


class ReactiveTracker:
    """Simple smoothed follower"""

    def __init__(self):
        self.position = None
        self.velocity = np.array([0.0, 0.0])
        self.last_time = 0
        self.smoothing = 0.3

    def update(self, az, el, dist, strength):
        """Update with smoothing"""
        current_time = time.time()

        if self.position is None:
            self.position = np.array([az, el])
            self.last_time = current_time
            return az, el

        # Calculate velocity
        dt = current_time - self.last_time
        if dt > 0:
            new_pos = np.array([az, el])

            # Handle azimuth wrap
            d_az = az - self.position[0]
            if d_az > 180:
                d_az -= 360
            elif d_az < -180:
                d_az += 360

            vel = np.array([d_az, el - self.position[1]]) / dt
            self.velocity = 0.8 * self.velocity + 0.2 * vel

        # Adaptive smoothing based on signal strength
        if strength > 500:
            smooth = 0.4  # Trust strong signals more
        else:
            smooth = 0.2  # Smooth weak signals more

        # Update position with smoothing
        target = np.array([az, el])
        self.position = (1 - smooth) * self.position + smooth * target

        self.last_time = current_time

        # Small prediction (20ms)
        predicted = self.position + self.velocity * 0.02
        predicted[0] = predicted[0] % 360
        predicted[1] = max(0, min(90, predicted[1]))

        return predicted[0], predicted[1]

    def reset(self):
        self.position = None
        self.velocity = np.array([0.0, 0.0])


class TrackingLogic:
    """Main tracking logic - fast and simple"""

    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.mode = TrackingMode.IDLE
        self.running = False

        # Trackers
        self.clutter_filter = FastClutterFilter()
        self.orbital = OrbitalTracker()
        self.reactive = ReactiveTracker()

        # Control state
        self.last_command_time = 0
        self.command_interval = 0.02  # 50Hz max command rate
        self.no_target_count = 0

        # Search state
        self.search_index = 0
        self.search_center = None

        print("[Track] Ready")

    def get_measurement(self):
        """Get current measurement"""
        with self.shared_data["lidar_data"].get_lock():
            dist, strength, _ = self.shared_data["lidar_data"][:]

        with self.shared_data["lidar_position"].get_lock():
            az, el = self.shared_data["lidar_position"][:]

        return az, el, dist, strength

    def send_command(self, az, el):
        """Send motor command with rate limiting"""
        current_time = time.time()

        # Rate limit to prevent oscillation
        if current_time - self.last_command_time < self.command_interval:
            return False

        # Check if motors are ready
        if self.shared_data["go_to_target"].value:
            return False

        # Send command
        self.shared_data["target_azimuth"].value = az
        self.shared_data["target_elevation"].value = el
        self.shared_data["go_to_target"].value = True

        self.last_command_time = current_time
        return True

    def search_pattern(self):
        """Simple search pattern"""
        if self.search_center is None:
            return None

        # Small spiral search
        patterns = [
            (0, 0), (5, 0), (5, 5), (0, 5), (-5, 5),
            (-5, 0), (-5, -5), (0, -5), (5, -5),
            (10, 0), (0, 10), (-10, 0), (0, -10)
        ]

        if self.search_index >= len(patterns):
            self.search_index = 0

        offset = patterns[self.search_index]
        self.search_index += 1

        target_az = (self.search_center[0] + offset[0]) % 360
        target_el = max(0, min(90, self.search_center[1] + offset[1]))

        return target_az, target_el

    def run(self):
        """Main loop - fast and simple"""
        self.running = True
        self.shared_data["tracking_logic_ready"].value = True

        while self.running and not self.shared_data["shutdown"].value:
            try:
                # Get measurement
                az, el, dist, strength = self.get_measurement()

                # Determine mode
                if self.shared_data["debug_mode"].value:
                    new_mode = TrackingMode.ORBITAL
                elif self.shared_data["reactive_mode"].value:
                    new_mode = TrackingMode.REACTIVE
                else:
                    new_mode = TrackingMode.IDLE

                # Mode change
                if new_mode != self.mode:
                    self.mode = new_mode
                    if self.mode == TrackingMode.ORBITAL:
                        self.orbital = OrbitalTracker()
                    elif self.mode == TrackingMode.REACTIVE:
                        self.reactive.reset()

                # Process based on mode
                if self.mode != TrackingMode.IDLE:
                    # Check if valid target
                    valid = dist > 0 and self.clutter_filter.is_valid(az, el, dist, strength)

                    if valid:
                        # Target detected
                        self.no_target_count = 0
                        self.search_center = (az, el)
                        self.search_index = 0

                        # Update tracker
                        if self.mode == TrackingMode.ORBITAL:
                            target_az, target_el = self.orbital.update(az, el, dist, strength)
                        else:
                            target_az, target_el = self.reactive.update(az, el, dist, strength)

                        # Send command
                        if self.send_command(target_az, target_el):
                            self.shared_data["predicted_azimuth"].value = target_az
                            self.shared_data["predicted_elevation"].value = target_el

                    else:
                        # No target
                        self.no_target_count += 1

                        if self.no_target_count < 50:  # Search for 50 cycles
                            # Try prediction or search
                            if self.mode == TrackingMode.ORBITAL:
                                pred = self.orbital.predict_lost()
                                if pred:
                                    self.send_command(pred[0], pred[1])
                            elif self.search_center:
                                search = self.search_pattern()
                                if search:
                                    self.send_command(search[0], search[1])
                        else:
                            # Reset after timeout
                            if self.mode == TrackingMode.ORBITAL:
                                self.orbital = OrbitalTracker()
                            else:
                                self.reactive.reset()
                            self.search_center = None
                            self.no_target_count = 0

                # Fast loop
                time.sleep(0.001)

            except Exception as e:
                time.sleep(0.01)

    def stop(self):
        self.running = False


def run_tracker_process(shared_data):
    """Entry point"""
    tracker = None
    try:
        tracker = TrackingLogic(shared_data)
        tracker.run()
    except Exception as e:
        print(f"[Track] Error: {e}")
    finally:
        if tracker:
            tracker.stop()