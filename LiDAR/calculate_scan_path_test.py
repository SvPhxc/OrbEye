import math

def calculate_scan_path(delta_azimuth, distance_meters, initial_pan_angle, initial_tilt_angle):
    """
    Generates a series of (pan, tilt) angle pairs for scanning motion in a semi-circular arc.
    The system moves uniformly along both azimuth and tilt to form a 3D semi-circular path.
    
    Args:
        delta_azimuth (float): Total horizontal angle to scan in degrees
        distance_meters (float): Distance to target object (2-12 meters)
        initial_pan_angle (float): Starting pan angle (azimuth) in degrees
        initial_tilt_angle (float): Starting tilt angle (elevation) in degrees
    
    Returns:
        list: List of (pan_angle, tilt_angle) tuples representing the semi-circular scan path
    """
    
    # Validate input parameters
    if distance_meters < 2 or distance_meters > 12:
        raise ValueError("distance_meters must be between 2 and 12 meters")
    
    if delta_azimuth <= 0:
        raise ValueError("delta_azimuth must be positive")
    
    # Calculate number of points using inverse linear relationship
    # At distance = 2m: max_points = delta_azimuth / 3.2
    # At distance = 12m: min_points should be significantly less
    
    max_points = delta_azimuth / 3.2  # Points at 2 meters
    min_points = max_points * 0.2     # Points at 12 meters (20% of max)
    
    # Linear interpolation: as distance increases from 2 to 12, points decrease from max to min
    # Formula: points = max_points - (distance - 2) * (max_points - min_points) / (12 - 2)
    num_points = max_points - (distance_meters - 2) * (max_points - min_points) / 10
    
    # Ensure we have at least 2 points and round to nearest integer
    num_points = max(2, round(num_points))
    
    # Generate the scan path
    scan_path = []
    
    # If we only have one point, just return the initial position
    if num_points == 1:
        return [(initial_pan_angle, initial_tilt_angle)]
    
    # For semi-circular motion, we'll create an arc where:
    # - Pan moves linearly across delta_azimuth
    # - Tilt follows a semi-circular pattern
    
    # Calculate the radius of the tilt arc based on delta_azimuth
    # The tilt will vary in a semi-circular pattern with amplitude proportional to delta_azimuth
    tilt_amplitude = delta_azimuth * 0.3  # 30% of azimuth range for tilt variation
    
    # Generate points along the semi-circular arc
    for i in range(num_points):
        # Progress parameter from 0 to 1
        progress = i / (num_points - 1) if num_points > 1 else 0
        
        # Calculate pan angle (linear progression)
        pan_angle = initial_pan_angle + (progress * delta_azimuth)
        
        # Calculate tilt angle (semi-circular progression)
        # Use sine function to create semi-circular motion
        # Progress from 0 to π for a semi-circle
        arc_angle = progress * math.pi
        tilt_offset = math.sin(arc_angle) * tilt_amplitude
        tilt_angle = initial_tilt_angle + tilt_offset
        
        scan_path.append((pan_angle, tilt_angle))
    
    return scan_path


def print_scan_info(delta_azimuth, distance_meters, initial_pan_angle, initial_tilt_angle):
    """
    Helper function to display scan path information and results.
    """
    print(f"Scan Parameters:")
    print(f"  Delta Azimuth: {delta_azimuth}°")
    print(f"  Distance: {distance_meters}m")
    print(f"  Initial Pan: {initial_pan_angle}°")
    print(f"  Initial Tilt: {initial_tilt_angle}°")
    print()
    
    path = calculate_scan_path(delta_azimuth, distance_meters, initial_pan_angle, initial_tilt_angle)
    
    # Calculate tilt range for the semi-circular arc
    tilt_values = [tilt for _, tilt in path]
    tilt_range = max(tilt_values) - min(tilt_values)
    
    print(f"Generated {len(path)} scan points (Semi-circular Arc):")
    print(f"  Tilt variation range: {tilt_range:.1f}°")
    for i, (pan, tilt) in enumerate(path):
        print(f"  Point {i+1}: Pan={pan:.1f}°, Tilt={tilt:.1f}°")
    print()

def execute_scan_sequence(scan_path):
    """
    Processes scan path sequentially and returns movement commands.
    
    Args:
        scan_path (list): List of (pan_angle, tilt_angle) tuples
        
    Returns:
        list: List of tuples (direction, degrees) where direction is string and degrees is int
    """
    
    if len(scan_path) < 2:
        return []
    
    commands = []
    
    # Process each step sequentially like a while loop
    step_index = 0
    while step_index < len(scan_path) - 1:
        prev_pan, prev_tilt = scan_path[step_index]
        curr_pan, curr_tilt = scan_path[step_index + 1]
        
        # Calculate pan movement
        pan_change = curr_pan - prev_pan
        if pan_change > 0:
            commands.append(("right", round(abs(pan_change))))
        elif pan_change < 0:
            commands.append(("left", round(abs(pan_change))))
        
        # Calculate tilt movement  
        tilt_change = curr_tilt - prev_tilt
        if tilt_change > 0:
            commands.append(("up", round(abs(tilt_change))))
        elif tilt_change < 0:
            commands.append(("down", round(abs(tilt_change))))
            
        step_index += 1
    
    return commands
