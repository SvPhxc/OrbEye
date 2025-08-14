import json
import random
import time
from datetime import datetime, timezone
from awscrt import mqtt
from awsiot import mqtt_connection_builder

# ==== CONFIGURATION ====
ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID = "basicPubSub"
PATH_TO_CERT = "LockedInMartinPi.cert.pem"
PATH_TO_KEY = "LockedInMartinPi.private.key"
PATH_TO_ROOT_CA = "root-CA.crt"

# Connect to AWS IoT
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=ENDPOINT,
    cert_filepath=PATH_TO_CERT,
    pri_key_filepath=PATH_TO_KEY,
    ca_filepath=PATH_TO_ROOT_CA,
    client_id=CLIENT_ID,
    clean_session=False,
    keep_alive_secs=6
)

print("Connecting to AWS IoT...")
mqtt_connection.connect().result()
print("Connected! Publishing all data types every 15 seconds...")

def publish_data(topic, value):
    payload = {
        "value": value,
        "timestamp": int(time.time())
    }
    mqtt_connection.publish(
        topic=topic,
        payload=json.dumps(payload),
        qos=mqtt.QoS.AT_LEAST_ONCE
    )
    print(f"Published to {topic}: {json.dumps(payload)}")

def generate_cpf():
    return {
        "format_version": "1.0",
        "prediction_source": "SGP4",
        "prediction_epoch": datetime.now(timezone.utc).isoformat(),
        "position_vector": {
            "x": round(random.uniform(-7000, 7000), 3),
            "y": round(random.uniform(-7000, 7000), 3),
            "z": round(random.uniform(-7000, 7000), 3)
        },
        "velocity_vector": {
            "x": round(random.uniform(-10, 10), 6),
            "y": round(random.uniform(-10, 10), 6),
            "z": round(random.uniform(-10, 10), 6)
        },
        "orbital_period": round(random.uniform(5000, 6000), 1),
        "semi_major_axis": round(random.uniform(6700, 7000), 1)
    }

def generate_nps():
    return {
        "time_of_flight": round(random.uniform(0.02, 0.04), 6),
        "range_correction": round(random.uniform(-0.2, 0.2), 4),
        "atmospheric_correction": round(random.uniform(-0.01, 0.01), 4),
        "calibration_factor": 1.0,
        "measurement_timestamp": datetime.now(timezone.utc).isoformat(),
        "return_strength": round(random.uniform(0.2, 0.5), 3),
        "background_noise": round(random.uniform(0.0005, 0.003), 5)
    }

def generate_tle():
    return {
        "line1": "1 25544U 98067A 25210.50000000 .00002182 00000-0 10270-4 0 9999",
        "line2": "2 25544 51.6461 339.7939 0001882 83.2919 276.8737 15.48919103999999",
        "satellite_name": "ISS (ZARYA)",
        "epoch": datetime.now(timezone.utc).isoformat(),
        "mean_motion": 15.48919103,
        "eccentricity": 0.0001882,
        "inclination": 51.6461
    }
def generate_realistic_position():
    # ISS orbital parameters (approximate)
    orbital_radius = random.uniform(6700, 7000)  # km from Earth center
    
    # Generate position on orbital sphere
    import math
    
    # Random orbital position (simplified)
    theta = random.uniform(0, 2 * math.pi)  # orbital angle
    phi = random.uniform(-math.radians(51.6), math.radians(51.6))  # ISS inclination
    
    x = orbital_radius * math.cos(phi) * math.cos(theta)
    y = orbital_radius * math.cos(phi) * math.sin(theta)
    z = orbital_radius * math.sin(phi)
    
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "z": round(z, 3)
    }

# Continuous loop
try:
    while True:
        print(f"\n=== Publishing cycle at {datetime.now()} ===")
        position = generate_realistic_position()
        x, y, z = position["x"], position["y"], position["z"]
        # All data types from your original code
        data_to_publish = {
            "tracker/orbit/cpf": generate_cpf(),
            "tracker/lidar/distance": round(random.uniform(300.0, 500.0), 2),
            "tracker/lidar/intensity": round(random.uniform(100.0, 300.0), 2),
            "tracker/orbit/nps": generate_nps(),
            "tracker/tracker/pan_angle": round(random.uniform(0.0, 180.0), 2),
            "tracker/tracker/tilt_angle": round(random.uniform(0.0, 90.0), 2),
            "tracker/orbit/tle": generate_tle(),
            "tracker/orbit/x": x,
            "tracker/orbit/y": y,
            "tracker/orbit/z": z
        }

        # Publish all data
        for topic, value in data_to_publish.items():
            publish_data(topic, value)
            time.sleep(0.5)  # Small delay between messages

        print("Waiting 15 seconds before next cycle...")
        time.sleep(3)

except KeyboardInterrupt:
    print("\nStopping...")
    mqtt_connection.disconnect().result()
