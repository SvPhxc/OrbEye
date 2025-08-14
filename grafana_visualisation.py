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

# ==== MQTT CONNECT ====
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
print("Connected!\n")

# ==== UTILITIES ====
def current_timestamp():
    return int(time.time())

def publish_simple(topic, payload_data):
    """Simplified publish function with direct payload"""
    # Add timestamp to the payload
    if isinstance(payload_data, dict):
        payload_data["timestamp"] = current_timestamp()
    else:
        payload_data = {
            "value": payload_data,
            "timestamp": current_timestamp()
        }
    
    mqtt_connection.publish(
        topic=topic, 
        payload=json.dumps(payload_data), 
        qos=mqtt.QoS.AT_LEAST_ONCE
    )
    print(f"Published to {topic}: {json.dumps(payload_data, indent=2)}")

def publish_original(topic, value):
    """Original publish function - fixed logic"""
    if isinstance(value, dict):
        # For dictionaries, send the dict directly as the value
        payload = {
            "value": value,
            "timestamp": {"timeInSeconds": current_timestamp()},
            "quality": "GOOD"
        }
    else:
        # For strings and numbers, use the appropriate type
        value_key = "stringValue" if isinstance(value, str) else "doubleValue"
        payload = {
            "value": {value_key: value},
            "timestamp": {"timeInSeconds": current_timestamp()},
            "quality": "GOOD"
        }
    
    mqtt_connection.publish(topic=topic, payload=json.dumps(payload), qos=mqtt.QoS.AT_LEAST_ONCE)
    print(f"Published to {topic}: {json.dumps(payload)}")

# ==== FAKE DATA GENERATORS ====
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

def publish_data_to_aws(shared_data):
    # Establish the connection once before the main loop.
    # We will assume 'mqtt_connection' is an accessible global object.
    mqtt_connection.connect().result()
    print("MQTT connection established.")

    try:
        while True:
            # Simple approach - direct payloads
            simple_topics = {
                #"tracker/orbit/cpf": shared_data["cpf"].value,
                "tracker/lidar/distance": shared_data["lidar_data"][0],
                "tracker/lidar/intensity": shared_data["lidar_data"][1],
                #"tracker/orbit/nps": shared_data["nps"].value,
                "tracker/tracker/pan_angle": shared_data["stepper_degrees"].value,
                "tracker/tracker/tilt_angle": shared_data["servo_degrees"].value
                #"tracker/orbit/altitude": shared_data["altitude"].value
                #"tracker/orbit/tle": generate_tle(),
                #"tracker/position/x": round(random.uniform(-7000.0, 7000.0), 2),
                #"tracker/position/y": round(random.uniform(-7000.0, 7000.0), 2),
                #"tracker/position/z": round(random.uniform(-7000.0, 7000.0), 2)
            }
            
            for topic, payload in simple_topics.items():
                publish_simple(topic, payload)
            
            print("\nAll messages published for this cycle.")

            # A longer pause here to control the overall publishing frequency
            time.sleep(5) 

    except KeyboardInterrupt:
        print("\n\nUser terminated the process.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Disconnecting from MQTT...")
        mqtt_connection.disconnect().result()
        print("Disconnected.")








# # Original approach with fixed logic
# for topic, value in simple_topics.items():
#     publish_original(f"original/{topic}", value)
#     time.sleep(1)


