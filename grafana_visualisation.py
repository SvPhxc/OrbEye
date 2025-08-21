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

def publish_simple(topic, payload_data, batch_timestamp=None):
    """Simplified publish function with direct payload"""
    # Use provided batch timestamp or generate new one
    timestamp = batch_timestamp if batch_timestamp is not None else current_timestamp()
    
    # Add timestamp to the payload
    if isinstance(payload_data, dict):
        payload_data["timestamp"] = timestamp
    else:
        payload_data = {
            "value": payload_data,
            "timestamp": timestamp
        }
    
    mqtt_connection.publish(
        topic=topic, 
        payload=json.dumps(payload_data), 
        qos=mqtt.QoS.AT_LEAST_ONCE
    )
    print(f"Published to {topic}: {json.dumps(payload_data, indent=2)}")

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
    try:
        # Check shutdown flag before doing anything
        if shared_data.get("shutdown") and shared_data["shutdown"].value:
            print("Shutdown flag is set. Exiting immediately.")
            return
            
        # Get a single timestamp for the entire batch
        batch_timestamp = current_timestamp()
        
        # Simple approach - direct payloads from shared data
        simple_topics = {
            "tracker/lidar/distance": round(shared_data["lidar_data"][0], 2),
            "tracker/lidar/intensity": round(shared_data["lidar_data"][1], 2),
            "tracker/tracker/pan_angle": round(shared_data["stepper_degrees"].value, 2),
            "tracker/tracker/tilt_angle": round(shared_data["servo_degrees"].value, 2),
            #"tracker/orbit/tle": shared_data["tle"].value,
            "tracker/position/x": round(shared_data["position_x"], 2), 
            "tracker/position/y": round(shared_data["position_y"], 2),  
            "tracker/position/z": round(shared_data["position_z"], 2)   
        }
        
        # Publish all topics at the same time with the same timestamp
        for topic, payload in simple_topics.items():
            # Check shutdown flag during publishing loop
            if shared_data.get("shutdown") and shared_data["shutdown"].value:
                print("Shutdown flag detected during publishing. Stopping...")
                break
            publish_simple(topic, payload, batch_timestamp)
        
        print("\nAll messages published successfully!")
        
    except KeyboardInterrupt:
        print("\n\nUser terminated the process (Ctrl+C).")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Graceful disconnect with timeout
        print("Disconnecting from MQTT...")
        try:
            # Use a short timeout for quick disconnect
            disconnect_future = mqtt_connection.disconnect()
            disconnect_future.result(timeout=2.0)  # 2 second timeout
            print("Disconnected gracefully.")
        except Exception as disconnect_error:
            print(f"Disconnect timeout or error: {disconnect_error}")
            print("Force closing connection...")
        
        print("Shutdown complete.")

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    # Mock shared_data structure for testing
    class MockValue:
        def __init__(self, value):
            self.value = value

    # Create test shared_data (replace with your actual shared data source)
    shared_data = {
        "lidar_data": [round(random.uniform(300.0, 500.0), 2), round(random.uniform(100.0, 300.0), 2)],
        "stepper_degrees": MockValue(round(random.uniform(0.0, 180.0), 2)),
        "servo_degrees": MockValue(round(random.uniform(0.0, 90.0), 2)),
        "tle": MockValue(generate_tle()),
        "position_x": round(random.uniform(-7000.0, 7000.0), 2),
        "position_y": round(random.uniform(-7000.0, 7000.0), 2),
        "position_z": round(random.uniform(-7000.0, 7000.0), 2),
        "shutdown": MockValue(False)  # Added shutdown flag
    }
    
    # Test the function once (like the working version)
    publish_data_to_aws(shared_data)