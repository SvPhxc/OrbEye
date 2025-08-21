# grafana_visualisation.py - Simple LiDAR Data Publisher

import json
import time
from awscrt import mqtt
from awsiot import mqtt_connection_builder

def publish_data_to_aws(shared_data):
    """Simple function to publish only LiDAR data to AWS IoT"""
    
    # ==== CONFIGURATION ====
    ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
    CLIENT_ID = "LiDAR_Publisher"
    PATH_TO_CERT = "LockedInMartinPi.cert.pem"
    PATH_TO_KEY = "LockedInMartinPi.private.key"
    PATH_TO_ROOT_CA = "root-CA.crt"

    print("[LiDAR Publisher] Starting...")

    # Skip if disabled
    if not shared_data["grafana_enabled"].value:
        print("[LiDAR Publisher] Disabled. Exiting.")
        return

    try:
        # Connect to AWS IoT
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            cert_filepath=PATH_TO_CERT,
            pri_key_filepath=PATH_TO_KEY,
            ca_filepath=PATH_TO_ROOT_CA,
            client_id=CLIENT_ID,
            clean_session=False,
            keep_alive_secs=30
        )

        print("[LiDAR Publisher] Connecting to AWS IoT...")
        mqtt_connection.connect().result()
        print("[LiDAR Publisher] Connected!")

        # Main loop - publish LiDAR data every 2 seconds
        while not shared_data["shutdown"].value:
            try:
                # Get LiDAR data from shared memory
                # lidar_data = [distance_cm, strength, timestamp]
                lidar_data = shared_data["lidar_data"]
                
                # Only publish if we have valid data (timestamp > 0)
                if lidar_data[2] > 0:
                    payload = {
                        "distance_cm": lidar_data[0],
                        "strength": lidar_data[1],
                        "timestamp": int(time.time()),
                        "valid": shared_data["lidar_valid"].value
                    }
                    
                    # Publish to AWS IoT
                    mqtt_connection.publish(
                        topic="tracker/lidar/data", 
                        payload=json.dumps(payload), 
                        qos=mqtt.QoS.AT_LEAST_ONCE
                    )
                    print(f"[LiDAR Publisher] Published: {payload}")
                
                time.sleep(2)  # Wait 2 seconds before next publish
                
            except Exception as e:
                print(f"[LiDAR Publisher] Error: {e}")
                time.sleep(2)

    except Exception as e:
        print(f"[LiDAR Publisher] Connection failed: {e}")
    
    finally:
        try:
            if 'mqtt_connection' in locals():
                mqtt_connection.disconnect().result()
                print("[LiDAR Publisher] Disconnected")
        except:
            pass
        print("[LiDAR Publisher] Stopped")

if __name__ == "__main__":
    print("Run this as part of the main system, not standalone.")