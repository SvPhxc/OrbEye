# grafana_visualisation.py - Simple LiDAR Data Publisher

import json
import time
from awscrt import mqtt
from awsiot import mqtt_connection_builder


def publish_data_to_aws(shared_data):
    """Simple function to publish only LiDAR data to AWS IoT"""
    
    # ==== CONFIGURATION ====
    ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
    CLIENT_ID = "basicPubSub"
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
        time.sleep(10)  # Wait for other processes to start
        print("[LiDAR Publisher] Waiting to connect to AWS IoT...")
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
                lidar_distance = lidar_data[0]
                lidar_intensity = lidar_data[1]
                stepper_degrees = round(shared_data["stepper_degrees"].value,2)
                servo_degrees = round(shared_data["servo_degrees"].value,2)
                current_time =  int(time.time())
                
                simple_topics = {
                    #"tracker/orbit/cpf": generate_cpf(),
                    "tracker/lidar/distance": lidar_distance,
                    "tracker/lidar/intensity": lidar_intensity,
                    #"tracker/orbit/nps": generate_nps(),
                    "tracker/tracker/pan_angle": stepper_degrees,
                    "tracker/tracker/tilt_angle": servo_degrees
                    #"tracker/orbit/tle": generate_tle(),
                    #"tracker/position/x": round(random.uniform(-7000.0, 7000.0), 2),
                    #"tracker/position/y": round(random.uniform(-7000.0, 7000.0), 2),
                    #"tracker/position/z": round(random.uniform(-7000.0, 7000.0), 2)
                }
                
                
                # Only publish if we have valid data (timestamp > 0)
                # if lidar_data[2] > 0:
                #     payload = {
                #         "distance_cm": lidar_data[0],
                #         "strength": lidar_data[1],
                #         "timestamp": int(time.time()),
                #         "valid": shared_data["lidar_valid"].value
                #     }
                    
                #     # Publish to AWS IoT
                #     mqtt_connection.publish(
                #         topic="tracker/lidar/data", 
                #         payload=json.dumps(payload), 
                #         qos=mqtt.QoS.AT_LEAST_ONCE
                #     )
                #     print(f"[LiDAR Publisher] Published: {payload}")
                
                # time.sleep(2)  # Wait 2 seconds before next publish
                
                
                
                for topic, payload in simple_topics.items():
                    if isinstance(payload, dict):
                        payload["timestamp"] = current_time
                    else:
                        payload = {
                            "value": payload,
                            "timestamp": current_time
                            }
                    mqtt_connection.publish(
                        topic=topic, 
                        payload=json.dumps(payload), 
                        qos=mqtt.QoS.AT_LEAST_ONCE)
                time.sleep(2)
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