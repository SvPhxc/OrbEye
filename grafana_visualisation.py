# grafana_visualisation.py

import json
import time
from datetime import datetime, timezone
from awscrt import mqtt
from awsiot import mqtt_connection_builder

def publish_data_to_aws(shared_data):
    """Main process function to publish real-time data to AWS IoT for Grafana visualization"""
    
    # ==== CONFIGURATION ====
    ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
    CLIENT_ID = "SatelliteTracker_Grafana"
    PATH_TO_CERT = "LockedInMartinPi.cert.pem"
    PATH_TO_KEY = "LockedInMartinPi.private.key"
    PATH_TO_ROOT_CA = "root-CA.crt"

    print("[GrafanaVis] Starting Grafana visualization process...")

    # Check if Grafana is enabled
    if not shared_data["grafana_enabled"].value:
        print("[GrafanaVis] Grafana visualization is disabled. Exiting.")
        return

    try:
        # ==== MQTT CONNECT ====
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            cert_filepath=PATH_TO_CERT,
            pri_key_filepath=PATH_TO_KEY,
            ca_filepath=PATH_TO_ROOT_CA,
            client_id=CLIENT_ID,
            clean_session=False,
            keep_alive_secs=30
        )

        print("[GrafanaVis] Connecting to AWS IoT...")
        mqtt_connection.connect().result()
        print("[GrafanaVis] Connected to AWS IoT!")

        # ==== UTILITIES ====
        def current_timestamp():
            return int(time.time())

        def publish_data(topic, payload_data):
            """Publish data with timestamp"""
            try:
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
                print(f"[GrafanaVis] Published to {topic}: {payload_data}")
            except Exception as e:
                print(f"[GrafanaVis] Error publishing to {topic}: {e}")

        def get_tle_data():
            """Extract TLE data from shared memory"""
            try:
                tle_string = shared_data["generated_tle"].value.decode().strip()
                if tle_string and tle_string != "No TLE generated yet.":
                    # Parse TLE string if it contains actual TLE data
                    lines = tle_string.split('\n')
                    if len(lines) >= 2:
                        return {
                            "line1": lines[0].strip(),
                            "line2": lines[1].strip(),
                            "satellite_name": "TRACKED_SATELLITE",
                            "epoch": datetime.now(timezone.utc).isoformat(),
                            "generated": True
                        }
                return {
                    "status": "No TLE available",
                    "generated": False,
                    "epoch": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                print(f"[GrafanaVis] Error reading TLE data: {e}")
                return {"error": str(e), "generated": False}

        def get_system_status():
            """Get comprehensive system status"""
            return {
                "system_state": shared_data["system_state"].value,
                "tracking_active": shared_data["lidar_track_mode_active"].value,
                "satellite_detected": shared_data["satellite_detected"].value,
                "target_reached": shared_data["target_reached"].value,
                "ekf_running": shared_data["ekf_running"].value,
                "ekf_initialized": shared_data["ekf_initialized"].value,
                "ekf_confidence": shared_data["ekf_confidence"].value,
                "orbit_patrol_active": shared_data["orbit_patrol_active"].value,
                "background_scan_active": shared_data["background_scan_active"].value,
                "scan_progress": shared_data["scan_progress"].value,
                "debug_mode": shared_data["debug_mode"].value
            }

        # ==== MAIN LOOP ====
        print("[GrafanaVis] Starting data publishing loop...")
        last_publish_time = 0
        publish_interval = 2.0  # Publish every 2 seconds

        while not shared_data["shutdown"].value:
            try:
                current_time = time.time()
                
                if current_time - last_publish_time >= publish_interval:
                    # === HARDWARE DATA ===
                    # Servo position (elevation)
                    publish_data("tracker/hardware/servo_degrees", shared_data["servo_degrees"].value)
                    
                    # Stepper position (azimuth)
                    publish_data("tracker/hardware/stepper_degrees", shared_data["stepper_degrees"].value)
                    
                    # Target positions
                    publish_data("tracker/targets/azimuth", shared_data["target_azimuth"].value)
                    publish_data("tracker/targets/elevation", shared_data["target_elevation"].value)

                    # === LIDAR DATA ===
                    lidar_data = shared_data["lidar_data"]
                    if lidar_data[2] > 0:  # Check if timestamp is valid
                        publish_data("tracker/lidar/distance_cm", lidar_data[0])
                        publish_data("tracker/lidar/strength", lidar_data[1])
                        publish_data("tracker/lidar/valid", shared_data["lidar_valid"].value)

                    # === TRACKING DATA ===
                    if shared_data["satellite_detected"].value:
                        satellite_points = shared_data["satellite_points"]
                        satellite_data = {
                            "azimuth": satellite_points[0],
                            "elevation": satellite_points[1],
                            "distance_cm": satellite_points[2],
                            "strength": satellite_points[3],
                            "detection_timestamp": satellite_points[4]
                        }
                        publish_data("tracker/satellite/detected", satellite_data)

                    # === EKF PREDICTIONS ===
                    if shared_data["ekf_running"].value:
                        ekf_data = {
                            "predicted_azimuth": shared_data["predicted_azimuth"].value,
                            "predicted_elevation": shared_data["predicted_elevation"].value,
                            "estimated_azimuth": shared_data["estimated_azimuth"].value,
                            "estimated_elevation": shared_data["estimated_elevation"].value,
                            "confidence": shared_data["ekf_confidence"].value,
                            "initialized": shared_data["ekf_initialized"].value
                        }
                        publish_data("tracker/ekf/state", ekf_data)

                    # === SYSTEM STATUS ===
                    publish_data("tracker/system/status", get_system_status())

                    # === TLE DATA ===
                    tle_data = get_tle_data()
                    publish_data("tracker/orbit/tle", tle_data)

                    # === OBSERVER LOCATION ===
                    if shared_data["observer_lat"].value != 0.0 or shared_data["observer_lon"].value != 0.0:
                        observer_data = {
                            "latitude": shared_data["observer_lat"].value,
                            "longitude": shared_data["observer_lon"].value,
                            "altitude_m": shared_data["observer_alt"].value
                        }
                        publish_data("tracker/observer/location", observer_data)

                    # === PERFORMANCE METRICS ===
                    performance_data = {
                        "total_movements": shared_data["total_movements"].value,
                        "lidar_reads": shared_data["lidar_reads"].value,
                        "movement_requests": shared_data["movement_request_id"].value,
                        "last_state_change": shared_data["last_state_change"].value
                    }
                    publish_data("tracker/performance/metrics", performance_data)

                    last_publish_time = current_time

                time.sleep(0.1)  # Small sleep to prevent excessive CPU usage

            except Exception as e:
                print(f"[GrafanaVis] Error in main loop: {e}")
                time.sleep(1)  # Wait before retrying

    except Exception as e:
        print(f"[GrafanaVis] Failed to initialize MQTT connection: {e}")
    
    finally:
        try:
            if 'mqtt_connection' in locals():
                print("[GrafanaVis] Disconnecting from AWS IoT...")
                mqtt_connection.disconnect().result()
                print("[GrafanaVis] Disconnected from AWS IoT")
        except Exception as e:
            print(f"[GrafanaVis] Error during disconnect: {e}")
        
        print("[GrafanaVis] Grafana visualization process terminated")

if __name__ == "__main__":
    # This allows the module to be tested independently
    print("This module should be run as part of the main tracking system.")
    print("To test independently, you would need to create mock shared_data.")