import json
import time
import os
from awscrt import mqtt
from awsiot import mqtt_connection_builder

# ==== CONFIGURATION ====
ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID = "basicPubSub"
PATH_TO_CERT = "LockedInMartinPi.cert.pem"
PATH_TO_KEY = "LockedInMartinPi.private.key"
PATH_TO_ROOT_CA = "root-CA.crt"
TOPIC = "/tracker/test"

def on_connection_interrupted(connection, error, **kwargs):
    print(f"Connection interrupted: {error}")

def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print(f"Connection resumed: return_code={return_code}, session_present={session_present}")
    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist, resubscribing...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()
        resubscribe_future.add_done_callback(lambda f: print(f"Resubscribe results: {f.result()}"))

def on_message_received(topic, payload, **kwargs):
    print(f"Received message from topic '{topic}': {payload}")

def main():
    # Verify certificate files exist
    cert_files = [PATH_TO_CERT, PATH_TO_KEY, PATH_TO_ROOT_CA]
    for cert_file in cert_files:
        if not os.path.exists(cert_file):
            print(f"Error: Certificate file not found: {cert_file}")
            return
        print(f"Found certificate file: {cert_file}")

    try:
        # Create MQTT connection
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            cert_filepath=PATH_TO_CERT,
            pri_key_filepath=PATH_TO_KEY,
            ca_filepath=PATH_TO_ROOT_CA,
            client_id=CLIENT_ID,
            clean_session=False,  # Try setting this to True if issues persist
            keep_alive_secs=30,
            on_connection_interrupted=on_connection_interrupted,
            on_connection_resumed=on_connection_resumed,
        )

        print("Connecting to AWS IoT...")
        connect_future = mqtt_connection.connect()
        connect_future.result(timeout=10)  # Add timeout
        print("Connected!")

        # Subscribe to topic
        print(f"Subscribing to topic {TOPIC}...")
        subscribe_future, _ = mqtt_connection.subscribe(
            topic=TOPIC,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message_received
        )
        subscribe_future.result(timeout=10)  # Add timeout
        print("Subscribed!")

        # Publish message
        message = json.dumps({
            "message": "hello world", 
            "timestamp": int(time.time()),
            "client_id": CLIENT_ID
        })
        print(f"Publishing to topic {TOPIC}: {message}")
        
        publish_future, packet_id = mqtt_connection.publish(
            topic=TOPIC, 
            payload=message, 
            qos=mqtt.QoS.AT_LEAST_ONCE
        )
        publish_future.result(timeout=10)  # Add timeout
        print(f"Published! Packet ID: {packet_id}")

        # Wait to receive the message back
        print("Waiting for messages...")
        time.sleep(5)

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Disconnecting...")
        try:
            disconnect_future = mqtt_connection.disconnect()
            disconnect_future.result(timeout=5)
            print("Disconnected.")
        except Exception as e:
            print(f"Error during disconnect: {e}")

if __name__ == "__main__":
    main()
