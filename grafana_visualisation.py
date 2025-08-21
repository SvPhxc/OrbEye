import os, json, random, time, traceback
from datetime import datetime, timezone
from awscrt import mqtt
from awsiot import mqtt_connection_builder

ENDPOINT = "a1rrwkx8cway2b-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID_BASE = "basicPubSub"
PATH_TO_CERT = "LockedInMartinPi.cert.pem"
PATH_TO_KEY  = "LockedInMartinPi.private.key"
PATH_TO_ROOT_CA = "root-CA.crt"

def make_connection():
    client_id = f"{CLIENT_ID_BASE}-{os.uname().nodename}-{os.getpid()}"
    conn = mqtt_connection_builder.mtls_from_path(
        endpoint=ENDPOINT,
        cert_filepath=PATH_TO_CERT,
        pri_key_filepath=PATH_TO_KEY,
        ca_filepath=PATH_TO_ROOT_CA,
        client_id=client_id,
        clean_session=True,
        keep_alive_secs=30,
        on_connection_interrupted=lambda c, e, **kw: print(f"[MQTT] interrupted: {e}"),
        on_connection_resumed=lambda c, rc, sp, **kw: print(f"[MQTT] resumed rc={rc} session_present={sp}")
    )
    print(f"Connecting to AWS IoT as {client_id} …")
    conn.connect().result()
    print("Connected!\n")
    return conn

def current_timestamp():
    return int(time.time())

def publish_simple(conn, topic, payload_data, batch_timestamp=None):
    ts = batch_timestamp if batch_timestamp is not None else current_timestamp()
    if isinstance(payload_data, dict):
        payload_data["timestamp"] = ts
    else:
        payload_data = {"value": payload_data, "timestamp": ts}

    fut, _ = conn.publish(
        topic=topic,
        payload=json.dumps(payload_data),
        qos=mqtt.QoS.AT_LEAST_ONCE
    )
    fut.result(timeout=5.0)  # wait for PUBACK
    print(f"Published to {topic}: {json.dumps(payload_data)}")

def publish_data_to_aws(shared_data):
    print("[AWS Publisher] Starting AWS IoT publishing process...")
    conn = None
    try:
        conn = make_connection()
        while not shared_data["shutdown"].value:
            try:
                if not shared_data["grafana_enabled"].value:
                    time.sleep(1)
                    continue

                # --- provide safe defaults for missing fields ---
                lidar = shared_data.get("lidar_data", [0.0, 0.0])
                pan   = getattr(shared_data.get("stepper_degrees"), "value", 0.0)
                tilt  = getattr(shared_data.get("servo_degrees"), "value", 0.0)

                batch_ts = current_timestamp()
                simple_topics = {
                    "tracker/lidar/distance": round(lidar[0], 2),
                    "tracker/lidar/intensity": round(lidar[1], 2),
                    "tracker/tracker/pan_angle": round(pan, 2),
                    "tracker/tracker/tilt_angle": round(tilt, 2),
                    "tracker/position/x": round(random.uniform(-7000.0, 7000.0), 2),
                    "tracker/position/y": round(random.uniform(-7000.0, 7000.0), 2),
                    "tracker/position/z": round(random.uniform(-7000.0, 7000.0), 2),
                }
                for topic, payload in simple_topics.items():
                    publish_simple(conn, topic, payload, batch_ts)

                time.sleep(5)

            except Exception as batch_error:
                print(f"[AWS Publisher] Error in publishing batch: {batch_error}")
                traceback.print_exc()
                time.sleep(1)
    finally:
        print("[AWS Publisher] Shutting down...")
        try:
            if conn:
                conn.disconnect().result(timeout=2.0)
                print("[AWS Publisher] Disconnected gracefully.")
        except Exception as e:
            print(f"[AWS Publisher] Disconnect timeout or error: {e}")
        print("[AWS Publisher] AWS publishing process terminated.")

if __name__ == "__main__":
    class MockValue:
        def __init__(self, value): self.value = value

    shared_data = {
        "lidar_data": [random.uniform(300.0, 500.0), random.uniform(100.0, 300.0)],
        "stepper_degrees": MockValue(random.uniform(0.0, 180.0)),
        "servo_degrees": MockValue(random.uniform(0.0, 90.0)),
        "shutdown": MockValue(False),
        "grafana_enabled": MockValue(True),
    }
    publish_data_to_aws(shared_data)
