import serial
import time
import struct

# --- Configuration ---
# The Raspberry Pi's primary UART is typically /dev/ttyS0 or /dev/serial0
SERIAL_PORT = "/dev/ttyS0"
# The BNO086 in UART-RVC mode transmits at 115200 baud
BAUD_RATE = 115200

# UART-RVC packet details
PACKET_HEADER = b'\xaa\xaa'
PACKET_LENGTH = 32
# Scaling factors from the datasheet
ROTATION_SCALE_FACTOR = 100.0  # To get degrees
ACCEL_SCALE_FACTOR = 1000.0  # To get m/s^2


def parse_rvc_packet(packet):
    """
    Parses a 32-byte UART-RVC packet from the BNO086.

    The packet structure is (all little-endian):
    - 2 bytes: Header (0xAAAA)
    - 2 bytes: Length (32)
    - 2 bytes: Yaw (signed int)
    - 2 bytes: Pitch (signed int)
    - 2 bytes: Roll (signed int)
    - 2 bytes: X Accel (signed int)
    - 2 bytes: Y Accel (signed int)
    - 2 bytes: Z Accel (signed int)
    - ... and other reserved data
    """
    if len(packet) != PACKET_LENGTH:
        return None

    # '<' indicates little-endian byte order.
    # 'h' is a signed short (2 bytes). We need 6 of them.
    # '10h' represents the 20 remaining bytes which we ignore.
    # We unpack the first 12 bytes of data (after header and length).
    try:
        # Unpack from index 4 to 16 to get the 6 sensor values
        yaw_raw, pitch_raw, roll_raw, x_accel_raw, y_accel_raw, z_accel_raw = \
            struct.unpack('<hhhhhh', packet[4:16])

        # Apply scaling factors
        yaw = yaw_raw / ROTATION_SCALE_FACTOR
        pitch = pitch_raw / ROTATION_SCALE_FACTOR
        roll = roll_raw / ROTATION_SCALE_FACTOR
        x_accel = x_accel_raw / ACCEL_SCALE_FACTOR
        y_accel = y_accel_raw / ACCEL_SCALE_FACTOR
        z_accel = z_accel_raw / ACCEL_SCALE_FACTOR

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "x_accel": x_accel,
            "y_accel": y_accel,
            "z_accel": z_accel,
        }

    except struct.error as e:
        print(f"Error unpacking data: {e}")
        return None


# --- Main execution ---
try:
    # Initialize the serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Successfully opened serial port: {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}. {e}")
    print("Please ensure the port is correct and you have the necessary permissions.")
    print("You might need to run the script with 'sudo'.")
    exit()

print("BNO086 UART-RVC Test with pyserial")
print("Reading heading and acceleration data...")

# Buffer to hold incoming serial data
serial_buffer = bytearray()

while True:
    try:
        # Read available data from the serial port
        incoming_data = ser.read(ser.in_waiting or 1)
        if incoming_data:
            serial_buffer.extend(incoming_data)

            # Look for the packet header in our buffer
            header_index = serial_buffer.find(PACKET_HEADER)

            if header_index != -1:
                # Check if we have a complete packet in the buffer
                if len(serial_buffer) >= header_index + PACKET_LENGTH:
                    # Extract the full packet (32 bytes)
                    packet_start = header_index
                    packet_end = packet_start + PACKET_LENGTH
                    packet = serial_buffer[packet_start:packet_end]

                    # Remove the processed packet from the buffer
                    serial_buffer = serial_buffer[packet_end:]

                    # Parse the extracted packet
                    sensor_data = parse_rvc_packet(packet)

                    if sensor_data:
                        print(f"Yaw: {sensor_data['yaw']:.2f} degrees")
                        print(f"Pitch: {sensor_data['pitch']:.2f} degrees")
                        print(f"Roll: {sensor_data['roll']:.2f} degrees")
                        print(
                            f"Accel - X: {sensor_data['x_accel']:.2f}, Y: {sensor_data['y_accel']:.2f}, Z: {sensor_data['z_accel']:.2f} m/s^2")
                        print("-" * 30)

                        # Small delay to make the output readable
                        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program terminated by user.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

# Clean up by closing the serial port
ser.close()
print("Serial port closed.")