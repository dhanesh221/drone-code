from pymavlink import mavutil

print("[DEBUG] Starting pymavlink script...")

# Step 1: Connect to MAVLink system
connection_string = '/dev/ttyACM0'  # or /dev/ttyACM0
master = mavutil.mavlink_connection(connection_string, baud=57600)  # common baud rate

print(f"[DEBUG] Connecting to MAVLink on: {connection_string}")
master = mavutil.mavlink_connection(connection_string)

# Step 2: Wait for heartbeat
print("[DEBUG] Waiting for heartbeat from vehicle...")
try:
    master.wait_heartbeat(timeout=30)
    print(f"[INFO] Heartbeat received — System ID: {master.target_system}, Component ID: {master.target_component}")
except Exception as e:
    print(f"[ERROR] No heartbeat received within timeout: {e}")
    exit(1)

# Step 3: Request data streams (optional but good practice)
print("[DEBUG] Requesting data stream at 1 Hz...")
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    1,  # Hz
    1   # Start streaming
)

print("[DEBUG] Entering telemetry loop. Listening for GLOBAL_POSITION_INT and SYS_STATUS messages...")
try:
    while True:
        # Listen for either GLOBAL_POSITION_INT or SYS_STATUS messages
        msg = master.recv_match(type=['GLOBAL_POSITION_INT', 'SYS_STATUS'], blocking=True, timeout=10)
        if msg is None:
            print("[WARNING] No telemetry data received within 10 seconds.")
            continue

        if msg.get_type() == 'GLOBAL_POSITION_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000  # in meters (MSL)
            rel_alt = msg.relative_alt / 1000  # relative altitude in meters
            print(f"[GPS] Lat: {lat:.7f}, Lon: {lon:.7f}, Alt (MSL): {alt:.2f} m, Rel Alt: {rel_alt:.2f} m")

        elif msg.get_type() == 'SYS_STATUS':
            voltage_battery = msg.voltage_battery / 1000 if msg.voltage_battery != -1 else None  # millivolts to volts
            current_battery = msg.current_battery / 100 if msg.current_battery != -1 else None  # 0.01 A units to amps
            battery_remaining = msg.battery_remaining  # percentage 0-100, -1 if unknown

            voltage_str = f"{voltage_battery:.2f} V" if voltage_battery is not None else "N/A"
            current_str = f"{current_battery:.2f} A" if current_battery is not None else "N/A"
            battery_str = f"{battery_remaining}%" if battery_remaining != -1 else "N/A"

            print(f"[BATTERY] Voltage: {voltage_str}, Current: {current_str}, Remaining: {battery_str}")

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
except Exception as e:
    print(f"[ERROR] Exception occurred: {e}")
finally:
    print("[INFO] Exiting pymavlink debug script.")
