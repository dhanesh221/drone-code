from pymavlink import mavutil, mavwp
import time


def connect_pixhawk(connection_string, baudrate=57600):
    print(f"[INFO] Connecting to Pixhawk on {connection_string} at {baudrate} baud...")
    master = mavutil.mavlink_connection(connection_string, baud=baudrate)
    master.wait_heartbeat()
    print(f"[✅] Connected (System {master.target_system}, Component {master.target_component})")
    return master


def clear_mission(master):
    print("[INFO] Clearing existing mission...")
    master.waypoint_clear_all_send()
    print("[✅] Mission cleared.")
    time.sleep(2)  # Give Pixhawk a moment to process


def parse_waypoints_file(filepath):
    print(f"[INFO] Reading waypoints from: {filepath}")
    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines[0].startswith("QGC WPL"):
        raise ValueError("❌ Invalid waypoint file format (expected QGC WPL header)")

    mission = mavwp.MAVWPLoader()

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 12:
            continue  # skip malformed line

        seq = int(parts[0])
        current = int(parts[1])
        frame = int(parts[2])
        command = int(parts[3])
        params = list(map(float, parts[4:8]))
        lat = float(parts[8])
        lon = float(parts[9])
        alt = float(parts[10])
        autocontinue = int(parts[11])

        print(f"[DEBUG] WP#{seq}: CMD={command}, Lat={lat}, Lon={lon}, Alt={alt}")

        msg = mavutil.mavlink.MAVLink_mission_item_message(
            0, 0,  # target_system/component will be set later
            seq, frame, command,
            current, autocontinue,
            *params, lat, lon, alt
        )
        mission.add(msg)

    print(f"[✅] Parsed {mission.count()} waypoints.")
    return mission


def upload_mission(master, mission):
    print("[INFO] Uploading mission to Pixhawk...")
    time.sleep(1)
    master.waypoint_clear_all_send()
    time.sleep(1)
    master.waypoint_count_send(mission.count())

    for i in range(mission.count()):
        print(f"[DEBUG] Waiting for MISSION_REQUEST for WP#{i}")
        msg = master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=10)
        if msg and msg.seq == i:
            wp = mission.wp(i)
            wp.target_system = master.target_system
            wp.target_component = master.target_component
            master.mav.send(wp)
            print(f"[✅] Sent WP#{i}")
        else:
            raise RuntimeError(f"❌ MISSION_REQUEST timeout or mismatch at seq {i}")

    # Wait for ACK
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
    if ack and ack.type == 0:
        print("[✅] Mission upload successful.")
    else:
        raise RuntimeError("❌ Mission ACK not received or failed.")


def arm_and_set_mode(master, mode='AUTO'):
    print(f"[INFO] Setting mode: {mode}")
    mode_id = master.mode_mapping().get(mode)
    if mode_id is None:
        raise ValueError(f"❌ Unknown mode: {mode}")
    master.set_mode(mode_id)

    print("[INFO] Arming drone...")
    master.arducopter_arm()
    master.motors_armed_wait()
    print("[✅] Drone armed.")


def start_mission(master):
    print("[INFO] Starting mission...")
    master.set_mission_current(0)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_MISSION_START,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    print("[🚀] Mission started.")


def main():
    # CHANGE THIS to your .waypoints file path
    waypoints_file = "mission.waypoints"

    # CHANGE this to your serial port (e.g., "/dev/ttyUSB0")
    connection_string = "/dev/ttyACM0"

    master = connect_pixhawk(connection_string)

    clear_mission(master)

    mission = parse_waypoints_file(waypoints_file)

    upload_mission(master, mission)

    arm_and_set_mode(master)

    start_mission(master)


if __name__ == "__main__":
    main()
