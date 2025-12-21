import asyncio
import json
import math
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan


# ---------- CONFIGURATION YOU SHOULD TUNE ----------
ALTITUDE_M = 10.0          # flight altitude (relative)
TRACK_SPACING_M = 8.0      # distance between passes on ground (meters)
CRUISE_SPEED_M_S = 5.0     # sensible cruise speed, not 1000 m/s
ACCEPTANCE_RADIUS_M = 1.0
# ---------------------------------------------------


def deg2rad(d):
    return d * math.pi / 180.0


def rad2deg(r):
    return r * 180.0 / math.pi


def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """
    Approximate conversion: WGS84 -> local tangent plane in meters.
    Good enough for small fields.
    """
    R = 6378137.0  # Earth radius [m]
    d_lat = deg2rad(lat - ref_lat)
    d_lon = deg2rad(lon - ref_lon)
    ref_lat_rad = deg2rad(ref_lat)

    x = R * d_lon * math.cos(ref_lat_rad)
    y = R * d_lat
    return x, y


def xy_to_latlon(x, y, ref_lat, ref_lon):
    R = 6378137.0
    ref_lat_rad = deg2rad(ref_lat)

    d_lat = y / R
    d_lon = x / (R * math.cos(ref_lat_rad))

    lat = ref_lat + rad2deg(d_lat)
    lon = ref_lon + rad2deg(d_lon)
    return lat, lon


def generate_serpentine_waypoints_from_plan(plan):
    """
    Use the existing mission.plan only to infer:
    - home position
    - bounding box of existing points (coverage rectangle)

    Then generate a fresh serpentine path over that rectangle.
    """
    mission = plan["mission"]
    items = mission["items"]

    # Home position (fallback to first item if not present)
    if "plannedHomePosition" in mission and mission["plannedHomePosition"]:
        home_lat, home_lon, _ = mission["plannedHomePosition"]
    else:
        first = items[0]
        home_lat = first["params"][4]
        home_lon = first["params"][5]

    # Collect all lat/lon from old mission to form bounding box
    lats = []
    lons = []
    for item in items:
        lats.append(item["params"][4])
        lons.append(item["params"][5])

    lat_min = min(lats)
    lat_max = max(lats)
    lon_min = min(lons)
    lon_max = max(lons)

    # Convert bbox corners and home to local XY
    ref_lat, ref_lon = home_lat, home_lon

    # Four corners of rectangle
    corners_latlon = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_max),
        (lat_max, lon_min),
    ]
    corners_xy = [latlon_to_xy(lat, lon, ref_lat, ref_lon)
                  for lat, lon in corners_latlon]

    xs = [p[0] for p in corners_xy]
    ys = [p[1] for p in corners_xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width_x = max_x - min_x
    height_y = max_y - min_y

    # Decide sweep orientation: along the longer dimension
    sweep_along_y = width_x >= height_y  # if True: passes are vertical (x constant)

    waypoints_xy = []

    if sweep_along_y:
        # Vertical lines x = const, moving along y
        num_passes = max(1, math.ceil(width_x / TRACK_SPACING_M))
        # Ensure last pass reaches max_x
        spacing = width_x / num_passes

        for i in range(num_passes + 1):
            x = min_x + i * spacing
            if x > max_x:
                x = max_x
            if i % 2 == 0:
                # bottom -> top
                waypoints_xy.append((x, min_y))
                waypoints_xy.append((x, max_y))
            else:
                # top -> bottom
                waypoints_xy.append((x, max_y))
                waypoints_xy.append((x, min_y))
    else:
        # Horizontal lines y = const, moving along x
        num_passes = max(1, math.ceil(height_y / TRACK_SPACING_M))
        spacing = height_y / num_passes

        for i in range(num_passes + 1):
            y = min_y + i * spacing
            if y > max_y:
                y = max_y
            if i % 2 == 0:
                # left -> right
                waypoints_xy.append((min_x, y))
                waypoints_xy.append((max_x, y))
            else:
                # right -> left
                waypoints_xy.append((max_x, y))
                waypoints_xy.append((min_x, y))

    # Make start closer to home
    home_x, home_y = latlon_to_xy(home_lat, home_lon, ref_lat, ref_lon)
    first_dist = math.hypot(waypoints_xy[0][0] - home_x, waypoints_xy[0][1] - home_y)
    last_dist = math.hypot(waypoints_xy[-1][0] - home_x, waypoints_xy[-1][1] - home_y)
    if last_dist < first_dist:
        waypoints_xy.reverse()

    # Convert back to lat/lon
    waypoints_latlon = [
        (*xy_to_latlon(x, y, ref_lat, ref_lon), ALTITUDE_M)
        for (x, y) in waypoints_xy
    ]

    # Prepend home as first waypoint (optional, but usually nice)
    waypoints_latlon.insert(0, (home_lat, home_lon, ALTITUDE_M))

    return waypoints_latlon


def build_mission_items(waypoints_latlon):
    mission_items = []
    for (lat, lon, alt) in waypoints_latlon:
        mission_items.append(
            MissionItem(
                latitude_deg=lat,
                longitude_deg=lon,
                relative_altitude_m=alt,
                speed_m_s=CRUISE_SPEED_M_S,
                is_fly_through=True,
                gimbal_pitch_deg=0.0,
                gimbal_yaw_deg=0.0,
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=0,
                camera_photo_interval_s=0.0,
                acceptance_radius_m=ACCEPTANCE_RADIUS_M,
                yaw_deg=float("nan"),
                camera_photo_distance_m=0.0,
                vehicle_action=MissionItem.VehicleAction.NONE,
            )
        )
    return mission_items


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone")
            break

    # Load mission.plan (only as input to our generator)
    with open("/home/sherlock/Downloads/Telegram Desktop/mission.plan", "r") as f:
        plan = json.load(f)

    waypoints_latlon = generate_serpentine_waypoints_from_plan(plan)
    mission_items = build_mission_items(waypoints_latlon)

    print(f"Generated {len(mission_items)} mission items (algorithmic serpentine)")

    await drone.mission.set_return_to_launch_after_mission(True)
    mission_plan = MissionPlan(mission_items)
    await drone.mission.upload_mission(mission_plan)

    print("Mission uploaded")

    # Arm and start
    print("-- Arming")
    await drone.action.arm()
    print("-- Starting mission")
    await drone.mission.start_mission()

    # Monitor progress
    async for progress in drone.mission.mission_progress():
        print(f"Mission progress: {progress.current}/{progress.total}")
        if progress.current == progress.total:
            print("Mission complete")
            break

    # Wait until landed
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("Landed and disarmed")
            break


if __name__ == "__main__":
    asyncio.run(run())
