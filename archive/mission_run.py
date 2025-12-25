import asyncio
import json
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone")
            break

    # Load mission.plan (QGC JSON)
    with open("/home/sherlock/Downloads/Telegram Desktop/mission.plan", "r") as f:
        plan = json.load(f)

    mission_items = []
    for item in plan["mission"]["items"]:
        lat = item["params"][4]
        lon = item["params"][5]
        alt = item["params"][6]
        mission_items.append(
    MissionItem(
        latitude_deg=lat,
        longitude_deg=lon,
        relative_altitude_m=alt,
        speed_m_s=1000.0,   
        is_fly_through=True,
        gimbal_pitch_deg=0.0,
        gimbal_yaw_deg=0.0,
        camera_action=MissionItem.CameraAction.NONE,
        loiter_time_s=0,
        camera_photo_interval_s=0.0,
        acceptance_radius_m=1.0,
        yaw_deg=float("nan"),
        camera_photo_distance_m=0.0,
        vehicle_action=MissionItem.VehicleAction.NONE
    )
)


    print(f"Loaded {len(mission_items)} mission items")

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


