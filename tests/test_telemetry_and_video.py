# -*- coding: utf-8 -*-
"""
SIYI HM30 Video and Telemetry Receiver

This script captures and displays the video stream and telemetry data from a
SIYI HM30 ground unit connected to a computer via Ethernet or USB-C.

It uses multithreading to handle both streams concurrently:
- One thread for the RTSP video stream using OpenCV.
- One thread for the MAVLink telemetry data over UDP using pymavlink.

The script overlays key telemetry data (GPS, Altitude) on the video feed.

Required Libraries:
- opencv-python
- pymavlink

You can install them using pip:
pip install opencv-python pymavlink
"""

import cv2
import threading
import time
from pymavlink import mavutil

# --- Configuration ---
# These are the default network values for the SIYI HM30.
# If you have changed the configuration on your HM30, update these values.

# RTSP URL for the video stream from the air unit
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'

# UDP port where the ground unit sends MAVLink telemetry data
# The script will listen on all network interfaces ('0.0.0.0') for this port.
TELEMETRY_UDP_PORT = 14550


def receive_video_stream(shared_data, lock):
    """
    Connects to the RTSP stream, reads frames, overlays telemetry, and displays them.
    """
    print(f"[Video Thread] Starting video stream capture from: {RTSP_URL}")
    
    # Configure OpenCV to use a larger network buffer
    import os
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

    # Attempt to connect to the video stream
    video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not video_capture.isOpened():
        print("[Video Thread] Error: Could not open video stream. Check the RTSP URL and network connection.")
        return

    print("[Video Thread] Video stream opened successfully.")
    
    while True:
        # Read a frame from the video stream
        ret, frame = video_capture.read()

        if not ret:
            print("[Video Thread] Error: Failed to grab frame. Reconnecting...")
            # Attempt to reconnect if the stream is lost
            video_capture.release()
            time.sleep(2)
            video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not video_capture.isOpened():
                print("[Video Thread] Failed to reconnect.")
                break
            continue
        
        # --- Overlay Telemetry Data ---
        with lock:
            lat = shared_data.get('lat', 'N/A')
            lon = shared_data.get('lon', 'N/A')
            alt = shared_data.get('alt', 'N/A')
            sats = shared_data.get('sats', 'N/A')

        # Format the text strings
        lat_str = f"Lat: {lat:.7f}" if isinstance(lat, float) else f"Lat: {lat}"
        lon_str = f"Lon: {lon:.7f}" if isinstance(lon, float) else f"Lon: {lon}"
        alt_str = f"Alt: {alt:.2f} m" if isinstance(alt, float) else f"Alt: {alt}"
        sats_str = f"Sats: {sats}"

        # Put text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 255, 0) # Bright Green
        thickness = 2
        
        cv2.putText(frame, lat_str, (10, 30), font, font_scale, font_color, thickness)
        cv2.putText(frame, lon_str, (10, 60), font, font_scale, font_color, thickness)
        cv2.putText(frame, alt_str, (10, 90), font, font_scale, font_color, thickness)
        cv2.putText(frame, sats_str, (10, 120), font, font_scale, font_color, thickness)

        # Display the frame in a window
        cv2.imshow('SIYI HM30 Video Stream', frame)

        # Check for 'q' key press to exit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Video Thread] 'q' key pressed. Closing video window.")
            break

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    print("[Video Thread] Video stream stopped.")


def receive_telemetry_data(shared_data, lock):
    """
    Listens for MAVLink telemetry, parses GPS and Alt data, and stores it.
    """
    # The connection string to listen for incoming UDP packets.
    # '0.0.0.0' means listen on all available network interfaces.
    connection_string = f'udp:0.0.0.0:{TELEMETRY_UDP_PORT}'
    print(f"[Telemetry Thread] Listening for MAVLink telemetry on: {connection_string}")

    try:
        # Start a MAVLink connection
        master = mavutil.mavlink_connection(connection_string)
        
        # Wait for the first heartbeat message to confirm connection
        print("[Telemetry Thread] Waiting for heartbeat from vehicle...")
        master.wait_heartbeat()
        print("[Telemetry Thread] Heartbeat received! Telemetry connected.")

        while True:
            # Wait for a new MAVLink message
            msg = master.recv_match(blocking=True)
            if not msg:
                continue

            msg_type = msg.get_type()
            
            # Use a lock to ensure thread-safe updates to the shared dictionary
            with lock:
                if msg_type == 'GPS_RAW_INT':
                    shared_data['lat'] = msg.lat / 1e7
                    shared_data['lon'] = msg.lon / 1e7
                    shared_data['sats'] = msg.satellites_visible
                    print(f"[Telemetry] Updated GPS: Lat={shared_data['lat']}, Lon={shared_data['lon']}, Sats={shared_data['sats']}")
                
                elif msg_type == 'GLOBAL_POSITION_INT':
                    # Altitude is in millimeters, convert to meters
                    shared_data['alt'] = msg.relative_alt / 1000.0
                    print(f"[Telemetry] Updated Altitude: {shared_data['alt']} m")

    except Exception as e:
        print(f"[Telemetry Thread] An error occurred: {e}")
    
    print("[Telemetry Thread] Telemetry listener stopped.")


if __name__ == "__main__":
    print("--- SIYI HM30 Data Receiver ---")
    
    # Shared data structure and a lock for thread-safe access
    shared_telemetry_data = {}
    data_lock = threading.Lock()

    print("Starting video and telemetry threads...")
    print("Press 'q' in the video window or Ctrl+C in the console to exit.")

    # Create thread for video, passing the shared data and lock
    video_thread = threading.Thread(target=receive_video_stream, args=(shared_telemetry_data, data_lock))
    video_thread.daemon = True

    # Create thread for telemetry, passing the shared data and lock
    telemetry_thread = threading.Thread(target=receive_telemetry_data, args=(shared_telemetry_data, data_lock))
    telemetry_thread.daemon = True

    # Start both threads
    video_thread.start()
    telemetry_thread.start()

    # Keep the main thread alive to allow the daemon threads to run
    # and to catch the KeyboardInterrupt (Ctrl+C).
    try:
        while video_thread.is_alive() and telemetry_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main Thread] Ctrl+C detected. Shutting down.")

    print("[Main Thread] Program finished.")

