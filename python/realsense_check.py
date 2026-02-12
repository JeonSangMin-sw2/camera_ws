import pyrealsense2 as rs

def list_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices connected")
        return

    for dev in devices:
        print(f"Device: {dev.get_info(rs.camera_info.name)}")
        sensors = dev.query_sensors()
        for sensor in sensors:
            print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
            print("  Profiles:")
            for profile in sensor.get_stream_profiles():
                stream_type = profile.stream_type()
                if stream_type == rs.stream.infrared:
                    sp = profile.as_video_stream_profile()
                    print(f"    Stream: {stream_type}, Index: {profile.stream_index()}, Resolution: {sp.width()}x{sp.height()}, FPS: {sp.fps()}, Format: {profile.format()}")

if __name__ == "__main__":
    list_profiles()
