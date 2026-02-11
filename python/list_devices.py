import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print(f"Total devices: {len(devices)}")
for i, dev in enumerate(devices):
    print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
