import sys
import rby1_sdk as rby

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    names = robot.model().robot_link_names
    print("Robot Link Names:")
    for i, name in enumerate(names):
        print(f" {i:2d}: {name}")
finally:
    robot.disconnect()
