import sys
import rby1_sdk as rby

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

model = robot.model()
print("Robot Joint Names:")
for i, name in enumerate(model.robot_joint_names):
    print(f"  {i}: {name}")

print("\nRight Arm Indices:", list(model.right_arm_idx))
print("Left Arm Indices:", list(model.left_arm_idx))
if hasattr(model, 'head_idx'):
    print("Head Indices:", list(model.head_idx))

robot.disconnect()
