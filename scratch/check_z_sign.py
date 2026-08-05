import sys
import numpy as np
import rby1_sdk as rby

# Copy get_z_sign logic exactly
def get_z_sign(robot, arm_side):
    try:
        dyn_model = robot.get_dynamics()
        names = robot.model().robot_joint_names
        state = dyn_model.make_state(
            [f"link_{arm_side}_arm_5", f"ee_{arm_side}"],
            names
        )
        state.set_q(robot.get_state().position)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        print(f"{arm_side} T[:3, 3] = {T[:3, 3]}")
        return -1.0 if T[2, 3] < 0.0 else 1.0
    except Exception as e:
        print(f"Error in get_z_sign for {arm_side}: {e}")
        return 1.0

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    print("Right arm z_sign:", get_z_sign(robot, "right"))
    print("Left arm z_sign:", get_z_sign(robot, "left"))
finally:
    robot.disconnect()
