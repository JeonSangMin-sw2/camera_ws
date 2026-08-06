import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    
    # Active joint names in URDF order
    active_joints = [
        'wheel_fr', 'wheel_fl', 'wheel_rr', 'wheel_rl',
        'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
        'head_0', 'head_1'
    ]
    
    print("Creating state with active joints list...")
    state = dyn_robot.make_state(["link_head_0", "ee_right"], active_joints)
    
    print("State joint names:")
    for idx, name in enumerate(state.get_joint_names()):
        print(f"Index {idx}: {name}")

if __name__ == "__main__":
    main()
