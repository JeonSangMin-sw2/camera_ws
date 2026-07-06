import sys
import os
import numpy as np

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk as rby

def test_ready_pose():
    robot = rby.create_robot("127.0.0.1:50051", "m")
    if not robot.connect():
        print("Failed to connect to robot simulator")
        return
        
    print("Connected successfully!")
    
    # Enable control manager
    if not robot.is_power_on(".*"):
        robot.power_on(".*")
    
    # Get state and limits
    dyn = robot.get_dynamics()
    model = robot.model()
    
    # Ready pose for left elbow
    # [-107.0, 12.0, 0.0, 0.0, -73.0, -45.0, 0.0]
    left_arm_deg = [-107.0, 12.0, 0.0, 0.0, -73.0, -45.0, 0.0]
    left_arm = np.radians(left_arm_deg)
    
    # Apply offsets
    l_j6_offset = -3.2122
    l_j5_offset = -6.5287
    l_j3_offset = 0.0 # staged elbow offset is 0
    
    left_arm[6] += np.radians(l_j6_offset)
    left_arm[5] += np.radians(l_j5_offset)
    left_arm[3] += np.radians(l_j3_offset)
    
    print("Target angles in degrees:")
    print([np.degrees(x) for x in left_arm])
    
    # Check limits using SDK dynamics model
    state = dyn.make_state(["ee_left"], model.robot_joint_names)
    q_lower = np.array(dyn.get_limit_q_lower(state))
    q_upper = np.array(dyn.get_limit_q_upper(state))
    
    arm_idx = model.left_arm_idx
    print("\nLeft Arm Limits & Target:")
    for i, joint_idx in enumerate(arm_idx):
        joint_name = model.robot_joint_names[joint_idx]
        val = left_arm[i]
        low = q_lower[joint_idx]
        upp = q_upper[joint_idx]
        in_limit = low <= val <= upp
        print(f"Joint {i} ({joint_name}): limit=[{np.degrees(low):.1f}, {np.degrees(upp):.1f}], target={np.degrees(val):.4f}, in_limit={in_limit}")
        
    # Attempt movej command
    torso = [0.0] * 6
    right_arm = [0.0] * 7
    head = [0.0, 0.0]
    
    comp_cmd = rby.ComponentBasedCommandBuilder()
    body_cmd = rby.BodyComponentBasedCommandBuilder()
    body_cmd.set_torso_command(
        rby.JointPositionCommandBuilder().set_minimum_time(3.0).set_position(torso)
    )
    body_cmd.set_right_arm_command(
        rby.JointPositionCommandBuilder().set_minimum_time(3.0).set_position(right_arm)
    )
    body_cmd.set_left_arm_command(
        rby.JointPositionCommandBuilder().set_minimum_time(5.0).set_position(list(left_arm))
    )
    comp_cmd.set_body_command(body_cmd)
    comp_cmd.set_head_command(
        rby.JointPositionCommandBuilder().set_minimum_time(5.0).set_position(head)
    )
    cmd = rby.RobotCommandBuilder().set_command(comp_cmd)
    
    print("\nSending movej command to elbow ready pose...")
    try:
        rv = robot.send_command(cmd, 10).get()
        print(f"Command finished. FinishCode: {rv.finish_code}")
    except Exception as e:
        print(f"Command execution threw exception: {e}")

if __name__ == "__main__":
    test_ready_pose()
