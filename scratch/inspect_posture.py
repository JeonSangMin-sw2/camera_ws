import numpy as np
import sys
import os
import rby1_sdk as rby
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_210630.npz"
data = np.load(dataset_path)

q_arm_list = data["q_arm"]
T_meas_list = data["marker"]

# Connect to robot to get kinematics
ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    dyn_model = robot.get_dynamics()
    names = robot.model().robot_joint_names
    
    # Let's inspect the 1st sample
    q_arm = q_arm_list[0]
    T_meas_r = T_meas_list[0, 0]
    T_meas_l = T_meas_list[0, 1]
    
    # Right arm joints are first 7, left arm are next 7
    model = robot.model()
    right_arm_idx = list(model.right_arm_idx[:7])
    left_arm_idx = list(model.left_arm_idx[:7])
    
    q_full = robot.get_state().position.copy()
    q_full[right_arm_idx] = q_arm[:7]
    q_full[left_arm_idx] = q_arm[7:14]
    
    # Calculate FK
    state = dyn_model.make_state(
        ["ee_right", "ee_left", "link_head_2"],
        names
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    
    T_head_to_right = dyn_model.compute_transformation(state, 2, 0) # head to ee_right
    T_head_to_left = dyn_model.compute_transformation(state, 2, 1)  # head to ee_left
    
    # Calculate EE position relative to head camera mount
    # Head camera nominal mount:
    # head_base_to_cam = [0.102, 0.009, 0.044, -90, 0, -90] (translation in meters, RPY in degrees)
    # The transformation is T_head_to_cam
    from core.calibration_optimizer import make_transform
    T_head_to_cam = make_transform([0.102, 0.009, 0.044, -90.0, 0.0, -90.0])
    
    T_cam_to_right_ee = np.linalg.inv(T_head_to_cam) @ T_head_to_right
    T_cam_to_left_ee = np.linalg.inv(T_head_to_cam) @ T_head_to_left
    
    # Now let's calculate the expected marker pose in camera frame
    # ee_to_marker nominal template:
    # left: [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
    # right: [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
    T_ee_to_marker_l = make_transform([0.0, 0.054, -0.048, 90.0, 0.0, 0.0])
    T_ee_to_marker_r = make_transform([0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
    
    T_cam_to_marker_l_exp = T_cam_to_left_ee @ T_ee_to_marker_l
    T_cam_to_marker_r_exp = T_cam_to_right_ee @ T_ee_to_marker_r
    
    print("\n--- SAMPLE 0 COMPARISON (mm) ---")
    print(f"Right Marker MEASURED: {T_meas_r[:3, 3]*1000.0}")
    print(f"Right Marker EXPECTED: {T_cam_to_marker_r_exp[:3, 3]*1000.0}")
    print(f"Left Marker MEASURED:  {T_meas_l[:3, 3]*1000.0}")
    print(f"Left Marker EXPECTED:  {T_cam_to_marker_l_exp[:3, 3]*1000.0}")
    
    print("\n--- Position Diff (MEASURED - EXPECTED) (mm) ---")
    print(f"Right Diff: {T_meas_r[:3, 3]*1000.0 - T_cam_to_marker_r_exp[:3, 3]*1000.0}")
    print(f"Left Diff:  {T_meas_l[:3, 3]*1000.0 - T_cam_to_marker_l_exp[:3, 3]*1000.0}")
    
finally:
    robot.disconnect()
