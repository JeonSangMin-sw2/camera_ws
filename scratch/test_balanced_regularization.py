import sys
import os
import numpy as np
import rby1_sdk as rby

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
data = np.load(dataset_path)
q_arm_list = data["q_arm"]
T_meas_list = data["marker"]
q_head_list = np.zeros((q_arm_list.shape[0], 2))

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    model = robot.model()
    arm_idx = np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]])
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # Bracket values from Step 1
    ee_to_marker = {
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0],
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0]
    }
    
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    
    joint_offsets = {
        "right": {"joint3": -2.7265, "joint5": -0.9283, "joint6": -0.2649},
        "left": {"joint3": -2.1863, "joint5": 0.0, "joint6": 0.0}
    }

    # Test different lambda weights
    for l_cam in [0.001, 0.01, 0.05, 0.1]:
        print(f"\n================ [l_cam_pos/rot = {l_cam}] ================")
        optimizer = QPCalibrationOptimizer(
            robot=robot,
            arm_idx=arm_idx,
            ee_links=ee_links,
            mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
            head_base_to_cam_nom=head_base_to_cam_nom,
            ee_to_marker_nom=ee_to_marker,
            head_idx=[0, 1],
            eps=1e-7,
            lambda_cam_pos=l_cam,
            lambda_cam_rot=l_cam,
            use_sag=False,
            optimize_head=False,
            optimize_camera=True,
            active_arms=["left", "right"],
            estimate_measurement_noise=True,
            apply_joint_offset_limits=True,
            joint_offsets_to_apply=joint_offsets
        )
        
        # We also need to add a moderate anchor on shoulders!
        # In QPCalibrationOptimizer, joint anchor penalty is lambda_joint_offset.
        # Let's check what lambda_joint_offset is set to! It is 1.0 by default.
        
        q_arm_offset, q_head_offset, xi_cam, _, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list
        )
        
        q_arm_offset, q_head_offset, xi_cam, _, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list,
            q_arm_offset_init=q_arm_offset,
            q_head_offset_init=q_head_offset,
            xi_mount_cam_init=xi_cam
        )
        
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list,
            q_arm_offset_init=q_arm_offset,
            q_head_offset_init=q_head_offset,
            xi_mount_cam_init=xi_cam
        )
        
        q_deg = np.degrees(q_arm_offset)
        print(f"Right Joint Offsets (deg): {q_deg[0:7]}")
        print(f"Left Joint Offsets (deg):  {q_deg[7:14]}")
        print(f"Head Base to Cam New:      {head_base_to_cam_new}")
        
finally:
    robot.disconnect()
