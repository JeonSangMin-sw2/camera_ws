import numpy as np
import sys
import os
import rby1_sdk as rby

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_210630.npz"
data = np.load(dataset_path)

q_arm_list = data["q_arm"]
T_meas_list = data["marker"]
q_head_list = np.zeros((q_arm_list.shape[0], 2)) # mock head since fixed camera

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    model = robot.model()
    arm_idx = np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]])
    ee_links = {
        "right": "ee_right",
        "left": "ee_left"
    }
    
    # Corrected bracket values
    ee_to_marker = {
        "left": [0.0, 0.05445, -0.05002, 90.12, 0.10, 0.00],
        "right": [0.0, -0.05445, -0.05002, 90.10, 0.10, 180.00]
    }

    # Locked camera translation (lambda_cam_pos = 1e6)
    # Stage 1: Global Rough Initialization
    optimizer_st1 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
        head_base_to_cam_nom=[0.102, 0.009, 0.044, -90.0, 0.0, -90.0],
        ee_to_marker_nom=ee_to_marker,
        head_idx=[0, 1],
        eps=1e-6,
        lambda_cam_pos=1e6, # LOCK translation
        lambda_cam_rot=1.0,
        use_sag=False,
        optimize_head=False,
        optimize_camera=True,
        active_arms=["left", "right"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=False
    )
    q_arm_offset, q_head_offset, xi_cam, _, _ = optimizer_st1.optimize(
        q_arm_list, q_head_list, T_meas_list
    )
    
    # Stage 2: Joint Priority Refinement
    optimizer_st2 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
        head_base_to_cam_nom=[0.102, 0.009, 0.044, -90.0, 0.0, -90.0],
        ee_to_marker_nom=ee_to_marker,
        head_idx=[0, 1],
        eps=1e-6,
        lambda_cam_pos=1e6, # LOCK translation
        lambda_cam_rot=1.0,
        use_sag=False,
        optimize_head=False,
        optimize_camera=False,
        active_arms=["left", "right"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=False
    )
    q_arm_offset, q_head_offset, _, _, _ = optimizer_st2.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_offset,
        q_head_offset_init=q_head_offset,
        xi_mount_cam_init=xi_cam
    )
    
    # Stage 3: Final Integration (All Free)
    optimizer_st3 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
        head_base_to_cam_nom=[0.102, 0.009, 0.044, -90.0, 0.0, -90.0],
        ee_to_marker_nom=ee_to_marker,
        head_idx=[0, 1],
        eps=1e-7,
        lambda_cam_pos=1e6, # LOCK translation
        lambda_cam_rot=1.0,
        use_sag=False,
        optimize_head=False,
        optimize_camera=True,
        active_arms=["left", "right"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=False
    )
    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer_st3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_offset,
        q_head_offset_init=q_head_offset,
        xi_mount_cam_init=xi_cam
    )
    
    q_deg = np.degrees(q_arm_offset)
    print("\n--- RESULTS WITH LOCKED CAMERA TRANSLATION ---")
    print(f"Right Joint Offsets (deg): {q_deg[0:7]}")
    print(f"Left Joint Offsets (deg):  {q_deg[7:14]}")
    print(f"Mount to Cam New:          {mount_to_cam_new}")
    print(f"Head Base to Cam New:      {head_base_to_cam_new}")
    
finally:
    robot.disconnect()
