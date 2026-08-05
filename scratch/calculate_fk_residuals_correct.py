import numpy as np
import json
import os
import sys
import rby1_sdk as rby
import yaml

# Add core path to import configs/kinematics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_194834.npz"

# Let's use the actual calibrated bracket values from the Step 1 sequential calibration report:
# --- RIGHT ARM ---
# Bracket Pos: X: +0.0, Y: -0.0, Z: +45.7 mm (Z = -48 + 45.7 = -2.3 mm)
# Bracket Rot: R: -1.83, P: -0.08, Y: 180.00 deg
# --- LEFT ARM ---
# Bracket Pos: X: +0.0, Y: +0.1, Z: +45.3 mm (Z = -48 + 45.3 = -2.7 mm)
# Bracket Rot: R: +2.48, P: -0.29, Y: 0.00 deg

Tf_r = [0.0/1000.0, -54.0/1000.0, -2.3/1000.0, 90.0 - 1.83, -0.08, 180.0]
Tf_l = [0.0/1000.0, 54.1/1000.0, -2.7/1000.0, 90.0 + 2.48, -0.29, 0.0]

# Nominal head base to camera (we can try nominal first)
head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]

T_ee_to_marker_r = make_transform(Tf_r)
T_ee_to_marker_l = make_transform(Tf_l)
T_head_base_to_cam = make_transform(head_base_to_cam_nom)

# Connect to robot to get kinematics
ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    dyn = robot.get_dynamics()
    model = robot.model()
    
    # Load dataset
    data = np.load(dataset_path)
    q_arm_list = data["q_arm"]
    q_head_list = data["q_head"] if "q_head" in data else None
    T_meas_list = data["marker"]
    
    num_samples = q_arm_list.shape[0]
    
    # Setup state for FK
    state = dyn.make_state(["link_head_0", "ee_right", "ee_left"], model.robot_joint_names)
    nominal_q = np.array(robot.get_state().position)
    
    print("\n--- FK Residual Analysis using Step 1 Calibrated Brackets (mm) ---")
    print(f"{'Sample':<6} | {'Nom. Dist':<10} | {'Meas. Dist':<10} | {'Dist Diff':<10} | {'R_pos Err':<10} | {'L_pos Err':<10}")
    print("-" * 75)
    
    pos_errs_r = []
    pos_errs_l = []
    
    for i in range(num_samples):
        q_arm = q_arm_list[i]
        T_meas_r = T_meas_list[i, 0]
        T_meas_l = T_meas_list[i, 1]
        
        # Build correct 26-joint vector
        q_full = nominal_q.copy()
        q_full[model.right_arm_idx] = q_arm[0:7]
        q_full[model.left_arm_idx] = q_arm[7:14]
        
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        
        T_hb_to_ee_r = dyn.compute_transformation(state, 0, 1)
        T_hb_to_ee_l = dyn.compute_transformation(state, 0, 2)
        
        T_hb_to_marker_r_model = T_hb_to_ee_r @ T_ee_to_marker_r
        T_hb_to_marker_l_model = T_hb_to_ee_l @ T_ee_to_marker_l
        
        T_cam_to_marker_r_model = np.linalg.inv(T_head_base_to_cam) @ T_hb_to_marker_r_model
        T_cam_to_marker_l_model = np.linalg.inv(T_head_base_to_cam) @ T_hb_to_marker_l_model
        
        pos_r_model = T_cam_to_marker_r_model[:3, 3] * 1000.0
        pos_l_model = T_cam_to_marker_l_model[:3, 3] * 1000.0
        
        pos_r_meas = T_meas_r[:3, 3] * 1000.0
        pos_l_meas = T_meas_l[:3, 3] * 1000.0
        
        dist_model = np.linalg.norm(pos_r_model - pos_l_model)
        dist_meas = np.linalg.norm(pos_r_meas - pos_l_meas)
        dist_diff = dist_meas - dist_model
        
        err_r = np.linalg.norm(pos_r_meas - pos_r_model)
        err_l = np.linalg.norm(pos_l_meas - pos_l_model)
        
        pos_errs_r.append(err_r)
        pos_errs_l.append(err_l)
        
        if i < 15 or i >= num_samples - 5:
            print(f"{i+1:<6} | {dist_model:8.2f} | {dist_meas:8.2f} | {dist_diff:+8.2f} | {err_r:8.2f} | {err_l:8.2f}")
        elif i == 15:
            print("...")
            
    pos_errs_r = np.array(pos_errs_r)
    pos_errs_l = np.array(pos_errs_l)
    
    print("\n--- Position Error Statistics (Calibrated Brackets) ---")
    print(f"Right Arm Model Position Error: Mean={np.mean(pos_errs_r):.2f}mm, Std={np.std(pos_errs_r):.2f}mm, Max={np.max(pos_errs_r):.2f}mm")
    print(f"Left Arm Model Position Error:  Mean={np.mean(pos_errs_l):.2f}mm, Std={np.std(pos_errs_l):.2f}mm, Max={np.max(pos_errs_l):.2f}mm")
    
finally:
    if robot:
        robot.disconnect()
