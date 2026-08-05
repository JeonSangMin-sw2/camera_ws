import numpy as np
import json
import os
import sys
import rby1_sdk as rby
import yaml

# Add core path to import configs/kinematics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform, se3_log

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_194834.npz"
setting_path = "/home/rainbow/camera_ws/config/setting.yaml"

if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

with open(setting_path, "r") as f:
    config = yaml.safe_load(f)

# Load marker brackets
Tf_r = config["camera"]["Tf_to_marker_right"] # [X, Y, Z, R, P, Y]
Tf_l = config["camera"]["Tf_to_marker_left"] # [X, Y, Z, R, P, Y]
head_base_to_cam_nom = config["camera"]["head_base_to_cam"]

T_ee_to_marker_r = make_transform(Tf_r)
T_ee_to_marker_l = make_transform(Tf_l)
T_head_base_to_cam = make_transform(head_base_to_cam_nom)

# Connect to robot to get kinematics
ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "a")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

# Query actual model name
robot_info = robot.get_robot_info()
model_name = robot_info.robot_model_name
print(f"Robot model version: {robot_info.robot_model_version}, model name: {model_name}")

robot.disconnect()

# Recreate with correct model
robot = rby.create_robot(ip, model_name.lower())
if not robot.connect():
    print(f"Error: Failed to connect with correct model '{model_name}'")
    sys.exit(1)

try:
    dyn = robot.get_dynamics()
    model = robot.model()
    
    # Load dataset
    data = np.load(dataset_path)
    q_full_list = data["q"]
    T_meas_list = data["marker"]
    
    num_samples = q_full_list.shape[0]
    
    # Setup state for FK
    state = dyn.make_state(["link_head_0", "ee_right", "ee_left"], model.robot_joint_names)
    
    print("\n--- Residual Analysis (mm & deg) ---")
    print(f"{'Sample':<6} | {'Nom. Dist':<10} | {'Meas. Dist':<10} | {'Dist Diff':<10} | {'R_pos Err':<10} | {'L_pos Err':<10}")
    print("-" * 70)
    
    pos_errs_r = []
    pos_errs_l = []
    
    for i in range(num_samples):
        q_full = q_full_list[i]
        T_meas_r = T_meas_list[i, 0]
        T_meas_l = T_meas_list[i, 1]
        
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        
        # FK transforms relative to link_head_0 (head base)
        T_hb_to_ee_r = dyn.compute_transformation(state, 0, 1)
        T_hb_to_ee_l = dyn.compute_transformation(state, 0, 2)
        
        # Model marker poses relative to head base
        T_hb_to_marker_r_model = T_hb_to_ee_r @ T_ee_to_marker_r
        T_hb_to_marker_l_model = T_hb_to_ee_l @ T_ee_to_marker_l
        
        # Modeled marker poses in the nominal camera frame
        T_cam_to_marker_r_model = np.linalg.inv(T_head_base_to_cam) @ T_hb_to_marker_r_model
        T_cam_to_marker_l_model = np.linalg.inv(T_head_base_to_cam) @ T_hb_to_marker_l_model
        
        # Measured vs modeled distance
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
    
    print("\n--- Position Error Statistics (Nominal parameters) ---")
    print(f"Right Arm Model Position Error: Mean={np.mean(pos_errs_r):.2f}mm, Std={np.std(pos_errs_r):.2f}mm, Max={np.max(pos_errs_r):.2f}mm")
    print(f"Left Arm Model Position Error:  Mean={np.mean(pos_errs_l):.2f}mm, Std={np.std(pos_errs_l):.2f}mm, Max={np.max(pos_errs_l):.2f}mm")
    
finally:
    if robot:
        robot.disconnect()
