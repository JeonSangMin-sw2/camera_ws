import numpy as np
import json
import os
import sys
import rby1_sdk as rby
import yaml
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize

# Add core path to import configs/kinematics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform, se3_log

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_194834.npz"

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
    
    # Compute FK for all samples
    T_hb_to_ee_r_list = []
    T_hb_to_ee_l_list = []
    for i in range(num_samples):
        q_full = nominal_q.copy()
        q_full[model.right_arm_idx] = q_arm_list[i][0:7]
        q_full[model.left_arm_idx] = q_arm_list[i][7:14]
        
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        
        T_hb_to_ee_r_list.append(dyn.compute_transformation(state, 0, 1))
        T_hb_to_ee_l_list.append(dyn.compute_transformation(state, 0, 2))
        
    T_hb_to_ee_r_list = np.array(T_hb_to_ee_r_list)
    T_hb_to_ee_l_list = np.array(T_hb_to_ee_l_list)
    
    # Let's run optimizations under different assumptions:
    # Case 1: Optimize ONLY camera extrinsics (6 parameters), lock brackets to nominal, lock joint offsets to 0
    # Nominal brackets:
    # Right: [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
    # Left:  [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
    T_ee_to_marker_r_nom = make_transform([0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
    T_ee_to_marker_l_nom = make_transform([0.0, 0.054, -0.048, 90.0, 0.0, 0.0])
    
    def loss_camera_only(x):
        # x is 6 parameters of head_base_to_cam: tx, ty, tz, rx, ry, rz (in deg)
        T_cam = make_transform(x)
        T_cam_inv = np.linalg.inv(T_cam)
        
        errs = []
        for i in range(num_samples):
            # Right arm
            T_marker_r_model = T_cam_inv @ T_hb_to_ee_r_list[i] @ T_ee_to_marker_r_nom
            pos_r_model = T_marker_r_model[:3, 3]
            pos_r_meas = T_meas_list[i, 0][:3, 3]
            errs.append(np.linalg.norm(pos_r_model - pos_r_meas))
            
            # Left arm
            T_marker_l_model = T_cam_inv @ T_hb_to_ee_l_list[i] @ T_ee_to_marker_l_nom
            pos_l_model = T_marker_l_model[:3, 3]
            pos_l_meas = T_meas_list[i, 1][:3, 3]
            errs.append(np.linalg.norm(pos_l_model - pos_l_meas))
            
        return np.mean(np.square(errs))
        
    # Initial guess for camera
    x0 = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    res_c = minimize(loss_camera_only, x0, method='BFGS')
    print("--- Case 1: Optimize Camera Pose Only (No joint offsets, nominal brackets) ---")
    print(f"Optimal head_base_to_cam: {res_c.x}")
    print(f"Mean position error: {np.sqrt(res_c.fun)*1000.0:.2f} mm")
    
    # Case 2: Optimize Camera Pose AND Marker Brackets (6 parameters of camera + 2 * 3 position parameters of brackets)
    def loss_camera_and_brackets(x):
        T_cam = make_transform([x[0], x[1], x[2], x[3], x[4], x[5]])
        T_cam_inv = np.linalg.inv(T_cam)
        
        T_ee_r = make_transform([x[6], x[7], x[8], 90.0, 0.0, 180.0])
        T_ee_l = make_transform([x[9], x[10], x[11], 90.0, 0.0, 0.0])
        
        errs = []
        for i in range(num_samples):
            # Right arm
            T_marker_r_model = T_cam_inv @ T_hb_to_ee_r_list[i] @ T_ee_r
            pos_r_model = T_marker_r_model[:3, 3]
            pos_r_meas = T_meas_list[i, 0][:3, 3]
            errs.append(np.linalg.norm(pos_r_model - pos_r_meas))
            
            # Left arm
            T_marker_l_model = T_cam_inv @ T_hb_to_ee_l_list[i] @ T_ee_l
            pos_l_model = T_marker_l_model[:3, 3]
            pos_l_meas = T_meas_list[i, 1][:3, 3]
            errs.append(np.linalg.norm(pos_l_model - pos_l_meas))
            
        return np.mean(np.square(errs))
        
    x0_cb = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0, 0.0, -0.054, -0.048, 0.0, 0.054, -0.048]
    res_cb = minimize(loss_camera_and_brackets, x0_cb, method='BFGS')
    print("\n--- Case 2: Optimize Camera Pose AND Brackets (No joint offsets) ---")
    print(f"Optimal head_base_to_cam: {res_cb.x[0:6]}")
    print(f"Optimal Right bracket: {res_cb.x[6:9]*1000.0} mm")
    print(f"Optimal Left bracket: {res_cb.x[9:12]*1000.0} mm")
    print(f"Mean position error: {np.sqrt(res_cb.fun)*1000.0:.2f} mm")

    # Case 3: Optimize Camera Pose, Brackets AND Joint Offsets (J3, J5, J6 only)
    # Let's see if J3, J5, J6 can be optimized without 어깨 관절 (J0, J1, J2)
    # To do this, we need to apply joint offsets to the FK computation inside the loop
    def loss_refined(x):
        # x: 6 cam, 6 brackets, 6 joint offsets (J3, J5, J6 for R and L)
        T_cam = make_transform(x[0:6])
        T_cam_inv = np.linalg.inv(T_cam)
        
        T_ee_r = make_transform([x[6], x[7], x[8], 90.0, 0.0, 180.0])
        T_ee_l = make_transform([x[9], x[10], x[11], 90.0, 0.0, 0.0])
        
        # Joint offsets: R_j3, R_j5, R_j6, L_j3, L_j5, L_j6
        R_j3, R_j5, R_j6 = np.radians(x[12]), np.radians(x[13]), np.radians(x[14])
        L_j3, L_j5, L_j6 = np.radians(x[15]), np.radians(x[16]), np.radians(x[17])
        
        errs = []
        for i in range(num_samples):
            # Apply offsets to nominal angles
            q_full = nominal_q.copy()
            q_full[model.right_arm_idx] = q_arm_list[i][0:7]
            q_full[model.left_arm_idx] = q_arm_list[i][7:14]
            
            # Apply offsets
            q_full[model.right_arm_idx[3]] += R_j3
            q_full[model.right_arm_idx[5]] += R_j5
            q_full[model.right_arm_idx[6]] += R_j6
            
            q_full[model.left_arm_idx[3]] += L_j3
            q_full[model.left_arm_idx[5]] += L_j5
            q_full[model.left_arm_idx[6]] += L_j6
            
            state.set_q(q_full)
            dyn.compute_forward_kinematics(state)
            
            T_hb_to_ee_r = dyn.compute_transformation(state, 0, 1)
            T_hb_to_ee_l = dyn.compute_transformation(state, 0, 2)
            
            # Right arm
            T_marker_r_model = T_cam_inv @ T_hb_to_ee_r @ T_ee_r
            pos_r_model = T_marker_r_model[:3, 3]
            pos_r_meas = T_meas_list[i, 0][:3, 3]
            errs.append(np.linalg.norm(pos_r_model - pos_r_meas))
            
            # Left arm
            T_marker_l_model = T_cam_inv @ T_hb_to_ee_l @ T_ee_l
            pos_l_model = T_marker_l_model[:3, 3]
            pos_l_meas = T_meas_list[i, 1][:3, 3]
            errs.append(np.linalg.norm(pos_l_model - pos_l_meas))
            
        return np.mean(np.square(errs))
        
    x0_all = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0, 
              0.0, -0.054, -0.048, 0.0, 0.054, -0.048, 
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    res_all = minimize(loss_refined, x0_all, method='BFGS')
    print("\n--- Case 3: Optimize Camera Pose, Brackets AND Joint Offsets (J3, J5, J6 only) ---")
    print(f"Optimal head_base_to_cam: {res_all.x[0:6]}")
    print(f"Optimal Right bracket: {res_all.x[6:9]*1000.0} mm")
    print(f"Optimal Left bracket: {res_all.x[9:12]*1000.0} mm")
    print(f"Optimal Right J3, J5, J6: {res_all.x[12:15]} deg")
    print(f"Optimal Left J3, J5, J6: {res_all.x[15:18]} deg")
    print(f"Mean position error: {np.sqrt(res_all.fun)*1000.0:.2f} mm")
    
finally:
    if robot:
        robot.disconnect()
