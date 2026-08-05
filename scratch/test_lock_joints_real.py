import sys
import os
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_210630.npz"
data = np.load(dataset_path)

q_arm_list = data["q_arm"]
T_meas_list = data["marker"]

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    model = robot.model()
    dyn_model = robot.get_dynamics()
    names = robot.model().robot_joint_names
    right_arm_idx_list = list(model.right_arm_idx[:7])
    left_arm_idx_list = list(model.left_arm_idx[:7])
    
    # Correct NOMINAL head_base_to_cam (which is correct for the robot base link head_0!)
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    T_head_to_cam_nom = make_transform(head_base_to_cam_nom)
    
    # Corrected bracket values (Z = -50.0 mm)
    ee_to_marker = {
        "left": [0.0, 0.05445, -0.05002, 90.12, 0.10, 0.00],
        "right": [0.0, -0.05445, -0.05002, 90.10, 0.10, 180.00]
    }
    T_ee_to_marker_l = make_transform(ee_to_marker["left"])
    T_ee_to_marker_r = make_transform(ee_to_marker["right"])
    
    # Fixed joint offsets
    jo_r = [-0.0, -0.0, -0.0, -1.8946, -0.0, -0.2606, -0.0784]
    jo_l = [-0.0, -0.0, -0.0, -2.2336, -0.0, 0.0, 0.0878]
    
    # Precompute FK (relative to link_head_0!)
    T_head_0_to_ee_r_list = []
    T_head_0_to_ee_l_list = []
    
    for i in range(len(q_arm_list)):
        q_arm = q_arm_list[i]
        
        state = dyn_model.make_state(
            ["ee_right", "ee_left", "link_head_0"],
            names
        )
        q_full = robot.get_state().position.copy()
        q_full[right_arm_idx_list] = q_arm[:7] + np.radians(jo_r)
        q_full[left_arm_idx_list] = q_arm[7:14] + np.radians(jo_l)
        state.set_q(q_full)
        dyn_model.compute_forward_kinematics(state)
        
        T_head_0_to_ee_r_list.append(dyn_model.compute_transformation(state, 2, 0))
        T_head_0_to_ee_l_list.append(dyn_model.compute_transformation(state, 2, 1))
        
    def residuals(xi_cam):
        from core.calibration_optimizer import se3_exp
        T_cam_dev = se3_exp(xi_cam)
        T_head_to_cam = T_head_to_cam_nom @ T_cam_dev
        T_cam_to_head = np.linalg.inv(T_head_to_cam)
        
        res = []
        for i in range(len(q_arm_list)):
            T_meas_r = T_meas_list[i, 0]
            T_meas_l = T_meas_list[i, 1]
            
            T_model_r = T_cam_to_head @ T_head_0_to_ee_r_list[i] @ T_ee_to_marker_r
            T_model_l = T_cam_to_head @ T_head_0_to_ee_l_list[i] @ T_ee_to_marker_l
            
            # Position errors (meters)
            res.extend(T_meas_r[:3, 3] - T_model_r[:3, 3])
            res.extend(T_meas_l[:3, 3] - T_model_l[:3, 3])
            
            # Rotation errors (radians)
            R_err_r = np.linalg.inv(T_model_r[:3, :3]) @ T_meas_r[:3, :3]
            R_err_l = np.linalg.inv(T_model_l[:3, :3]) @ T_meas_l[:3, :3]
            rot_axis_r = R_scipy.from_matrix(R_err_r).as_rotvec()
            rot_axis_l = R_scipy.from_matrix(R_err_l).as_rotvec()
            
            res.extend(rot_axis_r)
            res.extend(rot_axis_l)
            
        return res
        
    initial_guess = np.zeros(6)
    opt_res = least_squares(residuals, initial_guess, loss='huber')
    
    # Print results
    from core.calibration_optimizer import se3_exp
    T_cam_dev = se3_exp(opt_res.x)
    T_head_to_cam_new = T_head_to_cam_nom @ T_cam_dev
    
    t_new = T_head_to_cam_new[:3, 3]
    r_new = R_scipy.from_matrix(T_head_to_cam_new[:3, :3]).as_euler('zyx', degrees=True)
    
    print("\n====================================")
    print("  RESULTS WITH ALL JOINTS LOCKED (REAL HEAD BASE)")
    print("====================================")
    print(f"Optimized Camera Pos (m):        {t_new}")
    print(f"Optimized Camera RPY (deg):      {[r_new[2], r_new[1], r_new[0]]}")
    
    # Calculate final residuals
    final_res = residuals(opt_res.x)
    final_res_reshaped = np.array(final_res).reshape(-1, 12)
    pos_errs_r = np.linalg.norm(final_res_reshaped[:, 0:3], axis=1) * 1000.0
    pos_errs_l = np.linalg.norm(final_res_reshaped[:, 3:6], axis=1) * 1000.0
    rot_errs_r = np.degrees(np.linalg.norm(final_res_reshaped[:, 6:9], axis=1))
    rot_errs_l = np.degrees(np.linalg.norm(final_res_reshaped[:, 9:12], axis=1))
    
    print(f"Mean Right Pos Error:           {np.mean(pos_errs_r):.4f} mm")
    print(f"Max Right Pos Error:            {np.max(pos_errs_r):.4f} mm")
    print(f"Mean Left Pos Error:            {np.mean(pos_errs_l):.4f} mm")
    print(f"Max Left Pos Error:             {np.max(pos_errs_l):.4f} mm")
    print(f"Mean Right Rot Error:           {np.mean(rot_errs_r):.4f} deg")
    print(f"Mean Left Rot Error:            {np.mean(rot_errs_l):.4f} deg")
    
finally:
    robot.disconnect()
