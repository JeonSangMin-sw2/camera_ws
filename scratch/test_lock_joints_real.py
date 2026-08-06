import sys
import os
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
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
    
    # Real robot URDF parameters
    L_5_ee = 154.80  # mm
    z_sign = -1.0
    
    # Nominals
    T_head_to_cam_nom = np.eye(4)
    T_head_to_cam_nom[:3, :3] = R_scipy.from_euler('zyx', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    T_head_to_cam_nom[:3, 3] = [0.102, 0.009, 0.044]
    
    T_ee_to_marker_r = np.eye(4)
    T_ee_to_marker_r[:3, :3] = R_scipy.from_euler('zyx', [180.0, 0.0, 90.0], degrees=True).as_matrix()
    T_ee_to_marker_r[:3, 3] = [0.0, -0.05416, -0.00237]
    
    T_ee_to_marker_l = np.eye(4)
    T_ee_to_marker_l[:3, :3] = R_scipy.from_euler('zyx', [0.0, 0.0, 90.0], degrees=True).as_matrix()
    T_ee_to_marker_l[:3, 3] = [0.0, 0.05413, -0.00273]
    
    # Precompute FK up to Link 5 frame in simulator
    T_head_0_to_link5_r_list = []
    T_head_0_to_link5_l_list = []
    
    for i in range(len(q_arm_list)):
        q_arm = q_arm_list[i]
        
        # We compute FK up to link_right_arm_5 and link_left_arm_5
        state = dyn_model.make_state(
            ["link_right_arm_5", "link_left_arm_5", "link_head_0"],
            names
        )
        q_full = robot.get_state().position.copy()
        
        # We need to evaluate FK. We will set the arm joints (excluding offsets for now)
        q_full[right_arm_idx_list] = q_arm[:7]
        q_full[left_arm_idx_list] = q_arm[7:14]
        state.set_q(q_full)
        dyn_model.compute_forward_kinematics(state)
        
        T_head_0_to_link5_r_list.append(dyn_model.compute_transformation(state, 2, 0))
        T_head_0_to_link5_l_list.append(dyn_model.compute_transformation(state, 2, 1))

    # Helper to compute FK with joint offsets
    def compute_fk_with_offsets(q_arm_offset):
        T_ee_r_list = []
        T_ee_l_list = []
        
        for i in range(len(q_arm_list)):
            q_arm = q_arm_list[i]
            state = dyn_model.make_state(
                ["link_right_arm_5", "link_left_arm_5", "link_head_0"],
                names
            )
            q_full = robot.get_state().position.copy()
            q_full[right_arm_idx_list] = q_arm[:7] + q_arm_offset[:7]
            q_full[left_arm_idx_list] = q_arm[7:14] + q_arm_offset[7:14]
            state.set_q(q_full)
            dyn_model.compute_forward_kinematics(state)
            
            T_link5_r = dyn_model.compute_transformation(state, 2, 0)
            T_link5_l = dyn_model.compute_transformation(state, 2, 1)
            
            # Manually add the L_5_ee = 154.80 mm translation along Link 5's Z-axis (which is negative)
            # T_ee = T_link5 @ T_link5_to_ee
            # T_link5_to_ee translation is [0, 0, -0.1548] m.
            T_link5_to_ee_r = np.eye(4)
            T_link5_to_ee_r[2, 3] = -L_5_ee / 1000.0
            
            T_link5_to_ee_l = np.eye(4)
            T_link5_to_ee_l[2, 3] = -L_5_ee / 1000.0
            
            T_ee_r_list.append(T_link5_r @ T_link5_to_ee_r)
            T_ee_l_list.append(T_link5_l @ T_link5_to_ee_l)
            
        return T_ee_r_list, T_ee_l_list

    # --- CASE A: Lock camera at nominal, optimize joint offsets ---
    print("\nRunning Case A: Locking Camera at Nominal...")
    # Parameters to optimize: 14 joint offsets (in radians)
    # J3, J5, J6 are anchored to their Step 1 values:
    # Right: J3=-2.7265, J5=-0.9283, J6=-0.2649
    # Left: J3=-2.1863, J5=0.0, J6=0.0
    jo_nom = np.zeros(14)
    jo_nom[3] = np.radians(-2.7265)
    jo_nom[5] = np.radians(-0.9283)
    jo_nom[6] = np.radians(-0.2649)
    jo_nom[10] = np.radians(-2.1863)
    jo_nom[12] = np.radians(0.0)
    jo_nom[13] = np.radians(0.0)
    
    def residuals_case_a(jo):
        T_ee_r_list, T_ee_l_list = compute_fk_with_offsets(jo)
        T_cam_to_head = np.linalg.inv(T_head_to_cam_nom)
        
        res = []
        for i in range(len(q_arm_list)):
            T_meas_r = T_meas_list[i, 0]
            T_meas_l = T_meas_list[i, 1]
            
            T_model_r = T_cam_to_head @ T_ee_r_list[i] @ T_ee_to_marker_r
            T_model_l = T_cam_to_head @ T_ee_l_list[i] @ T_ee_to_marker_l
            
            res.extend(T_meas_r[:3, 3] - T_model_r[:3, 3])
            res.extend(T_meas_l[:3, 3] - T_model_l[:3, 3])
            
            R_err_r = T_model_r[:3, :3].T @ T_meas_r[:3, :3]
            R_err_l = T_model_l[:3, :3].T @ T_meas_l[:3, :3]
            res.extend(R_scipy.from_matrix(R_err_r).as_rotvec())
            res.extend(R_scipy.from_matrix(R_err_l).as_rotvec())
            
        # Add J3, J5, J6 anchors
        anchor_w = 5000.0
        for idx in [3, 5, 6, 10, 12, 13]:
            res.append(anchor_w * (jo[idx] - jo_nom[idx]))
            
        return res

    opt_a = least_squares(residuals_case_a, jo_nom, loss='huber')
    jo_deg = np.degrees(opt_a.x)
    print("Optimized Joint Offsets (deg):")
    print(f"  Right: {jo_deg[0:7]}")
    print(f"  Left:  {jo_deg[7:14]}")
    
    # Calculate final residuals
    final_res = residuals_case_a(opt_a.x)
    final_res_reshaped = np.array(final_res[:len(q_arm_list)*12]).reshape(-1, 12)
    pos_errs_r = np.linalg.norm(final_res_reshaped[:, 0:3], axis=1) * 1000.0
    pos_errs_l = np.linalg.norm(final_res_reshaped[:, 3:6], axis=1) * 1000.0
    print(f"Mean Right Pos Error: {np.mean(pos_errs_r):.4f} mm")
    print(f"Mean Left Pos Error:  {np.mean(pos_errs_l):.4f} mm")

finally:
    robot.disconnect()
