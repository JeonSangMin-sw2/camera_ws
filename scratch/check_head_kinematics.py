import sys
import os
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform

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
    
    # Corrected bracket values
    ee_to_marker = {
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0],
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0]
    }
    T_ee_to_marker_l = make_transform(ee_to_marker["left"])
    T_ee_to_marker_r = make_transform(ee_to_marker["right"])
    
    # Calibrated offsets from JSON
    jo_r = [0.6128, 4.8188, 0.4543, 2.7765, -0.4436, 0.9783, 0.2149]
    jo_l = [-0.6821, -4.3301, 0.7079, 2.2363, 0.4429, 0.0500, 0.0500]
    
    head_base_to_cam_new = [0.10205, 0.01006, 0.04128, -86.747, 0.233, -89.701]
    T_head_to_cam_new = make_transform(head_base_to_cam_new)
    T_cam_to_head = np.linalg.inv(T_head_to_cam_new)
    
    print("\n--- RESIDUALS AT OPTIMIZED SOLUTION ---")
    pos_errs_r = []
    pos_errs_l = []
    
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
        
        T_head_0_to_ee_r = dyn_model.compute_transformation(state, 2, 0)
        T_head_0_to_ee_l = dyn_model.compute_transformation(state, 2, 1)
        
        T_model_r = T_cam_to_head @ T_head_0_to_ee_r @ T_ee_to_marker_r
        T_model_l = T_cam_to_head @ T_head_0_to_ee_l @ T_ee_to_marker_l
        
        T_meas_r = T_meas_list[i, 0]
        T_meas_l = T_meas_list[i, 1]
        
        pos_errs_r.append(np.linalg.norm(T_meas_r[:3, 3] - T_model_r[:3, 3]) * 1000.0)
        pos_errs_l.append(np.linalg.norm(T_meas_l[:3, 3] - T_model_l[:3, 3]) * 1000.0)
        
    print(f"Mean Right Pos Error: {np.mean(pos_errs_r):.4f} mm")
    print(f"Mean Left Pos Error:  {np.mean(pos_errs_l):.4f} mm")
    
finally:
    robot.disconnect()
