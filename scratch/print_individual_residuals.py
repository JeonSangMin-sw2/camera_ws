import sys
import os
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import make_transform

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
    dyn_model = robot.get_dynamics()
    names = robot.model().robot_joint_names
    
    # Corrected nominal head_base_to_cam and ee_to_marker
    T_head_to_cam = make_transform([0.047, 0.009, 0.137, -90.0, 0.0, -90.0])
    T_ee_to_marker_l = make_transform([0.0, 0.05445, -0.05002, 90.12, 0.10, 0.00])
    T_ee_to_marker_r = make_transform([0.0, -0.05445, -0.05002, 90.10, 0.10, 180.00])
    
    model = robot.model()
    right_arm_idx = list(model.right_arm_idx[:7])
    left_arm_idx = list(model.left_arm_idx[:7])
    
    print("Sample | Right Pos Err (mm) | Right Rot Err (deg) | Left Pos Err (mm) | Left Rot Err (deg)")
    print("-" * 90)
    
    for i in range(len(q_arm_list)):
        q_arm = q_arm_list[i]
        T_meas_r = T_meas_list[i, 0]
        T_meas_l = T_meas_list[i, 1]
        
        # FK
        state = dyn_model.make_state(
            ["ee_right", "ee_left", "link_head_2"],
            names
        )
        q_full = robot.get_state().position.copy()
        q_full[right_arm_idx] = q_arm[:7]
        q_full[left_arm_idx] = q_arm[7:14]
        state.set_q(q_full)
        dyn_model.compute_forward_kinematics(state)
        
        T_head_to_right = dyn_model.compute_transformation(state, 2, 0)
        T_head_to_left = dyn_model.compute_transformation(state, 2, 1)
        
        T_cam_to_right_ee = np.linalg.inv(T_head_to_cam) @ T_head_to_right
        T_cam_to_left_ee = np.linalg.inv(T_head_to_cam) @ T_head_to_left
        
        T_model_r = T_cam_to_right_ee @ T_ee_to_marker_r
        T_model_l = T_cam_to_left_ee @ T_ee_to_marker_l
        
        # Translation errors
        t_err_r = T_meas_r[:3, 3] - T_model_r[:3, 3]
        t_err_l = T_meas_l[:3, 3] - T_model_l[:3, 3]
        pos_err_r = np.linalg.norm(t_err_r) * 1000.0
        pos_err_l = np.linalg.norm(t_err_l) * 1000.0
        
        # Rotation errors
        R_err_r = np.linalg.inv(T_model_r[:3, :3]) @ T_meas_r[:3, :3]
        R_err_l = np.linalg.inv(T_model_l[:3, :3]) @ T_meas_l[:3, :3]
        rot_err_r = np.degrees(np.arccos(np.clip((np.trace(R_err_r) - 1.0) / 2.0, -1.0, 1.0)))
        rot_err_l = np.degrees(np.arccos(np.clip((np.trace(R_err_l) - 1.0) / 2.0, -1.0, 1.0)))
        
        print(f" {i:2d}    | {pos_err_r:17.2f} | {rot_err_r:18.2f} | {pos_err_l:16.2f} | {rot_err_l:17.2f}")
        
finally:
    robot.disconnect()
