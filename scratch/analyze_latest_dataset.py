import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import make_transform, se3_log

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
    if not os.path.exists(urdf_path):
        print("URDF not found:", urdf_path)
        return
        
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    
    active_joints = [
        'wheel_fr', 'wheel_fl', 'wheel_rr', 'wheel_rl',
        'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
        'head_0', 'head_1'
    ]
    
    right_arm_idx = [10, 11, 12, 13, 14, 15, 16]
    left_arm_idx = [17, 18, 19, 20, 21, 22, 23]
    
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    T_meas = data["marker"]
    
    T_ee_to_marker_right = make_transform([0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0])
    T_ee_to_marker_left = make_transform([0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0])
    
    T_cam_nom = make_transform([0.102, 0.009, 0.044, -90.0, 0.0, -90.0])
    T_cam_opt = make_transform([0.10205285103156425, 0.010062629132623957, 0.041283808624917664, -86.74678102962154, 0.2326207733234383, -89.7014112806581])
    
    q_offset_right = np.radians([0.6128188845267524, 4.818751109707167, 0.4542828381384349, 2.7764738154536284, -0.44357780923081497, 0.978285103827088, 0.21489365749816672])
    q_offset_left = np.radians([-0.6820776414696303, -4.330110071904625, 0.7078501384860189, 2.2363306757379893, 0.44287246791202134, 0.05000009608108779, 0.05000026088138836])
    
    print("Evaluating residuals (mean absolute translation error in mm, rotation error in deg):")
    
    def eval_errors(T_cam, q_offsets_dict):
        errs_trans = []
        errs_rot = []
        for i in range(q_arm.shape[0]):
            q_full = np.zeros(dyn_robot.get_dof())
            q_full[right_arm_idx] = q_arm[i, 0:7]
            q_full[left_arm_idx] = q_arm[i, 7:14]
            
            if "right" in q_offsets_dict:
                q_full[right_arm_idx] += q_offsets_dict["right"]
            if "left" in q_offsets_dict:
                q_full[left_arm_idx] += q_offsets_dict["left"]
                
            # FK Right
            state_r = dyn_robot.make_state(["link_head_0", "ee_right"], active_joints)
            state_r.set_q(q_full)
            dyn_robot.compute_forward_kinematics(state_r)
            T_fk_r = dyn_robot.compute_transformation(state_r, 0, 1)
            T_pred_r = np.linalg.inv(T_cam) @ T_fk_r @ T_ee_to_marker_right
            T_meas_r = T_meas[i, 0]
            
            # FK Left
            state_l = dyn_robot.make_state(["link_head_0", "ee_left"], active_joints)
            state_l.set_q(q_full)
            dyn_robot.compute_forward_kinematics(state_l)
            T_fk_l = dyn_robot.compute_transformation(state_l, 0, 1)
            T_pred_l = np.linalg.inv(T_cam) @ T_fk_l @ T_ee_to_marker_left
            T_meas_l = T_meas[i, 1]
            
            err_r = np.linalg.inv(T_pred_r) @ T_meas_r
            xi_r = se3_log(err_r)
            errs_trans.append(np.linalg.norm(xi_r[3:]) * 1000.0)
            errs_rot.append(np.degrees(np.linalg.norm(xi_r[:3])))
            
            err_l = np.linalg.inv(T_pred_l) @ T_meas_l
            xi_l = se3_log(err_l)
            errs_trans.append(np.linalg.norm(xi_l[3:]) * 1000.0)
            errs_rot.append(np.degrees(np.linalg.norm(xi_l[:3])))
            
        return np.mean(errs_trans), np.mean(errs_rot)

    t_err, r_err = eval_errors(T_cam_nom, {})
    print(f"1. Nominal (No calibration): Trans Err = {t_err:.2f} mm, Rot Err = {r_err:.2f} deg")
    
    t_err, r_err = eval_errors(T_cam_opt, {})
    print(f"2. Only Camera Optimized   : Trans Err = {t_err:.2f} mm, Rot Err = {r_err:.2f} deg")
    
    t_err, r_err = eval_errors(T_cam_opt, {"right": q_offset_right, "left": q_offset_left})
    print(f"3. Fully Optimized         : Trans Err = {t_err:.2f} mm, Rot Err = {r_err:.2f} deg")

if __name__ == "__main__":
    main()
