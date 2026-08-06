import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_log

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    
    active_joints = [
        'wheel_fr', 'wheel_fl', 'wheel_rr', 'wheel_rl',
        'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
        'head_0', 'head_1'
    ]
    
    class MockRobot:
        def __init__(self, dyn_robot, active_joints):
            self._dyn_robot = dyn_robot
            self._active_joints = active_joints
            
        def get_dynamics(self):
            return self._dyn_robot
            
        class MockModel:
            def __init__(self, active_joints):
                self.robot_joint_names = active_joints
        
            @property
            def right_arm_idx(self):
                return [10, 11, 12, 13, 14, 15, 16]
                
            @property
            def left_arm_idx(self):
                return [17, 18, 19, 20, 21, 22, 23]
                
        def model(self):
            return self.MockModel(self._active_joints)
            
        class MockState:
            def __init__(self):
                self.position = np.zeros(26)
                
        def get_state(self):
            return self.MockState()
            
    mock_robot = MockRobot(dyn_robot, active_joints)
    
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    T_meas = data["marker"]
    
    active_arms = ["right", "left"]
    right_arm_idx = [10, 11, 12, 13, 14, 15, 16]
    left_arm_idx = [17, 18, 19, 20, 21, 22, 23]
    arm_idx = np.concatenate([right_arm_idx, left_arm_idx])
    
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    
    ee_to_marker_nom = {
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0],
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0]
    }
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # We define a function to compute the average residuals
    def eval_errors(T_cam, q_offsets):
        errs_trans = []
        errs_rot = []
        for i in range(q_arm.shape[0]):
            q_full = np.zeros(dyn_robot.get_dof())
            q_full[right_arm_idx] = q_arm[i, 0:7]
            q_full[left_arm_idx] = q_arm[i, 7:14]
            
            if q_offsets is not None:
                q_full[right_arm_idx] += q_offsets[:7]
                q_full[left_arm_idx] += q_offsets[7:]
                
            # FK Right
            state_r = dyn_robot.make_state(["link_head_0", "ee_right"], active_joints)
            state_r.set_q(q_full)
            dyn_robot.compute_forward_kinematics(state_r)
            T_fk_r = dyn_robot.compute_transformation(state_r, 0, 1)
            T_pred_r = np.linalg.inv(T_cam) @ T_fk_r @ make_transform(ee_to_marker_nom["right"])
            T_meas_r = T_meas[i, 0]
            
            # FK Left
            state_l = dyn_robot.make_state(["link_head_0", "ee_left"], active_joints)
            state_l.set_q(q_full)
            dyn_robot.compute_forward_kinematics(state_l)
            T_fk_l = dyn_robot.compute_transformation(state_l, 0, 1)
            T_pred_l = np.linalg.inv(T_cam) @ T_fk_l @ make_transform(ee_to_marker_nom["left"])
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

    # CASE 1
    T_cam_case1 = make_transform([0.0926909783778458, 0.010491669348928388, 0.034263496840896204, -83.952021343563, 0.3734713136903065, -89.65373456252455])
    t_err, r_err = eval_errors(T_cam_case1, None)
    print(f"CASE 1 (Only Camera): Trans Err = {t_err:.2f} mm, Rot Err = {r_err:.2f} deg")
    
    # CASE 2
    T_cam_case2 = make_transform([0.1070456904010868, 0.006221465779563716, 0.03623440373999896, -92.72521714918152, -0.03999666555994106, -90.07437159285783])
    q_offsets_case2 = np.radians([5.12282606, 4.75369556, 1.40841753, 1.99999972, -1.08188965, 1.24917585, -1.19440276, 2.45238687, -5.47109273, -1.81854809, 1.99999973, 0.91547352, 1.20971821, 0.9713167])
    t_err, r_err = eval_errors(T_cam_case2, q_offsets_case2)
    print(f"CASE 2 (Camera + Joints): Trans Err = {t_err:.2f} mm, Rot Err = {r_err:.2f} deg")

if __name__ == "__main__":
    main()
