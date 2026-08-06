import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import make_transform, compute_fk, se3_log

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    joint_names = dyn_robot.get_joint_names()
    
    right_arm_idx = [10, 11, 12, 13, 14, 15, 16]
    
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    T_meas = data["marker"]
    
    T_ee_to_marker_right = make_transform([0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0])
    T_cam_nom = make_transform([0.102, 0.009, 0.044, -90.0, 0.0, -90.0])
    
    q_base = q_arm[0, 0:7].copy()
    T_meas_r = T_meas[0, 0]
    
    print("Target Marker Translation in Cam Frame:", T_meas_r[:3, 3] * 1000.0)
    
    for sign0 in [1, -1]:
        for sign1 in [1, -1]:
            for sign3 in [1, -1]:
                q = q_base.copy()
                q[0] *= sign0
                q[1] *= sign1
                q[3] *= sign3
                
                q_full = np.zeros(dyn_robot.get_dof())
                q_full[right_arm_idx] = q
                
                state = dyn_robot.make_state(["link_head_0", "ee_right"], joint_names)
                state.set_q(q_full)
                dyn_robot.compute_forward_kinematics(state)
                T_fk = dyn_robot.compute_transformation(state, 0, 1)
                
                T_pred = np.linalg.inv(T_cam_nom) @ T_fk @ T_ee_to_marker_right
                pos = T_pred[:3, 3] * 1000.0
                dist = np.linalg.norm(pos - T_meas_r[:3, 3]*1000.0)
                print(f"Signs: J0={sign0}, J1={sign1}, J3={sign3} -> Pred: {pos} mm, Dist = {dist:.1f} mm")

if __name__ == "__main__":
    main()
