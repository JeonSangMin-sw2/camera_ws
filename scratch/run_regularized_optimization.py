import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
import scipy.sparse as spa
import qpsolvers
from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_log, add_weighted_normal_equation

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
    
    # Load dataset
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

    def run_opt_with_reg(reg_weight):
        opt = QPCalibrationOptimizer(
            robot=mock_robot,
            arm_idx=arm_idx,
            ee_links=ee_links,
            mount_to_cam_nom=head_base_to_cam_nom,
            head_base_to_cam_nom=head_base_to_cam_nom,
            ee_to_marker_nom=ee_to_marker_nom,
            head_idx=None,
            camera_link="link_head_0",
            max_iter=500,
            lambda_cam_pos=0.0,
            lambda_cam_rot=0.0,
        )
        opt.optimize_arm = True
        opt.optimize_camera = True
        
        # Override compute_step
        def custom_compute_step(q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
            dim = opt.total_dim()
            H = np.zeros((dim, dim))
            g = np.zeros(dim)
            total_err = 0.0
            weights = opt.noise_estimator.weights()
            residual_samples = []
            
            for q_arm_sample, q_head_sample, T_meas_pair in zip(q_arm_list, [None]*len(q_arm_list), T_meas_list):
                for side_idx, arm_side in enumerate(["right", "left"]):
                    T_meas_sample = T_meas_pair[side_idx]
                    Jb_joint, _, T_ee_to_marker, T_model = opt.evaluate_sample(
                        q_arm_sample, None, arm_side, q_arm_offset, None, xi_mount_cam
                    )
                    T_err = np.linalg.inv(T_model) @ T_meas_sample
                    xi = se3_log(T_err)
                    J = opt.build_jacobian(
                        q_arm_sample, None, arm_side, q_arm_offset, None, xi_mount_cam,
                        Jb_joint, T_ee_to_marker, T_model
                    )
                    add_weighted_normal_equation(H, g, J, xi, weights)
                    total_err += np.linalg.norm(xi)
                    residual_samples.append(xi)
            
            opt.noise_estimator.update(residual_samples)
            
            # Apply joint regularization
            if reg_weight > 0.0:
                for idx in range(len(q_arm_offset)):
                    H[idx, idx] += reg_weight
                    g[idx] += -reg_weight * q_arm_offset[idx]
                    
            P = 0.5 * (H + H.T)
            if opt.qp_regularization > 0.0:
                P += opt.qp_regularization * np.eye(dim)
            q_val = -g
            lb, ub = opt._build_qp_bounds(dim, q_arm_offset, q_head_offset, xi_mount_cam)
            
            dx = qpsolvers.solve_qp(
                spa.csc_matrix(P),
                q_val,
                lb=lb,
                ub=ub,
                solver=opt.qp_solver,
                **opt.qp_kwargs,
            )
            return np.asarray(dx, dtype=np.float64).reshape(-1), total_err
            
        opt.compute_step = custom_compute_step
        
        q_arm_offset, q_head_offset, xi_mount_cam, mount_to_cam_new, head_base_to_cam_new = opt.optimize(q_arm, None, T_meas)
        return q_arm_offset, mount_to_cam_new

    # Run with different weights
    weights = [0.0, 10.0, 100.0, 1000.0, 10000.0]
    for w in weights:
        q_offsets, T_cam = run_opt_with_reg(w)
        # Compute residual translation error
        t_err, r_err = eval_errors(make_transform(T_cam), q_offsets)
        print(f"\nRegularization Weight w = {w}")
        print(f"  Right Arm J0, J1, J3 Offsets: {np.degrees(q_offsets[0]):.3f}, {np.degrees(q_offsets[1]):.3f}, {np.degrees(q_offsets[3]):.3f} deg")
        print(f"  Left Arm J0, J1, J3 Offsets : {np.degrees(q_offsets[7]):.3f}, {np.degrees(q_offsets[8]):.3f}, {np.degrees(q_offsets[10]):.3f} deg")
        print(f"  Camera Pose (Absolute): {[round(x, 4) for x in T_cam]}")
        print(f"  Mean Residual Error: Trans = {t_err:.3f} mm, Rot = {r_err:.3f} deg")

if __name__ == "__main__":
    main()
