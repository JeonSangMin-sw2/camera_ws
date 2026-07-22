import os
import sys
import numpy as np
import yaml

sys.path.append("/home/jsm/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer, se3_log, adjoint
from scratch.analyze_v12_dataset import OfflineRobot

def load_offline_robot():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        return None
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def diagnose():
    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260722_175935.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260722_173120.npz"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    robot = load_offline_robot()
    if robot is None:
        print("Failed to load robot.")
        return

    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())

    staged_joint_offsets = {
        "right": {"joint3": -1.2302416979621213, "joint5": -7.4496923496490615, "joint6": -5.641026190394765},
        "left": {"joint3": -1.6685655234061672, "joint5": -4.848966338794833, "joint6": 10.556777124959687}
    }

    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_joint_offsets,
    )

    print("=========================================================================")
    print(" 1. DIAGNOSING MEASURED MARKER DATA QUALITY (T_meas)")
    print("=========================================================================")
    print(f"Total poses in dataset: {len(q_arm_list)}")
    
    right_pos = []
    left_pos = []
    for T_pair in T_meas_list:
        right_pos.append(T_pair[0][:3, 3])
        left_pos.append(T_pair[1][:3, 3])
    
    right_pos = np.array(right_pos)
    left_pos = np.array(left_pos)

    print(f"Right Marker Range: X [{right_pos[:,0].min():.3f}, {right_pos[:,0].max():.3f}], Y [{right_pos[:,1].min():.3f}, {right_pos[:,1].max():.3f}], Z [{right_pos[:,2].min():.3f}, {right_pos[:,2].max():.3f}] m")
    print(f"Left  Marker Range: X [{left_pos[:,0].min():.3f}, {left_pos[:,0].max():.3f}], Y [{left_pos[:,1].min():.3f}, {left_pos[:,1].max():.3f}], Z [{left_pos[:,2].min():.3f}, {left_pos[:,2].max():.3f}] m")

    # Evaluate Residuals at Ground-Truth Candidate vs Optimizer Solution
    q_arm_opt, q_head_opt, xi_cam_opt, _, _ = optimizer.optimize(q_arm_list, q_head_list, T_meas_list)

    # Candidate GT offsets (j0 ~ 2 deg, others ~ 0.5 deg)
    q_arm_gt = np.zeros(14)
    q_arm_gt[0] = np.radians(2.0) # Right J1 (P)
    q_arm_gt[3] = np.radians(-1.23) # Right J4 (E)
    q_arm_gt[5] = np.radians(-7.45) # Right J6 (W2)
    q_arm_gt[6] = np.radians(-5.64) # Right J7 (W3)
    
    q_arm_gt[10] = np.radians(-1.67) # Left J4
    q_arm_gt[12] = np.radians(-4.85) # Left J6
    q_arm_gt[13] = np.radians(10.56) # Left J7
    
    q_head_gt = np.zeros(2)

    def compute_residual_rms(q_arm_off, q_head_off, xi_cam):
        rot_errs = []
        pos_errs = []
        for q_a, q_h, T_pair in zip(q_arm_list, q_head_list, T_meas_list):
            for side_idx, arm_side in enumerate(["right", "left"]):
                T_m = T_pair[side_idx]
                _, _, _, T_model = optimizer.evaluate_sample(
                    q_a, q_h, arm_side, q_arm_off, q_head_off, xi_cam
                )
                T_err = np.linalg.inv(T_model) @ T_m
                xi = se3_log(T_err)
                rot_errs.append(np.linalg.norm(xi[:3]))
                pos_errs.append(np.linalg.norm(xi[3:]))
        return np.rad2deg(np.mean(rot_errs)), np.mean(pos_errs) * 1000.0

    rot_opt, pos_opt = compute_residual_rms(q_arm_opt, q_head_opt, xi_cam_opt)
    rot_gt, pos_gt = compute_residual_rms(q_arm_gt, q_head_gt, np.zeros(6))

    print("\n=========================================================================")
    print(" 2. RESIDUAL COMPARISON: OPTIMIZER SOLUTION vs GROUND TRUTH CANDIDATE")
    print("=========================================================================")
    print(f" Optimizer Solution (J0=5.4deg, J2=6.9deg, J4=-8.7deg) -> Rot Error: {rot_opt:.4f} deg, Pos Error: {pos_opt:.4f} mm")
    print(f" Ground-Truth Candidate (J0=2.0deg, others ~ 0.5deg)  -> Rot Error: {rot_gt:.4f} deg, Pos Error: {pos_gt:.4f} mm")
    print("=========================================================================")

    # Jacobian SVD Analysis to check Kinematic Coupling / Degeneracy
    print("\n=========================================================================")
    print(" 3. JACOBIAN COUPLING & DEGENERACY ANALYSIS (SVD)")
    print("=========================================================================")
    H_sum = np.zeros((22, 22))
    for q_a, q_h, T_pair in zip(q_arm_list, q_head_list, T_meas_list):
        for side_idx, arm_side in enumerate(["right", "left"]):
            Jb, _, T_ee_to_marker, T_model = optimizer.evaluate_sample(
                q_a, q_h, arm_side, q_arm_opt, q_head_opt, xi_cam_opt
            )
            J = optimizer.build_jacobian(q_a, q_h, arm_side, q_arm_opt, q_head_opt, xi_cam_opt, Jb, T_ee_to_marker, T_model)
            H_sum += J.T @ J

    singular_values = np.linalg.svd(H_sum, compute_uv=False)
    cond_num = singular_values[0] / singular_values[-1]
    print(f"Hessian Condition Number (Higher means worse coupling): {cond_num:.2e}")
    print(f"Smallest 5 Singular Values of Hessian: {singular_values[-5:]}")
    print("=========================================================================")

if __name__ == "__main__":
    diagnose()
