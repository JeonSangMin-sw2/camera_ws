import os
import sys
import numpy as np

sys.path.append("/home/rainbow/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer
from scratch.analyze_v12_dataset import OfflineRobot

def analyze_coupling(dataset_path=None, robot_version="1.2"):
    if dataset_path is None:
        dataset_path = "/home/rainbow/camera_ws/result_0903/result_step2/dataset_20260903_053554.npz"
        if not os.path.exists(dataset_path):
            dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260901_125049.npz"
        if not os.path.exists(dataset_path):
            dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260828_082005.npz"

    print(f"=== ANALYZING COUPLING ON DATASET: {os.path.basename(dataset_path)} (Robot v{robot_version}) ===")
    
    urdf_path = f"/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v{robot_version}.urdf"
    if not os.path.exists(urdf_path):
        urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    robot = OfflineRobot(dyn_robot)
    
    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    print(f"Total samples: {len(q_arm_list)}, q_arm shape: {q_arm_list.shape}")
    
    # Set nominal torso
    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    cfg = get_both_arm_config(robot.model(), version=robot_version)
    head_cfg = get_head_config(robot.model())
    
    opt = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        lambda_cam_pos=0.0,
        lambda_cam_rot=0.0,
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=False
    )
    
    param_names = [
        "R_J0", "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6",
        "L_J0", "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6",
        "Head_Pan", "Head_Tilt",
        "Cam_rx", "Cam_ry", "Cam_rz", "Cam_tx", "Cam_ty", "Cam_tz"
    ]
    
    # Evaluate Jacobians at nominal
    q_arm_off = np.zeros(14)
    q_head_off = np.zeros(2)
    xi_cam = np.zeros(6)
    
    J_list = []
    for q_a, q_h, T_pair in zip(q_arm_list, q_head_list, T_meas_list):
        for side_idx, arm_side in enumerate(["right", "left"]):
            Jb, _, T_ee_to_marker, T_model = opt.evaluate_sample(
                q_a, q_h, arm_side, q_arm_off, q_head_off, xi_cam
            )
            J = opt.build_jacobian(q_a, q_h, arm_side, q_arm_off, q_head_off, xi_cam, Jb, T_ee_to_marker, T_model)
            J_list.append(J)
            
    J_total = np.vstack(J_list) # (N*2*6, 22)
    print(f"J_total shape: {J_total.shape}")
    
    # 1. Parameter Sensitivities (Column Norms)
    col_norms = np.linalg.norm(J_total, axis=0)
    print("\n" + "="*70)
    print("1. PARAMETER SENSITIVITY (Column L2 Norm ||J_k||)")
    print("   -> Higher value means the residual is very sensitive to this parameter.")
    print("   -> Lower value means the parameter has weak observability in this motion.")
    print("="*70)
    for name, norm in zip(param_names, col_norms):
        print(f"  {name:12s} : {norm:10.2f}")
        
    # 2. Correlation Matrix (Cosine Similarity)
    D_inv = np.diag(1.0 / np.maximum(col_norms, 1e-12))
    corr_matrix = D_inv @ (J_total.T @ J_total) @ D_inv
    
    print("\n" + "="*70)
    print("2. HEAD & SHOULDER COUPLING ANALYSIS (|Correlation| > 0.3)")
    print("="*70)
    head_indices = [14, 15] # Head_Pan, Head_Tilt
    shoulder_indices = [0, 1, 2, 7, 8, 9] # R_J0, R_J1, R_J2, L_J0, L_J1, L_J2
    cam_indices = list(range(16, 22))
    
    print("\n[A] Head Joints vs Shoulder Joints Correlation:")
    for h_idx in head_indices:
        h_name = param_names[h_idx]
        print(f"\n  --- {h_name} vs Shoulders ---")
        for s_idx in shoulder_indices:
            s_name = param_names[s_idx]
            c_val = corr_matrix[h_idx, s_idx]
            marker = " *** HIGH COUPLING ***" if abs(c_val) > 0.7 else (" * NOTICEABLE *" if abs(c_val) > 0.4 else "")
            print(f"    {h_name:10s} <--> {s_name:8s} : {c_val:+7.4f}{marker}")

    print("\n[B] Head Joints vs Camera Extrinsics Correlation:")
    for h_idx in head_indices:
        h_name = param_names[h_idx]
        print(f"\n  --- {h_name} vs Camera ---")
        for c_idx in cam_indices:
            c_name = param_names[c_idx]
            c_val = corr_matrix[h_idx, c_idx]
            marker = " *** CRITICAL COUPLING ***" if abs(c_val) > 0.8 else (" * NOTICEABLE *" if abs(c_val) > 0.4 else "")
            print(f"    {h_name:10s} <--> {c_name:8s} : {c_val:+7.4f}{marker}")

    print("\n[C] Shoulder Joints vs Camera Extrinsics Correlation:")
    for s_idx in shoulder_indices:
        s_name = param_names[s_idx]
        high_corrs = []
        for c_idx in cam_indices:
            c_val = corr_matrix[s_idx, c_idx]
            if abs(c_val) > 0.3:
                high_corrs.append((param_names[c_idx], c_val))
        if high_corrs:
            corr_str = ", ".join([f"{cn}: {cv:+7.4f}" for cn, cv in high_corrs])
            print(f"  {s_name:8s} correlated with Camera: {corr_str}")
            
    print("\n[D] Shoulder-to-Shoulder (Intra-arm & Inter-arm) Correlation:")
    for i, s1 in enumerate(shoulder_indices):
        for s2 in shoulder_indices[i+1:]:
            c_val = corr_matrix[s1, s2]
            if abs(c_val) > 0.3:
                print(f"  {param_names[s1]:8s} <--> {param_names[s2]:8s} : {c_val:+7.4f}")

    # 3. SVD / Degeneracy directions
    U, S, Vt = np.linalg.svd(J_total, full_matrices=False)
    print("\n" + "="*70)
    print("3. CONDITION NUMBER & WEAKEST DIRECTIONS (Nullspace analysis)")
    print("="*70)
    print(f"Condition Number (All 22 params) : {S[0]/S[-1]:.2f}")
    print(f"Singular values min / max        : {S[-1]:.4f} / {S[0]:.4f}")
    
    print("\nWeakest Singular Vector (Index 21, Singular Value = {:.4f}):".format(S[-1]))
    vec = Vt[-1]
    sorted_idx = np.argsort(np.abs(vec))[::-1]
    for idx in sorted_idx[:8]:
        print(f"    {param_names[idx]:12s} : {vec[idx]:+8.4f} (abs {abs(vec[idx]):.4f})")

    print("\n2nd Weakest Singular Vector (Index 20, Singular Value = {:.4f}):".format(S[-2]))
    vec2 = Vt[-2]
    sorted_idx2 = np.argsort(np.abs(vec2))[::-1]
    for idx in sorted_idx2[:8]:
        print(f"    {param_names[idx]:12s} : {vec2[idx]:+8.4f} (abs {abs(vec2[idx]):.4f})")

    print("\n3rd Weakest Singular Vector (Index 19, Singular Value = {:.4f}):".format(S[-3]))
    vec3 = Vt[-3]
    sorted_idx3 = np.argsort(np.abs(vec3))[::-1]
    for idx in sorted_idx3[:8]:
        print(f"    {param_names[idx]:12s} : {vec3[idx]:+8.4f} (abs {abs(vec3[idx]):.4f})")

if __name__ == "__main__":
    analyze_coupling()
