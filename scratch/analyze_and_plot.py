import os
import sys
import glob
import numpy as np
import yaml
import json

# Ensure headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Add workspace path
sys.path.append("/home/jsm/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer

# Standalone 3D circle fitting (SVD method)
def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    basis1 = Vt[0, :]
    basis2 = Vt[1, :]
    
    points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
    
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    center_3d = centroid + xc * basis1 + yc * basis2
    pts_center = points - center_3d
    dists_from_center = np.linalg.norm(pts_center - np.outer(np.dot(pts_center, normal), normal), axis=1)
    residuals = dists_from_center - r
    rmse = np.sqrt(np.mean(residuals**2))
    
    return center_3d, normal, r, rmse, residuals, points_2d, (xc, yc)

# Offline Robot implementation
class OfflineRobot:
    def __init__(self, dyn_robot, version="1.3"):
        self.dyn_robot = dyn_robot
        self.version = version
        self._model = self._create_model_meta()
        
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self.dyn_robot
        
    def get_state(self):
        class State:
            def __init__(self, dof):
                self.position = np.zeros(dof)
        return State(self.dyn_robot.get_dof())
        
    def _create_model_meta(self):
        class ModelMeta:
            def __init__(self, dyn_robot):
                self.robot_joint_names = dyn_robot.get_joint_names()
                self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
                self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
                self.head_idx = [self.robot_joint_names.index(f"head_{i}") for i in range(2)] if "head_0" in self.robot_joint_names else []
                self.torso_idx = [self.robot_joint_names.index(f"torso_{i}") for i in range(6)] if "torso_0" in self.robot_joint_names else []
        return ModelMeta(self.dyn_robot)

def load_offline_robot(version="1.3"):
    urdf_path = f"/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v{version}.urdf"
    if not os.path.exists(urdf_path):
        urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot, version)

def main():
    print("================================================================================")
    print(" 1. STEP 1 - SWEEP CIRCLE FITTING ANALYSIS & OUTLIER DETECTION")
    print("================================================================================")
    
    txt_dir = "/home/jsm/camera_ws/result/result_txt"
    sweep_files = sorted(glob.glob(os.path.join(txt_dir, "sweep_points_*_marker_axis_*.txt")))
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    axes1 = axes1.ravel()
    
    summary_report = []
    summary_report.append("=== STEP 1 SWEEP DATA DIAGNOSTICS ===")
    
    for idx, fpath in enumerate(sweep_files[:6]):
        fname = os.path.basename(fpath)
        angles = []
        cam_pts = []
        
        with open(fpath, "r") as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip():
                    continue
                parts = [float(x.strip()) for x in line.strip().split(",")]
                angles.append(parts[0])
                cam_pts.append(parts[1:4])
        
        cam_pts = np.array(cam_pts)
        angles = np.array(angles)
        
        center, normal, r, rmse, residuals, pts_2d, center_2d = fit_circle_3d(cam_pts)
        
        # Outlier Detection (Threshold: 1.5 mm deviation)
        outliers = np.abs(residuals) > 1.5
        num_outliers = np.sum(outliers)
        
        report_line = (
            f"[{fname}]\n"
            f"  - Points Count: {len(cam_pts)}, Outliers (>1.5mm): {num_outliers} ({num_outliers/len(cam_pts)*100:.1f}%)\n"
            f"  - Fitted Radius: {r:.2f} mm\n"
            f"  - RMSE Noise: {rmse:.4f} mm, Max Dev: {np.max(np.abs(residuals)):.4f} mm"
        )
        print(report_line)
        summary_report.append(report_line)
        
        # Plot circle fit & residuals
        ax = axes1[idx]
        # Plot nominal vs measured distance from center
        ax.plot(angles, residuals, 'o-', label='Residuals (mm)', color='royalblue', markersize=3)
        if num_outliers > 0:
            ax.plot(angles[outliers], residuals[outliers], 'x', color='red', label='Outliers (>1.5mm)', markersize=6, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--')
        ax.axhline(1.5, color='orange', linestyle=':')
        ax.axhline(-1.5, color='orange', linestyle=':')
        ax.set_title(fname.replace("sweep_points_", "").replace(".txt", ""), fontsize=10)
        ax.set_xlabel("Joint Angle (deg)")
        ax.set_ylabel("Radial Error (mm)")
        ax.grid(True)
        ax.legend()
        
    fig1.suptitle("Step 1 Sweep Data Circle Fitting Residuals & Jumpy Points", fontsize=14)
    plt.tight_layout()
    fig1.savefig("/home/jsm/camera_ws/scratch/step1_sweep_analysis.png", dpi=150)
    plt.close(fig1)
    
    print("\n================================================================================")
    print(" 2. STEP 2 - COUPLING DEGREE ANALYSIS (PARAMETER CORRELATION)")
    print("================================================================================")
    
    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260716_205134.npz"
    q_arm_list, q_head_list, marker_list = load_npz_dataset(dataset_path)
    
    # Load offline robot (v1.3 based on file properties)
    robot = load_offline_robot("1.3")
    model = robot.model()
    
    # Setup Torso positions matching step2 data collection
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for j_idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[j_idx] = val
        
    # Read settings to fetch Nominal Extrinsics
    with open("/home/jsm/camera_ws/config/setting.yaml", "r") as f:
        setting = yaml.safe_load(f)
    camera_cfg = setting.get("camera", {})
    
    ee_to_marker_nom = {
        "left": camera_cfg.get("Tf_to_marker_left_v13", [0.097, 0.0, -0.005, 90.0, 0.0, -90.0]),
        "right": camera_cfg.get("Tf_to_marker_right_v13", [0.097, 0.0, -0.005, 90.0, 0.0, -90.0])
    }
    
    cfg = get_both_arm_config(robot.model(), version="1.3")
    head_cfg = get_head_config(robot.model())
    
    opt = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_nom,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        lambda_cam_pos=0.0,
        lambda_cam_rot=0.0,
    )
    
    J_list = []
    q_arm_offset = np.zeros(len(opt.arm_idx))
    q_head_offset = np.zeros(len(opt.head_idx))
    xi_mount_cam = np.zeros(6)
    
    for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_list, marker_list):
        for side_idx, arm_side in enumerate(["right", "left"]):
            T_meas = T_meas_pair[side_idx]
            Jb_joint, _, T_ee_to_marker, T_model = opt.evaluate_sample(
                q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam
            )
            J = opt.build_jacobian(
                q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam,
                Jb_joint, T_ee_to_marker, T_model
            )
            J_list.append(J)
            
    J_total = np.vstack(J_list) # Shape (N*12, 22)
    H = J_total.T @ J_total
    
    # Calculate parameter correlation matrix from Covariance
    C = np.linalg.inv(H + 1e-6 * np.eye(H.shape[0]))
    std_dev = np.sqrt(np.diag(C))
    R_corr = C / np.outer(std_dev, std_dev)
    
    # Analyze condition numbers
    S = np.linalg.svd(J_total, compute_uv=False)
    cond_all = S[0] / S[-1]
    
    S_joint = np.linalg.svd(J_total[:, :16], compute_uv=False)
    cond_joint = S_joint[0] / S_joint[-1]
    
    S_cam = np.linalg.svd(J_total[:, 16:], compute_uv=False)
    cond_cam = S_cam[0] / S_cam[-1]
    
    param_names = (
        [f"R_J{i}" for i in range(7)] + 
        [f"L_J{i}" for i in range(7)] + 
        ["Head_Pan", "Head_Tilt"] + 
        ["Cam_rx", "Cam_ry", "Cam_rz", "Cam_tx", "Cam_ty", "Cam_tz"]
    )
    
    coupling_report = (
        f"=== STEP 2 COUPLING & OBSERVABILITY ANALYSIS ===\n"
        f"Condition Number (All 22 Params) : {cond_all:.2f}\n"
        f"Condition Number (Joint Only)     : {cond_joint:.2f}\n"
        f"Condition Number (Camera Only)    : {cond_cam:.2f}\n"
        f"\nHighly Coupled Parameter Pairs (|Correlation| > 0.8):\n"
    )
    print(coupling_report)
    summary_report.append(coupling_report)
    
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            corr_val = R_corr[i, j]
            if abs(corr_val) > 0.8:
                line = f"  - {param_names[i]} <--> {param_names[j]} : {corr_val:+.4f}"
                print(line)
                summary_report.append(line)
                
    # Plot Correlation Matrix
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im = ax2.imshow(R_corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig2.colorbar(im, ax=ax2, label='Correlation Coefficient')
    ax2.set_xticks(np.arange(len(param_names)))
    ax2.set_yticks(np.arange(len(param_names)))
    ax2.set_xticklabels(param_names, rotation=90, fontsize=8)
    ax2.set_yticklabels(param_names, fontsize=8)
    ax2.set_title("Parameter Correlation Matrix (Step 2 - 22 DOFs)", fontsize=12)
    plt.tight_layout()
    fig2.savefig("/home/jsm/camera_ws/scratch/step2_coupling_matrix.png", dpi=150)
    plt.close(fig2)
    
    print("\n================================================================================")
    print(" 3. STEP 2 - 3D TRAJECTORY VISUALIZATION IN HEAD BASE FRAME")
    print("================================================================================")
    
    # Transform dataset points into link_head_0 frame
    model = robot.model()
    # T_head_base_to_cam_nom
    T_head_base_to_cam_nom = opt.T_mount_to_cam_nom.copy()
    
    pts_marker_meas_right = []
    pts_marker_meas_left = []
    pts_marker_model_right = []
    pts_marker_model_left = []
    
    for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_list, marker_list):
        # 1. Setup q_full in kinematics model
        q_full = robot.get_state().position.copy()
        q_full[opt.arm_idx] = q_arm
        if opt.head_idx is not None and q_head is not None:
            q_full[opt.head_idx] = q_head
            
        # Torso
        for j_idx, val in zip(model.torso_idx, torso_angles):
            q_full[j_idx] = val
            
        # Compute FK for Right and Left Arms
        for arm_side, side_idx in [("right", 0), ("left", 1)]:
            # State from link_head_0 to arm ee link
            state = robot.get_dynamics().make_state(
                ["link_head_0", opt.ee_links[arm_side]],
                model.robot_joint_names
            )
            state.set_q(q_full)
            robot.get_dynamics().compute_forward_kinematics(state)
            T_head_base_to_ee = robot.get_dynamics().compute_transformation(state, 0, 1)
            
            # Predict marker pose (Model)
            T_ee_to_marker = opt.get_nominal_ee_to_marker(arm_side)
            T_model_marker = T_head_base_to_ee @ T_ee_to_marker
            p_model = T_model_marker[:3, 3] * 1000.0 # to mm
            
            # Measured marker pose:
            # We need the camera pose relative to link_head_0
            state_cam = robot.get_dynamics().make_state(
                ["link_head_0", "link_head_2"],
                model.robot_joint_names
            )
            state_cam.set_q(q_full)
            robot.get_dynamics().compute_forward_kinematics(state_cam)
            T_head_base_to_cam_link = robot.get_dynamics().compute_transformation(state_cam, 0, 1)
            T_head_base_to_cam = T_head_base_to_cam_link @ T_head_base_to_cam_nom
            
            T_meas = T_meas_pair[side_idx]
            T_meas_marker = T_head_base_to_cam @ T_meas
            p_meas = T_meas_marker[:3, 3] * 1000.0 # to mm
            
            if arm_side == "right":
                pts_marker_model_right.append(p_model)
                pts_marker_meas_right.append(p_meas)
            else:
                pts_marker_model_left.append(p_model)
                pts_marker_meas_left.append(p_meas)
                
    pts_marker_meas_right = np.array(pts_marker_meas_right)
    pts_marker_meas_left = np.array(pts_marker_meas_left)
    pts_marker_model_right = np.array(pts_marker_model_right)
    pts_marker_model_left = np.array(pts_marker_model_left)
    
    # 3D Plotting
    try:
        fig3 = plt.figure(figsize=(14, 10))
        
        # Right Arm Marker Trajectory
        ax_r = fig3.add_subplot(121, projection='3d')
        ax_r.scatter(pts_marker_meas_right[:, 0], pts_marker_meas_right[:, 1], pts_marker_meas_right[:, 2], 
                   c='crimson', label='Measured Marker', alpha=0.6, s=15)
        ax_r.scatter(pts_marker_model_right[:, 0], pts_marker_model_right[:, 1], pts_marker_model_right[:, 2], 
                   c='navy', label='Model Predicted', alpha=0.6, s=15)
        ax_r.set_title("Right Marker 3D Trajectory in Head Base Frame (link_head_0)")
        ax_r.set_xlabel("X (mm)")
        ax_r.set_ylabel("Y (mm)")
        ax_r.set_zlabel("Z (mm)")
        ax_r.grid(True)
        ax_r.legend()
        
        # Left Arm Marker Trajectory
        ax_l = fig3.add_subplot(122, projection='3d')
        ax_l.scatter(pts_marker_meas_left[:, 0], pts_marker_meas_left[:, 1], pts_marker_meas_left[:, 2], 
                   c='crimson', label='Measured Marker', alpha=0.6, s=15)
        ax_l.scatter(pts_marker_model_left[:, 0], pts_marker_model_left[:, 1], pts_marker_model_left[:, 2], 
                   c='navy', label='Model Predicted', alpha=0.6, s=15)
        ax_l.set_title("Left Marker 3D Trajectory in Head Base Frame (link_head_0)")
        ax_l.set_xlabel("X (mm)")
        ax_l.set_ylabel("Y (mm)")
        ax_l.set_zlabel("Z (mm)")
        ax_l.grid(True)
        ax_l.legend()
        
        fig3.suptitle("Step 2 Marker 3D Trajectory Comparison (link_head_0 frame)", fontsize=14)
        plt.tight_layout()
        fig3.savefig("/home/jsm/camera_ws/scratch/step2_marker_trajectories.png", dpi=150)
        plt.close(fig3)
    except Exception as e:
        print(f"Skipping 3D plotting due to Matplotlib 3D projection issue: {e}")
    
    # Write summary report to file
    with open("/home/jsm/camera_ws/scratch/analysis_summary.txt", "w") as f:
        f.write("\n".join(summary_report))
        
    print("\nDiagnostics complete. Plots saved:")
    print("  - Circle fitting / Outliers: /home/jsm/camera_ws/scratch/step1_sweep_analysis.png")
    print("  - Parameter coupling: /home/jsm/camera_ws/scratch/step2_coupling_matrix.png")
    print("  - 3D trajectories: /home/jsm/camera_ws/scratch/step2_marker_trajectories.png")
    print("  - Text Summary: /home/jsm/camera_ws/scratch/analysis_summary.txt")

if __name__ == "__main__":
    main()
