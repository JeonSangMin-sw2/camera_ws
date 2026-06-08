import time
import logging
import os
import numpy as np
import rby1_sdk as rby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from CalibratorBase import BaseCalibrator

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.use_angle_based_fitting = True

    def perform_3step_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        if log_callback:
            log_callback("\n" + "="*60)
            log_callback("   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE")
            log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
            log_callback("="*60 + "\n")
            
        def run_single_sweep(offset):
            return self.perform_calibration_sweep_continuous(
                arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                current_offset_deg=offset, sweep_duration=sweep_duration,
                use_angle_based_fitting=use_angle_based_fitting
            )
            
        max_iterations = 8
        staged_offset = current_offset_deg
        final_res = None
        first_res = None
        converged = False
        
        for i in range(1, max_iterations + 1):
            if log_callback:
                log_callback(f"\n[ITERATION {i}/{max_iterations}] Sweeping with staged offset {staged_offset:.4f}°...")
                
            res = run_single_sweep(staged_offset)
            if not res:
                if log_callback: log_callback(f"[ERROR] Iteration {i} sweep failed. Aborting calibration.")
                return None
                
            if first_res is None:
                first_res = res

            angle_error = res.get('angle_between_normals', 0.0)
            sign = res.get('sign', 1.0)
            center_dist = res.get('center_dist', 999.0)
            r_A = res.get('r_A', 0.0)
            r_B = res.get('r_B', 0.0)
            size_error = abs(r_A - r_B)
            current_error = max(size_error, center_dist)
            
            # Print iteration summary
            if log_callback:
                log_callback(f"  * Angle Error (Deviation)     : {angle_error:.4f}°")
                log_callback(f"  * Circle Size Error (r_A-r_B) : {size_error:.4f} mm")
                log_callback(f"  * Center Distance Error       : {center_dist:.4f} mm")
                log_callback(f"  * Max Fitting Error Metric    : {current_error:.4f} mm")
            
            # Direct correction
            # When looking at angle error (use_angle_based_fitting is True), we deactivate the 0.05 deg step correction.
            # Otherwise (when use_angle_based_fitting is False), if circle center distance is close (<= 1.0 mm),
            # we apply a 0.05 deg step correction instead of the raw angle_error.
            if (not use_angle_based_fitting) and (center_dist <= 1.0):
                step_correction = 0.05
                if log_callback:
                    log_callback(f"  [STEP CONTROL] Center distance {center_dist:.4f} mm is close (<= 1.0 mm). Applying 0.05° step correction instead of {angle_error:.4f}°.")
            else:
                step_correction = angle_error

            staged_offset += sign * step_correction
            final_res = res
            
            # Elbow Safety Check: Elbow offset must unconditionally be negative.
            if mode == "elbow" and staged_offset > 0.0:
                if log_callback:
                    log_callback(f"  [SAFETY CONTROL] Elbow joint offset must unconditionally be negative. Changing sign of positive staged offset {staged_offset:.4f}° to {-staged_offset:.4f}° for safety!")
                staged_offset = -staged_offset
                
            if log_callback:
                log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                
            # Convergence check: angle error <= 0.1° and center distance <= 0.1 mm
            if angle_error <= 0.1 and center_dist <= 0.1:
                converged = True
                if log_callback:
                    log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                    log_callback(f"  * Circle Normals Angle Error: {angle_error:.4f}° <= 0.1°")
                    log_callback(f"  * Center Distance Error: {center_dist:.4f} mm <= 0.1 mm")
                    log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                break

        # Elbow Safety Check on final recommended offset
        if mode == "elbow" and staged_offset > 0.0:
            if log_callback:
                log_callback(f"  [SAFETY CONTROL] Elbow joint offset must unconditionally be negative. Changing sign of final positive offset {staged_offset:.4f}° to {-staged_offset:.4f}° for safety!")
            staged_offset = -staged_offset

        # Run one final validation sweep with the recommended joint offset
        if log_callback:
            log_callback(f"\n[VALIDATION SWEEP] Running final validation sweep with recommended offset {staged_offset:.4f}°...")
        validation_res = run_single_sweep(staged_offset)
        if validation_res:
            validation_res['recommended_joint_offset'] = staged_offset
            validation_res['converged'] = converged
            if first_res:
                plot_path = self.save_calibration_comparison_plot(arm_side, mode, first_res, validation_res, log_callback=log_callback)
                validation_res['plot_path_combined'] = plot_path
            return validation_res
        else:
            if log_callback:
                log_callback("[WARN] Validation sweep failed. Returning last calibration result.")
            if final_res:
                final_res['recommended_joint_offset'] = staged_offset
                final_res['converged'] = converged
                if first_res:
                    plot_path = self.save_calibration_comparison_plot(arm_side, mode, first_res, final_res, log_callback=log_callback)
                    final_res['plot_path_combined'] = plot_path
            return final_res

    def perform_move_to_ready_pose(self, arm_side, mode, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving to {arm_side} Pitch/Head Ready Pose (Mode: {mode})...")
        torso = [0, 0, 0, 0, 0, 0]
        
        # 1. First move the inactive arm to zero pose to avoid collision
        if log_callback: log_callback("[INFO] Moving inactive arm to zero pose first...")
        if arm_side == "right":
            success_other = self.movej(self.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0)
        else:
            success_other = self.movej(self.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0)
            
        if not success_other:
            if log_callback: log_callback("[ERROR] Failed to move inactive arm to zero pose.")
            return False
            
        # 2. Move active arm and head/torso to ready pose
        if log_callback: log_callback("[INFO] Moving active arm, torso, and head to ready pose...")
        if mode == "wrist_pitch":
            if arm_side == "right":
                right_arm = np.deg2rad([-55, -45, 25, -127, 90, 0, 0])
                left_arm = None
            else:
                right_arm = None
                left_arm = np.deg2rad([-55, 45, -25, -127, -90, 0, 0])
        elif mode == "elbow":
            if arm_side == "right":
                right_arm = np.deg2rad([-107, -17, 0, 0, 73, -90, 73])
                left_arm = None
            else:
                right_arm = None
                left_arm = np.deg2rad([-107, 17, 0, 0, -73, -90, -73])
        else: # head mode
            if arm_side == "right":
                right_arm = np.deg2rad([-90, -45, 73, -107, 90, 90, 0])
                left_arm = None
            else:
                right_arm = None
                left_arm = np.deg2rad([-90, 45, -73, -107, -90, 90, 0])

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=[0, 0], minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def save_debug_orthogonal_plot(self, arm_side, frame, dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, angle_error_deg=None, log_callback=None):
        try:
            pts_a = []
            pts_b = []
            
            # Project points depending on frame
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                if frame == "camera":
                    pts_a.append(p_cam * 1000.0)
                elif frame == "torso":
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    pts_a.append(p_meas_t5 * 1000.0)
                elif frame == "ee":
                    q_mod = np.array(q_full)
                    q_mod[arm_idx[cand_joint]] += optimal_offset_rad
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    pts_a.append(p_ee * 1000.0)

            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                if frame == "camera":
                    pts_b.append(p_cam * 1000.0)
                elif frame == "torso":
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    pts_b.append(p_meas_t5 * 1000.0)
                elif frame == "ee":
                    q_mod = np.array(q_full)
                    q_mod[arm_idx[cand_joint]] += optimal_offset_rad
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    pts_b.append(p_ee * 1000.0)

            pts_a = np.array(pts_a)
            pts_b = np.array(pts_b)
            
            # 3D fit circles
            c_A, R_c_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A = BaseCalibrator.fit_circle_3d(pts_a)
            c_B, R_c_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B = BaseCalibrator.fit_circle_3d(pts_b)
            
            n_A = R_c_A[:, 2]
            n_B = R_c_B[:, 2]
            u_A = R_c_A[:, 0]
            v_A = R_c_A[:, 1]
            u_B = R_c_B[:, 0]
            v_B = R_c_B[:, 1]

            angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
            diff_centers = c_B - c_A
            center_dist = np.linalg.norm(diff_centers - np.dot(diff_centers, n_A) * n_A)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            def generate_circle_pts(center, normal, radius, u, v, num_points=100):
                theta = np.linspace(0, 2*np.pi, num_points)
                circle_pts = []
                for t in theta:
                    p = center + radius * (np.cos(t) * u + np.sin(t) * v)
                    circle_pts.append(p)
                return np.array(circle_pts)

            circle_pts_a = generate_circle_pts(c_A, n_A, r_A, u_A, v_A)
            circle_pts_b = generate_circle_pts(c_B, n_B, r_B, u_B, v_B)
            
            # Subplot 1 (Top-Left): Sweep A Circle Fit (2D plane projection)
            theta_fit = np.linspace(0, 2*np.pi, 200)
            fit_x = uc_A + r_A * np.cos(theta_fit)
            fit_y = vc_A + r_A * np.sin(theta_fit)
            axes[0, 0].scatter(pts_2d_A[:, 0], pts_2d_A[:, 1], c='red', s=15, alpha=0.6, label='Raw Points')
            axes[0, 0].plot(fit_x, fit_y, 'r--', linewidth=2, label=f'Fit Circle (r={r_A:.1f}mm)')
            axes[0, 0].scatter([uc_A], [vc_A], c='darkred', marker='X', s=80, label='Center')
            axes[0, 0].set_xlabel('U (mm)')
            axes[0, 0].set_ylabel('V (mm)')
            axes[0, 0].set_title(f'Sweep A Local 2D Circle Fit (RMSE: {rmse_A:.4f} mm)')
            axes[0, 0].set_aspect('equal')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
            
            # Subplot 2 (Top-Right): Sweep B Circle Fit (2D plane projection)
            fit_x_b = uc_B + r_B * np.cos(theta_fit)
            fit_y_b = vc_B + r_B * np.sin(theta_fit)
            axes[0, 1].scatter(pts_2d_B[:, 0], pts_2d_B[:, 1], c='blue', s=15, alpha=0.6, label='Raw Points')
            axes[0, 1].plot(fit_x_b, fit_y_b, 'b--', linewidth=2, label=f'Fit Circle (r={r_B:.1f}mm)')
            axes[0, 1].scatter([uc_B], [vc_B], c='darkblue', marker='X', s=80, label='Center')
            axes[0, 1].set_xlabel('U (mm)')
            axes[0, 1].set_ylabel('V (mm)')
            axes[0, 1].set_title(f'Sweep B Local 2D Circle Fit (RMSE: {rmse_B:.4f} mm)')
            axes[0, 1].set_aspect('equal')
            axes[0, 1].grid(True)
            axes[0, 1].legend()

            # Subplot 3 (Bottom-Left): Comparison Top View (X-Y Projection)
            axes[1, 0].scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.5, label='Sweep A Raw')
            axes[1, 0].scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
            axes[1, 0].plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r-', linewidth=1.5, label='Sweep A Fit')
            axes[1, 0].plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b-', linewidth=1.5, label='Sweep B Fit')
            axes[1, 0].scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
            axes[1, 0].scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
            axes[1, 0].plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label='Center Shift')
            axes[1, 0].set_xlabel('X (mm)')
            axes[1, 0].set_ylabel('Y (mm)')
            axes[1, 0].set_title('Top View Comparison (X-Y Projection)')
            axes[1, 0].set_aspect('equal')
            axes[1, 0].grid(True)
            axes[1, 0].legend()

            # Subplot 4 (Bottom-Right): Comparison Side View (Y-Z Projection)
            axes[1, 1].scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.5, label='Sweep A Raw')
            axes[1, 1].scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
            axes[1, 1].plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r-', linewidth=1.5, label='Sweep A Fit')
            axes[1, 1].plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b-', linewidth=1.5, label='Sweep B Fit')
            axes[1, 1].scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
            axes[1, 1].scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
            
            # Normal Vectors Projection
            scale = min(r_A, r_B) * 0.4
            axes[1, 1].arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
            axes[1, 1].arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
            axes[1, 1].set_xlabel('Y (mm)')
            axes[1, 1].set_ylabel('Z (mm)')
            axes[1, 1].set_title('Side View Comparison (Y-Z Projection)')
            axes[1, 1].set_aspect('equal')
            axes[1, 1].grid(True)
            axes[1, 1].legend()

            display_angle = angle_error_deg if angle_error_deg is not None else angle_between_normals
            status_text = "PASS" if (display_angle < 0.1 and center_dist < 0.1) else "WARNING"
            fig.suptitle(
                f"Orthogonal Multi-View Analysis ({arm_side.upper()} Arm, {frame.upper()} Frame)\n"
                f"Status: {status_text} | Axis Angle Error: {display_angle:.4f}° (Target < 0.1°)\n"
                f"Axis Center Distance: {center_dist:.4f} mm (Target < 0.1 mm)",
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            result_dir = os.path.join(os.path.dirname(__file__), "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"debug_orthogonal_circles_{arm_side}_{frame}.png"))
            plt.savefig(plot_save_path, dpi=150)
            plt.close()
            if log_callback:
                log_callback(f"[SUCCESS] Orthogonal debug plot saved to: {plot_save_path}")
                log_callback(f"  * Alignment check: {status_text} (Angle error = {display_angle:.4f}°, Center distance = {center_dist:.4f} mm)")
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to save orthogonal debug plot for {frame}: {e}")

    def save_calibration_comparison_plot(self, arm_side, mode, first_res, final_res, log_callback=None):
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(16, 14))

            def plot_column(res, col_idx, stage_name):
                pts_a = res['pts_a_cam']
                pts_b = res['pts_b_cam']
                c_A = res['c_A']
                c_B = res['c_B']
                n_A = res['n_A']
                n_B = res['n_B']
                r_A = res['r_A']
                r_B = res['r_B']
                rmse_A = res['rmse_A']
                rmse_B = res['rmse_B']
                angle_error = res['angle_between_normals']
                center_dist = res['center_dist']

                # Fit circles again to retrieve the local coordinate frames R_c_A, R_c_B
                c_A_fit, R_c_A, r_A_fit, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_a)
                c_B_fit, R_c_B, r_B_fit, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_b)

                theta = np.linspace(0, 2 * np.pi, 200)

                # --- 1. 2D SUBPLOT (Row 0, Col col_idx) ---
                pts_a_proj = (pts_a - c_A) @ R_c_A[:, :2]
                pts_b_proj = (pts_b - c_A) @ R_c_A[:, :2]
                c_B_proj = (c_B - c_A) @ R_c_A[:, :2]

                circle_A_2d_x = r_A * np.cos(theta)
                circle_A_2d_y = r_A * np.sin(theta)

                circle_B_3d = c_B + r_B * (np.cos(theta)[:, None] * R_c_B[:, 0] + np.sin(theta)[:, None] * R_c_B[:, 1])
                circle_B_proj = (circle_B_3d - c_A) @ R_c_A[:, :2]

                ax_2d = fig.add_subplot(2, 2, col_idx + 1)
                
                # Plot Sweep A
                ax_2d.scatter(pts_a_proj[:, 0], pts_a_proj[:, 1], c='red', s=15, alpha=0.5, label=f'Sweep A (r={r_A:.1f}mm, RMSE={rmse_A:.2f}mm)')
                ax_2d.plot(circle_A_2d_x, circle_A_2d_y, 'r--', linewidth=2)
                ax_2d.scatter([0], [0], c='darkred', marker='X', s=100, label='Center A (0,0)')

                # Plot Sweep B
                ax_2d.scatter(pts_b_proj[:, 0], pts_b_proj[:, 1], c='blue', s=15, alpha=0.5, label=f'Sweep B (r={r_B:.1f}mm, RMSE={rmse_B:.2f}mm)')
                ax_2d.plot(circle_B_proj[:, 0], circle_B_proj[:, 1], 'b--', linewidth=2)
                ax_2d.scatter([c_B_proj[0]], [c_B_proj[1]], c='darkblue', marker='X', s=100, label=f'Center B ({c_B_proj[0]:.1f},{c_B_proj[1]:.1f})')

                ax_2d.plot([0, c_B_proj[0]], [0, c_B_proj[1]], color='purple', linestyle=':', linewidth=2, label=f'Center Shift = {center_dist:.2f}mm')

                ax_2d.set_xlabel('U (mm)')
                ax_2d.set_ylabel('V (mm)')
                ax_2d.set_title(f'[{stage_name}] 2D Combined Circle Fit')
                ax_2d.set_aspect('equal')
                ax_2d.grid(True)
                ax_2d.legend(loc='upper right')

                # --- 2. 3D SUBPLOT (Row 1, Col col_idx) ---
                ax_3d = fig.add_subplot(2, 2, col_idx + 3, projection='3d')

                ax_3d.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], c='red', s=10, alpha=0.4)
                ax_3d.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], c='blue', s=10, alpha=0.4)

                circle_A_3d = c_A + r_A * (np.cos(theta)[:, None] * R_c_A[:, 0] + np.sin(theta)[:, None] * R_c_A[:, 1])
                ax_3d.plot(circle_A_3d[:, 0], circle_A_3d[:, 1], circle_A_3d[:, 2], 'r-', linewidth=2, label='Sweep A Fit')
                ax_3d.plot(circle_B_3d[:, 0], circle_B_3d[:, 1], circle_B_3d[:, 2], 'b-', linewidth=2, label='Sweep B Fit')

                ax_3d.scatter([c_A[0]], [c_A[1]], [c_A[2]], c='darkred', marker='X', s=120)
                ax_3d.scatter([c_B[0]], [c_B[1]], [c_B[2]], c='darkblue', marker='X', s=120)

                scale = min(r_A, r_B) * 0.5
                ax_3d.quiver(c_A[0], c_A[1], c_A[2], n_A[0]*scale, n_A[1]*scale, n_A[2]*scale, color='darkred', linewidth=3, arrow_length_ratio=0.15, label=f'Normal A: {np.round(n_A,2)}')
                ax_3d.quiver(c_B[0], c_B[1], c_B[2], n_B[0]*scale, n_B[1]*scale, n_B[2]*scale, color='darkblue', linewidth=3, arrow_length_ratio=0.15, label=f'Normal B: {np.round(n_B,2)}')

                ax_3d.plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2)

                # Set equal aspect ratio for 3D plot
                pts_all = np.vstack((pts_a, pts_b))
                max_range = np.array([pts_all[:, 0].max() - pts_all[:, 0].min(),
                                      pts_all[:, 1].max() - pts_all[:, 1].min(),
                                      pts_all[:, 2].max() - pts_all[:, 2].min()]).max() / 2.0
                mid_x = (pts_all[:, 0].max() + pts_all[:, 0].min()) * 0.5
                mid_y = (pts_all[:, 1].max() + pts_all[:, 1].min()) * 0.5
                mid_z = (pts_all[:, 2].max() + pts_all[:, 2].min()) * 0.5
                ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

                ax_3d.set_xlabel('X (mm)')
                ax_3d.set_ylabel('Y (mm)')
                ax_3d.set_zlabel('Z (mm)')
                ax_3d.set_title(f'[{stage_name}] 3D Rotation Axes\nAngle Dev: {angle_error:.3f}° | Center Dist: {center_dist:.2f}mm')
                ax_3d.legend()

            plot_column(first_res, 0, "BEFORE")
            plot_column(final_res, 1, "AFTER")

            fig.suptitle(
                f"Joint Calibration Comparison: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_res['angle_between_normals']:.3f}°, Center Dist = {first_res['center_dist']:.2f} mm\n"
                f"After : Angle Dev = {final_res['angle_between_normals']:.3f}°, Center Dist = {final_res['center_dist']:.2f} mm",
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            result_dir = os.path.join(os.path.dirname(__file__), "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"circle_fit_{arm_side}_{mode}_joint_calib.png"))
            plt.savefig(plot_save_path, dpi=150)
            plt.close()

            if log_callback:
                log_callback(f"[SUCCESS] Saved combined calibration comparison plot to: {plot_save_path}")
            return plot_save_path
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to generate comparison plot: {e}")
            import traceback
            if log_callback:
                log_callback(traceback.format_exc())
            return None

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        import threading
        
        class MoveThread(threading.Thread):
            def __init__(self, calibrator, robot, torso, right_arm, left_arm, head, minimum_time):
                super().__init__()
                self.calibrator = calibrator
                self.robot = robot
                self.torso = torso
                self.right_arm = right_arm
                self.left_arm = left_arm
                self.head = head
                self.minimum_time = minimum_time
                self.success = False

            def run(self):
                self.success = self.calibrator.movej(
                    self.robot, torso=self.torso, 
                    right_arm=self.right_arm, left_arm=self.left_arm, 
                    head=self.head, minimum_time=self.minimum_time,
                    apply_offsets=False
                )

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} CONTINUOUS OFFSET CALIBRATION SWEEP")
            if current_offset_deg != 0.0:
                log_callback(f"   [Baseline Shift (Current Applied Offset): {current_offset_deg:.4f}°]")
            log_callback("="*50)

        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system is not initialized.")
            return None
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return None

        # Pre-check marker visibility
        initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
        if not initial_check:
            if log_callback: log_callback("[ERROR] Marker is not visible.")
            if status_callback: status_callback(False)
            return None
        if status_callback: status_callback(True)

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

        # Define joint parameters based on mode
        if mode == "wrist_pitch":
            cand_joint = 5
            sweep_joint_A = 4
            sweep_joint_B = 6
        else: # elbow mode
            cand_joint = 3
            sweep_joint_A = 2
            sweep_joint_B = 4

        dyn_model = self.robot.get_dynamics()
        ee_name = f"ee_{arm_side}"

        # Arm cand baseline pose (shifted by current offset)
        active_offset = self.joint_offsets.get(mode, 0.0)
        nominal_joint_pos = initial_joint_pos[cand_joint] - np.radians(active_offset)
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = nominal_joint_pos + np.radians(current_offset_deg)

        # 1. PHYSICAL SWEEP JOINT A
        if log_callback: log_callback(f"\n--- [1/2] Commencing Continuous Sweep on Joint A (Index {sweep_joint_A}, duration={sweep_duration}s) ---")
        
        # Move to start position (-20 deg)
        q_start_A = list(q_cand)
        q_start_A[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(-20.0)
        if arm_side == "left":
            self.movej(self.robot, left_arm=q_start_A, head=None, minimum_time=2.0, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=q_start_A, head=None, minimum_time=2.0, apply_offsets=False)
        time.sleep(1.0)

        # Launch motion thread to move from -20 to +20 deg
        q_end_A = list(q_cand)
        q_end_A[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(20.0)
        
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_A if arm_side == "right" else None,
            left_arm=q_end_A if arm_side == "left" else None,
            head=None, minimum_time=sweep_duration
        )
        
        dataset_A = []
        move_thread.start()
        
        while move_thread.is_alive():
            res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    dataset_A.append((q_full_captured, pose))
            time.sleep(0.01) # 100Hz polling (consistent with Joint B)
            
        move_thread.join()
        if log_callback: log_callback(f"    -> Swept {len(dataset_A)} dense raw coordinate frames during Joint A motion.")

        # Return to baseline candidate pose
        if arm_side == "left":
            self.movej(self.robot, left_arm=q_cand, head=None, minimum_time=1.5, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=q_cand, head=None, minimum_time=1.5, apply_offsets=False)
        time.sleep(1.0)

        # 2. PHYSICAL SWEEP JOINT B
        if log_callback: log_callback(f"\n--- [2/2] Commencing Continuous Sweep on Joint B (Index {sweep_joint_B}, duration={sweep_duration}s) ---")
        
        # Move to start position (-20 deg)
        q_start_B = list(q_cand)
        q_start_B[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(-20.0)
        if arm_side == "left":
            self.movej(self.robot, left_arm=q_start_B, head=None, minimum_time=2.0, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=q_start_B, head=None, minimum_time=2.0, apply_offsets=False)
        time.sleep(1.0)

        # Launch motion thread to move from -20 to +20 deg
        q_end_B = list(q_cand)
        q_end_B[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(20.0)
        
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_B if arm_side == "right" else None,
            left_arm=q_end_B if arm_side == "left" else None,
            head=None, minimum_time=sweep_duration
        )
        
        dataset_B = []
        move_thread.start()
        
        while move_thread.is_alive():
            res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    dataset_B.append((q_full_captured, pose))
            time.sleep(0.01) # 30Hz polling
            
        move_thread.join()
        if log_callback: log_callback(f"    -> Swept {len(dataset_B)} dense raw coordinate frames during Joint B motion.")

        # Return arm to original ready pose (head=None)
        if log_callback: log_callback("\n[INFO] Sweep finished. Returning arm to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)

        if len(dataset_A) < 10 or len(dataset_B) < 10:
            if log_callback: log_callback("[ERROR] Too few valid captured points. Calibration failed.")
            return None

        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # Save FULL captured continuous sweep points to debug txt files before downsampling
        # (Commented out to stop saving txt files as requested)
        # self.save_debug_points(
        #     arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, 
        #     cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback
        # )
        
        # Keep up to 200 points for speed and accuracy
        raw_len_A = len(dataset_A)
        raw_len_B = len(dataset_B)
        
        max_pts = 200
        if len(dataset_A) > max_pts:
            indices_A = np.round(np.linspace(0, len(dataset_A) - 1, max_pts)).astype(int)
            dataset_A = [dataset_A[idx] for idx in indices_A]
        if len(dataset_B) > max_pts:
            indices_B = np.round(np.linspace(0, len(dataset_B) - 1, max_pts)).astype(int)
            dataset_B = [dataset_B[idx] for idx in indices_B]
            
        if log_callback:
            log_callback(f"Swept {raw_len_A} dense raw coordinate frames during Joint A motion... downsampled to {len(dataset_A)} for optimization.")
            log_callback(f"Swept {raw_len_B} dense raw coordinate frames during Joint B motion... downsampled to {len(dataset_B)} for optimization.")

        # 3. OFFLINE DIRECT ANGLE OPTIMIZATION (Fitted Circle Normal Orthogonality in Camera Frame)
        if log_callback: log_callback("\n--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---")
        
        poses_A = [pose for q_full, pose in dataset_A]
        angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, pose in dataset_A]
        
        poses_B = [pose for q_full, pose in dataset_B]
        angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, pose in dataset_B]
        
        # Calculate nominal axes in camera frame at nominal ready pose (without current_offset_deg)
        q_ready = np.array(dataset_A[0][0])
        active_offset = self.joint_offsets.get(mode, 0.0)
        q_ready[arm_idx[cand_joint]] = initial_joint_pos[cand_joint] - np.radians(active_offset)
        
        a_cand_local = np.array([0.0, 1.0, 0.0])
        a_A_local = np.array([0.0, 0.0, 1.0])
        if mode == "wrist_pitch":
            a_B_local = np.array([1.0, 0.0, 0.0])
        else:
            a_B_local = np.array([0.0, 0.0, 1.0])

        T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, "link_head_2", "link_torso_5")
        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
        
        parent_link_A = f"link_{arm_side}_arm_{sweep_joint_A}"
        parent_link_B = f"link_{arm_side}_arm_{sweep_joint_B}"
        
        T_t5_to_parent_A = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, parent_link_A, base_link="link_torso_5")
        T_t5_to_parent_B = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, parent_link_B, base_link="link_torso_5")
        
        # Transform local axes to torso frame
        a_cand_t5 = T_t5_to_parent_A[:3, :3] @ a_cand_local
        a_A_t5 = T_t5_to_parent_A[:3, :3] @ a_A_local
        a_B_t5 = T_t5_to_parent_B[:3, :3] @ a_B_local
        
        # Transform from torso frame to camera frame (R_torso_to_cam = T_t5_to_cam[:3, :3].T)
        R_t5_to_cam = T_t5_to_cam[:3, :3].T
        a_cand_cam = R_t5_to_cam @ a_cand_t5
        a_A_cam = R_t5_to_cam @ a_A_t5
        a_B_cam_nom = R_t5_to_cam @ a_B_t5
        
        a_cand_cam /= np.linalg.norm(a_cand_cam)
        a_A_cam /= np.linalg.norm(a_A_cam)
        a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)

        # Fit Sweep A rotation axis in camera frame using constrained circle fitting (using nominal axis as prior)
        res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
            poses_A, angles_A, axis_prior=a_A_cam
        )
        # Fit Sweep B rotation axis in camera frame using constrained circle fitting (using nominal axis as prior)
        res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
            poses_B, angles_B, axis_prior=a_B_cam_nom
        )
        
        n_A = res_A['axis_opt']
        n_B = res_B['axis_opt']
        
        # Enforce that normal vectors point in the same general direction
        if np.dot(n_A, n_B) < 0:
            n_B = -n_B
            
        r_A = res_A['radius']
        r_B = res_B['radius']
        rmse_A = res_A['rmse']
        rmse_B = res_B['rmse']
        pts_2d_A = res_A['pts_2d']
        pts_2d_B = res_B['pts_2d']
        uc_A = res_A['uc_opt']
        vc_A = res_A['vc_opt']
        uc_B = res_B['uc_opt']
        vc_B = res_B['vc_opt']
        
        # Compute center distance in camera frame
        try:
            pts_a_cam = [pose[:3, 3] * 1000.0 for _, pose in dataset_A]
            pts_b_cam = [pose[:3, 3] * 1000.0 for _, pose in dataset_B]
            
            c_A_c, R_c_A_c, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_a_cam)
            c_B_c, R_c_B_c, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_b_cam)
            n_A_c = R_c_A_c[:, 2]
            diff_centers = c_B_c - c_A_c
            center_dist = np.linalg.norm(diff_centers - np.dot(diff_centers, n_A_c) * n_A_c)
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to compute center distance in camera frame: {e}")
            center_dist = 999.0

        # Calculate angle between normals (the magnitude of the misalignment)
        angle_between_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))
        
        # Enforce that normal vectors point in the direction of the physical kinematic axes
        n_A = n_A if np.dot(n_A, a_A_cam) > 0 else -n_A
        n_B = n_B if np.dot(n_B, a_B_cam_nom) > 0 else -n_B
        
        # Define reference direction for B based on measured normal A and candidate axis
        ref_B = np.cross(a_cand_cam, n_A)
        if np.linalg.norm(ref_B) > 1e-6:
            ref_B /= np.linalg.norm(ref_B)
            if np.dot(ref_B, a_B_cam_nom) < 0:
                ref_B = -ref_B
        else:
            ref_B = a_B_cam_nom

        # Project the cross product of the reference direction and actual axis onto the candidate joint axis
        cross_val = np.cross(ref_B, n_B)
        proj = np.dot(cross_val, a_cand_cam)
        
        # If proj is positive, it means the physical angle is positive, so the required offset to correct it is negative.
        # If proj is negative, the physical angle is negative, so the required offset to correct it is positive.
        sign = -1.0 if proj > 0 else 1.0
        
        if log_callback:
            log_callback(f"  [DEBUG] Physically aligned n_A: {np.round(n_A, 4)}")
            log_callback(f"  [DEBUG] Physically aligned n_B: {np.round(n_B, 4)}")
            log_callback(f"  [DEBUG] Nominal a_B_cam_nom: {np.round(a_B_cam_nom, 4)}")
            log_callback(f"  [DEBUG] cross(a_B_cam_nom, n_B) projection onto a_cand_cam: proj={proj:.6f}")
            log_callback(f"  [DEBUG] Mathematically resolved offset direction sign: {sign:.1f}")
        
        # Proportional correction based on circle size error (radii difference) and center distance error
        size_error = abs(r_A - r_B)
        
        # Safeguard: check if circle fitting failed or is completely invalid
        if center_dist > 100.0 or size_error > 100.0:
            if log_callback:
                log_callback("[ERROR] Circle fitting failed or error is too large. Aborting step adjustment.")
            optimal_offset_deg = 0.0
        else:
            if (not use_angle_based_fitting) and (center_dist <= 1.0):
                optimal_offset_deg = sign * 0.05
                if log_callback:
                    log_callback(f"  [STEP CONTROL] Center distance {center_dist:.4f} mm is close (<= 1.0 mm). Applying 0.05° step correction.")
            else:
                error_metric = max(size_error, center_dist)
                optimal_offset_deg = sign * error_metric * 0.5
                
                # Maximum step size clamp to prevent excessive joint movements
                max_step_deg = 5.0
                if abs(optimal_offset_deg) > max_step_deg:
                    if log_callback:
                        log_callback(f"  [SAFETY CONTROL] Clamping calculated step {optimal_offset_deg:.4f}° to max limit ±{max_step_deg}°")
                    optimal_offset_deg = np.clip(optimal_offset_deg, -max_step_deg, max_step_deg)
                
        optimal_offset_rad = np.radians(optimal_offset_deg)

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()} - CONTINUOUS)")
            log_callback("="*50)
            log_callback(f"  * Camera Circle Normals Angle (Reference): {angle_between_normals:.4f} deg")
            log_callback(f"  * Circle Sizes (r_A / r_B)               : {r_A:.2f} / {r_B:.2f} mm")
            log_callback(f"  * Circle Size Error (abs: r_A - r_B)     : {size_error:.4f} mm")
            log_callback(f"  * Estimated Circle Center Distance       : {center_dist:.4f} mm")
            log_callback(f"  * Calculated Offset Correction           : {optimal_offset_deg:.6f} deg")
            log_callback("="*50)

        # Simultaneously generate and save orthogonal debug plot (using camera frame)
        self.save_debug_orthogonal_plot(
            arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam, 
            optimal_offset_rad, ee_name, arm_idx, cand_joint, 
            angle_error_deg=angle_between_normals, log_callback=log_callback
        )

        # Simultaneous Marker Axis 6 parameter calculation
        marker_6_res = None
        if mode == "wrist_pitch":
            try:
                captured_poses_torso = []
                for q_full, pose_cam_to_marker in dataset_B:
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                    captured_poses_torso.append(T_t5_to_marker)
                    
                marker_6_res = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
                    captured_poses_torso, 
                    np.linspace(-20.0, 20.0, len(dataset_B)), 
                    axis_prior=[1.0, 0.0, 0.0]
                )
                if marker_6_res:
                    marker_6_res['axis'] = marker_6_res['axis_opt']
                
                if log_callback and marker_6_res:
                    log_callback("\n" + "-"*50)
                    log_callback("  [SIMULTANEOUS MARKER AXIS 6 ESTIMATION RESULTS (CONTINUOUS)]")
                    log_callback("-"*50)
                    log_callback(f"    - Fitted Sweep Radius    : {marker_6_res['radius']:.3f} mm")
                    # log_callback(f"    - Axis 6 Fitting RMSE    : {marker_6_res['rmse']:.3f} mm")
                    log_callback(f"    - Axis Direction Vector  : {np.round(marker_6_res['axis_opt'], 4)}")
                    log_callback(f"    - Jitter StdDev (Tilt)   : {np.std(marker_6_res.get('tilt_list', [0.0])):.3f} deg")
                    log_callback("-"*50 + "\n")
            except Exception as e:
                if log_callback:
                    log_callback(f"\n[WARN] Failed simultaneous Marker Axis 6 calculation: {e}\n")

        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
            'angle_between_normals': angle_between_normals,
            'sign': sign,
            'center_dist': center_dist,
            'pts_2d_A': pts_2d_A,
            'uc_A': uc_A,
            'vc_A': vc_A,
            'r_A': r_A,
            'pts_2d_B': pts_2d_B,
            'uc_B': uc_B,
            'vc_B': vc_B,
            'r_B': r_B,
            'rmse_A': rmse_A,
            'rmse_B': rmse_B,
            'marker_6_res': marker_6_res,
            'pts_a_cam': np.array(pts_a_cam),
            'pts_b_cam': np.array(pts_b_cam),
            'c_A': c_A_c,
            'c_B': c_B_c,
            'n_A': n_A,
            'n_B': n_B
        }
