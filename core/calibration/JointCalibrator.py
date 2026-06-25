import time
import logging
import os
import numpy as np
import rby1_sdk as rby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.use_angle_based_fitting = True

    def perform_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None, save_debug=False):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        if log_callback:
            log_callback("\n" + "="*60)
            log_callback("   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE")
            log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
            log_callback("="*60 + "\n")
            
        # save_debug는 첫 번째 sweep(원본 데이터)에서만 저장
        _sweep_count = [0]
        def run_single_sweep(offset):
            _sweep_count[0] += 1
            do_save = save_debug and (_sweep_count[0] == 1)
            return self.perform_calibration_sweep_continuous(
                arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                current_offset_deg=offset, sweep_duration=sweep_duration,
                use_angle_based_fitting=use_angle_based_fitting, save_debug=do_save
            )
            
        max_iterations = 8
        staged_offset = current_offset_deg
        final_res = None
        first_res = None
        converged = False
        
        for i in range(1, max_iterations + 1):
            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Joint calibration aborted due to stop request.")
                return None
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
            
            is_v13_mode = mode in ("wrist_roll_v13", "wrist_pitch_v13")
            if is_v13_mode:
                center_dist = res.get('perp_dist_after', 999.0)
                angle_dev = abs(angle_error - 90.0)
            else:
                center_dist = res.get('center_dist', 999.0)
                angle_dev = angle_error
                
            r_A = res.get('r_A', 0.0)
            r_B = res.get('r_B', 0.0)
            size_error = abs(r_A - r_B)
            current_error = max(size_error, center_dist)
            
            # Print iteration summary
            if log_callback:
                if is_v13_mode:
                    log_callback(f"  * Angle Error (Deviation from 90°) : {angle_dev:.4f}°")
                    log_callback(f"  * Perpendicular Distance (After)   : {center_dist:.4f} mm")
                    log_callback(f"  * Perpendicular Distance (Before)  : {res.get('perp_dist_before', 999.0):.4f} mm")
                else:
                    log_callback(f"  * Angle Error (Deviation)     : {angle_error:.4f}°")
                    if mode == "wrist_pitch_v13":
                        log_callback(f"  * Forearm Length (Center Dist): {center_dist:.4f} mm")
                        log_callback(f"  * Radii Difference (r3 - r5)  : {size_error:.4f} mm")
                    else:
                        log_callback(f"  * Circle Size Error (r_A-r_B) : {size_error:.4f} mm")
                        log_callback(f"  * Center Distance Error       : {center_dist:.4f} mm")
                        log_callback(f"  * Max Fitting Error Metric    : {current_error:.4f} mm")
            
            # Direct correction
            # When looking at angle error (use_angle_based_fitting is True), we deactivate the 0.05 deg step correction.
            # Otherwise (when use_angle_based_fitting is False), if circle center distance is close (<= 1.0 mm),
            # we apply a 0.05 deg step correction instead of the raw angle_error.
            # Use the pre-calculated damped optimal offset correction to ensure convergence
            step_correction = res.get('optimal_offset', 0.0)
            staged_offset += step_correction
            final_res = res
            # Elbow Safety Check: Elbow offset must unconditionally be negative.
            if mode == "elbow":
                if i == 1:
                    if log_callback:
                        log_callback(f"  [SAFETY CONTROL] Elbow joint offset must unconditionally be negative. Forcing first staged offset {staged_offset:.4f}° to {-abs(staged_offset):.4f}° for safety!")
                    staged_offset = -abs(staged_offset)
                else:
                    if staged_offset > 0.0:
                        if log_callback:
                            log_callback(f"  [SAFETY CONTROL] Elbow joint offset must unconditionally be negative. Clipping positive staged offset {staged_offset:.4f}° to 0.0° for safety!")
                        staged_offset = 0.0
                    staged_offset = min(staged_offset, 0.0)
            
            # General range safety limits (0 to -3.0 degrees for elbow)
            if mode == "elbow":
                min_val, max_val = -3.0, 0.0
                if staged_offset < min_val or staged_offset > max_val:
                    if log_callback:
                        log_callback(f"  [SAFETY WARNING] Staged offset {staged_offset:.4f}° exceeds safe bounds [{min_val}°, {max_val}°]!")
                        log_callback(f"                   Possible fitting failure, loose bracket, or joint twist. Clamping to boundary.")
                    staged_offset = np.clip(staged_offset, min_val, max_val)
                
            if log_callback:
                log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                
            # Convergence check:
            # - For v1.3: step correction < 0.05° or angle deviation <= 0.5° or perp_dist_after <= 0.1 mm
            # - For others: step correction < 0.05° or angle error <= 0.1° or center_dist <= 0.1 mm
            if is_v13_mode:
                converged_criteria = (abs(step_correction) < 0.05 or angle_dev <= 0.5 or center_dist <= 0.1)
            else:
                converged_criteria = (abs(step_correction) < 0.05 or angle_dev <= 0.1 or center_dist <= 0.1)

            if converged_criteria:
                converged = True
                if log_callback:
                    log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                    if abs(step_correction) < 0.05:
                        log_callback(f"  * Step Correction: {step_correction:.4f}° < 0.05° (reached resolution limit)")
                    else:
                        if is_v13_mode:
                            log_callback(f"  * Circle Normals Angle Error Deviation: {angle_dev:.4f}° <= 0.5°")
                        else:
                            log_callback(f"  * Circle Normals Angle Error: {angle_dev:.4f}° <= 0.1°")
                        log_callback(f"  * Center Distance Error: {center_dist:.4f} mm <= 0.1 mm")
                    log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                break

        # General range safety limits on final recommended offset
        if mode == "elbow":
            min_val, max_val = -3.0, 0.0
            if staged_offset < min_val or staged_offset > max_val:
                if log_callback:
                    log_callback(f"  [SAFETY WARNING] Recommended final offset {staged_offset:.4f}° exceeds safe bounds [{min_val}°, {max_val}°]!")
                    log_callback(f"                   Clamping recommended offset to safe boundary.")
                staged_offset = np.clip(staged_offset, min_val, max_val)

        if getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[INFO] Joint calibration aborted before validation sweep.")
            return None

        # Build clean final output dict — UI only needs these fields
        # _plot_data is internal-only; strip it before returning to the UI
        final_output = {
            'mode': mode,
            'recommended_joint_offset': staged_offset,
            'optimal_offset': staged_offset,
            'converged': converged,
        }

        # Run one final validation sweep with the recommended joint offset
        if log_callback:
            log_callback(f"\n[VALIDATION SWEEP] Running final validation sweep with recommended offset {staged_offset:.4f}°...")
        validation_res = run_single_sweep(staged_offset)

        if validation_res:
            if first_res:
                plot_path = self.save_calibration_comparison_plot(arm_side, mode, first_res, validation_res, log_callback=log_callback)
                final_output['plot_path_combined'] = plot_path
        else:
            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Joint calibration aborted during validation sweep.")
                return None
            if log_callback:
                log_callback("[WARN] Validation sweep failed. Returning last calibration result.")
            if final_res and first_res:
                plot_path = self.save_calibration_comparison_plot(arm_side, mode, first_res, final_res, log_callback=log_callback)
                final_output['plot_path_combined'] = plot_path

        return final_output


    def perform_move_to_ready_pose(self, arm_side, mode, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving to {arm_side} Pitch/Head Ready Pose (Mode: {mode})...")
        torso = [0, 0, 0, 0, 0, 0]
        
        # 1. First move the inactive arm to zero pose to avoid collision
        if log_callback: log_callback("[INFO] Moving inactive arm to zero pose first...")
        if arm_side == "right":
            success_other = self.movej(self.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0,apply_offsets=False)
        else:
            success_other = self.movej(self.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0,apply_offsets=False)
            
        if not success_other:
            if log_callback: log_callback("[ERROR] Failed to move inactive arm to zero pose.")
            return False
            
        # 2. Move active arm and head/torso to ready pose
        if log_callback: log_callback("[INFO] Moving active arm, torso, and head to ready pose...")
        
        version_key = "v1.3" if self.is_v13() else "v1.2"
        
        ready_mode = "elbow" if mode == "elbow" else "wrist_pitch"
        
        if arm_side == "right":
            right_arm = self.get_ready_pose(version_key, "joint", ready_mode, "right")
            left_arm = None
        else:
            right_arm = None
            left_arm = self.get_ready_pose(version_key, "joint", ready_mode, "left")

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=[0, 0], minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def save_debug_orthogonal_plot(self, arm_side, frame, dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, angle_error_deg=None, log_callback=None):
        return
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

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            def plot_column(res, col_idx, stage_name):
                # Read from internal _plot_data bundle
                pd = res.get('_plot_data', {})
                pts_a      = pd.get('pts_a_cam')
                pts_b      = pd.get('pts_b_cam')
                c_A        = pd.get('c_A')
                c_B        = pd.get('c_B')
                n_A        = pd.get('n_A')
                n_B        = pd.get('n_B')
                r_A        = pd.get('r_A', res.get('r_A', 1.0))
                r_B        = pd.get('r_B', res.get('r_B', 1.0))
                angle_error  = pd.get('angle_between_normals', res.get('angle_between_normals', 0.0))
                center_dist  = pd.get('center_dist', res.get('center_dist', 0.0))

                if pts_a is None or c_A is None or n_A is None:
                    # No plot data available for this sweep — leave panel blank
                    for row in range(2):
                        axes[row, col_idx].set_title(f'[{stage_name}] No plot data')
                        axes[row, col_idx].axis('off')
                    return

                # Compute local frames algebraically from normals (Z axes)
                def get_local_vectors(n):
                    n = n / np.linalg.norm(n)
                    if abs(n[0]) < 0.9:
                        u = np.cross(n, [1, 0, 0])
                    else:
                        u = np.cross(n, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(n, u)
                    v = v / np.linalg.norm(v)
                    return u, v

                u_A, v_A = get_local_vectors(n_A)
                u_B, v_B = get_local_vectors(n_B)

                theta = np.linspace(0, 2 * np.pi, 200)
                circle_pts_a = c_A + r_A * (np.cos(theta)[:, None] * u_A + np.sin(theta)[:, None] * v_A)
                circle_pts_b = c_B + r_B * (np.cos(theta)[:, None] * u_B + np.sin(theta)[:, None] * v_B)

                # --- 1. TOP VIEW (Row 0, Col col_idx): X-Y Projection ---
                ax_top = axes[0, col_idx]
                ax_top.scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_top.scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_top.plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_top.plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_top.scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
                ax_top.scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
                ax_top.plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label='Center Shift')

                scale = min(r_A, r_B) * 0.4
                ax_top.arrow(c_A[0], c_A[1], n_A[0]*scale, n_A[1]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_top.arrow(c_B[0], c_B[1], n_B[0]*scale, n_B[1]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
                ax_top.set_xlabel('X (mm)')
                ax_top.set_ylabel('Y (mm)')
                ax_top.set_title(f'[{stage_name}] Top View (X-Y Projection)')
                ax_top.set_aspect('equal')
                ax_top.grid(True)
                ax_top.legend(loc='upper right')

                # --- 2. SIDE VIEW (Row 1, Col col_idx): Y-Z Projection ---
                ax_side = axes[1, col_idx]
                ax_side.scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_side.scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_side.plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_side.plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_side.scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
                ax_side.scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
                ax_side.plot([c_A[1], c_B[1]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2, label='Center Shift')
                ax_side.arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_side.arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
                ax_side.set_xlabel('Y (mm)')
                ax_side.set_ylabel('Z (mm)')
                ax_side.set_title(f'[{stage_name}] Side View (Y-Z Projection)\nAngle Dev: {angle_error:.3f}° | Center Dist: {center_dist:.2f}mm')
                ax_side.set_aspect('equal')
                ax_side.grid(True)
                ax_side.legend(loc='upper right')

            def compute_shortest_distance_between_lines(cA, nA, cB, nB):
                nA_norm = nA / np.linalg.norm(nA)
                nB_norm = nB / np.linalg.norm(nB)
                cross = np.cross(nA_norm, nB_norm)
                cross_norm = np.linalg.norm(cross)
                diff = cB - cA
                if cross_norm > 1e-4:
                    return abs(np.dot(diff, cross)) / cross_norm
                else:
                    return np.linalg.norm(diff - np.dot(diff, nA_norm) * nA_norm)

            nominal_dist_35 = None
            if mode == "wrist_pitch_v13" and self.robot:
                try:
                    dyn_model = self.robot.get_dynamics()
                    names = self.robot.model().robot_joint_names
                    state_3_5 = dyn_model.make_state(
                        [f"link_{arm_side}_arm_3", f"link_{arm_side}_arm_5"],
                        names
                    )
                    state_3_5.set_q(np.zeros(len(names)))
                    dyn_model.compute_forward_kinematics(state_3_5)
                    T_3_5 = dyn_model.compute_transformation(state_3_5, 0, 1)
                    nominal_dist_35 = np.linalg.norm(T_3_5[:3, 3]) * 1000.0
                except Exception:
                    pass

            plot_column(first_res, 0, "BEFORE")
            plot_column(final_res, 1, "AFTER")

            before_dist_str = ""
            after_dist_str = ""
            if mode == "wrist_pitch_v13":
                first_pd = first_res.get('_plot_data', {})
                final_pd = final_res.get('_plot_data', {})
                if all(k in first_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_before = compute_shortest_distance_between_lines(
                        first_pd['c_A'], first_pd['n_A'], first_pd['c_B'], first_pd['n_B']
                    )
                    before_dist_str = f" | Axis 3-5 Dist = {dist_before:.2f} mm"
                if all(k in final_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_after = compute_shortest_distance_between_lines(
                        final_pd['c_A'], final_pd['n_A'], final_pd['c_B'], final_pd['n_B']
                    )
                    after_dist_str = f" | Axis 3-5 Dist = {dist_after:.2f} mm"
                    if nominal_dist_35 is not None:
                        after_dist_str += f" (Nom: {nominal_dist_35:.2f} mm)"

            first_pd = first_res.get('_plot_data', first_res)
            final_pd = final_res.get('_plot_data', final_res)
            fig.suptitle(
                f"Joint Calibration: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {first_pd.get('center_dist', 0.0):.2f} mm{before_dist_str}\n"
                f"After : Angle Dev = {final_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {final_pd.get('center_dist', 0.0):.2f} mm{after_dist_str}",
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
                log_callback(f"[ERROR] Failed to save combined calibration comparison plot: {e}")
            import traceback
            if log_callback:
                log_callback(traceback.format_exc())
            return None

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None, save_debug=False):
        if getattr(self, 'stop_requested', False):
            return None

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

        if getattr(self, 'stop_requested', False):
            return None

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

        # Define joint parameters based on mode
        if mode == "wrist_roll_v13":
            cand_joint = 6
            sweep_joint_A = 6
            sweep_joint_B = 5
        elif mode == "wrist_pitch_v13":
            cand_joint = 5
            sweep_joint_A = 5
            sweep_joint_B = 3
        elif mode == "wrist_pitch":
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
        if mode == "wrist_roll_v13":
            offset_key = "wrist_roll"
        elif mode == "wrist_pitch_v13":
            offset_key = "wrist_pitch"
        else:
            offset_key = mode
        active_offset = self.joint_offsets.get(offset_key, 0.0)
        nominal_joint_pos = initial_joint_pos[cand_joint] - np.radians(active_offset)
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = nominal_joint_pos + np.radians(current_offset_deg)

        # 1. PHYSICAL SWEEP JOINT A
        if log_callback: log_callback(f"\n--- [1/2] Commencing Continuous Sweep on Joint A (Index {sweep_joint_A}, duration={sweep_duration}s) ---")
        
        if getattr(self, 'stop_requested', False):
            return None

        # Determine sweep ranges
        range_A = 20.0
        range_B = 20.0
        if mode == "wrist_pitch_v13":
            range_B = 10.0

        # Move to start position (-20 deg)
        q_start_A = list(q_cand)
        q_start_A[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(-range_A)
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=q_start_A, head=None, minimum_time=2.0, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start_A, head=None, minimum_time=2.0, apply_offsets=False)
            
        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to move to Joint A start pose or stop was requested.")
            return None
            
        time.sleep(1.0)

        # Launch motion thread to move from -20 to +20 deg
        q_end_A = list(q_cand)
        q_end_A[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(range_A)
        
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_A if arm_side == "right" else None,
            left_arm=q_end_A if arm_side == "left" else None,
            head=None, minimum_time=sweep_duration
        )
        
        dataset_A = []
        if self.robot and self.robot != "mock_robot":
            initial_full_pose_A = np.array(self.robot.get_state().position)
        else:
            initial_full_pose_A = np.zeros(20)

        t_start_A = time.time()
        move_thread.start()
        
        while move_thread.is_alive():
            if getattr(self, 'stop_requested', False):
                if self.robot and self.robot != "mock_robot":
                    self.robot.cancel_control()
                move_thread.join()
                return None
            res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                
                t_elapsed = time.time() - t_start_A
                ratio = min(1.0, max(0.0, t_elapsed / sweep_duration))
                q_full_captured = np.copy(initial_full_pose_A)
                global_joint_idx = arm_idx[sweep_joint_A]
                q_full_captured[global_joint_idx] = q_start_A[sweep_joint_A] + ratio * (q_end_A[sweep_joint_A] - q_start_A[sweep_joint_A])
                
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    dataset_A.append((q_full_captured, pose))
            time.sleep(0.01) # 100Hz polling (consistent with Joint B)
            
        move_thread.join()
        if not move_thread.success:
            if log_callback: log_callback("[ERROR] Joint A sweep motion failed or was cancelled.")
            return None
        if log_callback: log_callback(f"    -> Swept {len(dataset_A)} dense raw coordinate frames during Joint A motion.")

        if getattr(self, 'stop_requested', False):
            return None
            
        time.sleep(0.5)

        # 2. PHYSICAL SWEEP JOINT B
        if log_callback: log_callback(f"\n--- [2/2] Commencing Continuous Sweep on Joint B (Index {sweep_joint_B}, duration={sweep_duration}s) ---")
        
        if getattr(self, 'stop_requested', False):
            return None

        # Move to start position (-20 deg or -10 deg)
        q_start_B = list(q_cand)
        q_start_B[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(-range_B)
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=q_start_B, head=None, minimum_time=2.0, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start_B, head=None, minimum_time=2.0, apply_offsets=False)
            
        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to move to Joint B start pose or stop was requested.")
            return None
            
        time.sleep(1.0)

        # Launch motion thread to move from -20 to +20 deg (or -10 to +10 deg)
        q_end_B = list(q_cand)
        q_end_B[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(range_B)
        
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_B if arm_side == "right" else None,
            left_arm=q_end_B if arm_side == "left" else None,
            head=None, minimum_time=sweep_duration
        )
        
        dataset_B = []
        if self.robot and self.robot != "mock_robot":
            initial_full_pose_B = np.array(self.robot.get_state().position)
        else:
            initial_full_pose_B = np.zeros(20)

        t_start_B = time.time()
        move_thread.start()
        
        while move_thread.is_alive():
            if getattr(self, 'stop_requested', False):
                if self.robot and self.robot != "mock_robot":
                    self.robot.cancel_control()
                move_thread.join()
                return None
            res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                
                t_elapsed = time.time() - t_start_B
                ratio = min(1.0, max(0.0, t_elapsed / sweep_duration))
                q_full_captured = np.copy(initial_full_pose_B)
                global_joint_idx = arm_idx[sweep_joint_B]
                q_full_captured[global_joint_idx] = q_start_B[sweep_joint_B] + ratio * (q_end_B[sweep_joint_B] - q_start_B[sweep_joint_B])
                
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    dataset_B.append((q_full_captured, pose))
            time.sleep(0.01) # 30Hz polling
            
        move_thread.join()
        if not move_thread.success:
            if log_callback: log_callback("[ERROR] Joint B sweep motion failed or was cancelled.")
            return None
        if log_callback: log_callback(f"    -> Swept {len(dataset_B)} dense raw coordinate frames during Joint B motion.")

        if getattr(self, 'stop_requested', False):
            return None

        # Return arm to original ready pose (head=None)
        if log_callback: log_callback("\n[INFO] Sweep finished. Returning arm to initial pose...")
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to return arm to initial pose or stop was requested.")
            return None

        if len(dataset_A) < 10 or len(dataset_B) < 10:
            if log_callback: log_callback("[ERROR] Too few valid captured points. Calibration failed.")
            return None

        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # Save FULL captured continuous sweep points to debug txt files before downsampling
        if save_debug:
            self.save_debug_points(
                arm_side, sweep_joint_A, dataset_A, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, "joint_A", log_callback
            )
            self.save_debug_points(
                arm_side, sweep_joint_B, dataset_B, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, "joint_B", log_callback
            )
        
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

        return self.compute_calibration_results(
            arm_side=arm_side,
            mode=mode,
            dataset_A=dataset_A,
            dataset_B=dataset_B,
            initial_joint_pos=initial_joint_pos,
            current_offset_deg=current_offset_deg,
            use_angle_based_fitting=use_angle_based_fitting,
            save_debug=save_debug,
            log_callback=log_callback
        )

    def compute_calibration_results(self, arm_side, mode, dataset_A, dataset_B, initial_joint_pos, current_offset_deg=0.0, use_angle_based_fitting=None, save_debug=False, log_callback=None):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected or available for calibration computation.")
            return None

        # 3. OFFLINE DIRECT ANGLE OPTIMIZATION (Fitted Circle Normal Orthogonality in Camera Frame)
        if log_callback: log_callback("\n--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---")
        
        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx

        # Define joint parameters based on mode
        if mode == "wrist_roll_v13":
            cand_joint = 6
            sweep_joint_A = 6
            sweep_joint_B = 5
        elif mode == "wrist_pitch_v13":
            cand_joint = 5
            sweep_joint_A = 5
            sweep_joint_B = 3
        elif mode == "wrist_pitch":
            cand_joint = 5
            sweep_joint_A = 4
            sweep_joint_B = 6
        else: # elbow mode
            cand_joint = 3
            sweep_joint_A = 2
            sweep_joint_B = 4

        dyn_model = self.robot.get_dynamics()
        ee_name = f"ee_{arm_side}"

        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        if mode in ["wrist_roll_v13", "wrist_pitch_v13"]:
            # Load Tf_to_marker
            version_suffix = "_v13" if self.is_v13() else "_v12"
            tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
            tf_vec = self.camera_config.get(tf_key)
            if tf_vec is None:
                tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            if tf_vec is None:
                if arm_side == "left":
                    tf_vec = [0.0, 0.0775, -0.06677, 90.0, 0.0, 0.0]
                else:
                    tf_vec = [0.0, -0.0775, -0.06677, 90.0, 0.0, 180.0]
            T_ee_to_marker = self.make_transform(tf_vec)            # Fit circles to measured points in torso frame
            poses_a_t5 = []
            for q_full, pose in dataset_A:
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                poses_a_t5.append(T_t5_to_cam @ pose)
            
            poses_b_t5 = []
            for q_full, pose in dataset_B:
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                poses_b_t5.append(T_t5_to_cam @ pose)

            angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, _ in dataset_A]
            angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, _ in dataset_B]

            # Fit Sweep A rotation axis in torso frame
            # For wrist_roll_v13: Sweep A is Joint 6 (Roll, rotates about Z axis)
            # For wrist_pitch_v13: Sweep A is Joint 5 (Pitch, rotates about Y axis)
            a_A_prior = np.array([0.0, 0.0, 1.0]) if mode == "wrist_roll_v13" else np.array([0.0, 1.0, 0.0])
            res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
                poses_a_t5, angles_A, axis_prior=a_A_prior
            )
            c_A = res_A['c_opt']
            n_A = res_A['axis_opt']
            r_A = res_A['radius']
            rmse_A = res_A['rmse']
            pts_2d_A = res_A['pts_2d']
            uc_A = res_A['uc_opt']
            vc_A = res_A['vc_opt']

            # Fit Sweep B rotation axis in torso frame
            # For wrist_roll_v13: Sweep B is Joint 5 (Pitch, rotates about Y axis)
            # For wrist_pitch_v13: Sweep B is Joint 3 (Elbow, rotates about Y axis)
            a_B_prior = np.array([0.0, 1.0, 0.0])
            res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
                poses_b_t5, angles_B, axis_prior=a_B_prior
            )
            c_B = res_B['c_opt']
            n_B = res_B['axis_opt']
            r_B = res_B['radius']
            rmse_B = res_B['rmse']
            pts_2d_B = res_B['pts_2d']
            uc_B = res_B['uc_opt']
            vc_B = res_B['vc_opt']

            angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
            diff_centers = c_B - c_A
            n_A_norm = n_A / np.linalg.norm(n_A)
            # Perpendicular distance from c_B to the Joint-A rotation axis (line through c_A, dir n_A_norm)
            axial_offset   = float(np.dot(diff_centers, n_A_norm))           # mm, along Joint-A axis
            lateral_offset = float(np.linalg.norm(                           # mm, perpendicular to axis
                diff_centers - axial_offset * n_A_norm))

            if mode == "wrist_roll_v13":
                # Sweep A = Joint 6 (Wrist Roll) → defines axis (c_A, n_A_norm)
                # Sweep B = Joint 5 (Wrist Pitch) → c_B must lie ON the Joint 6 axis
                #
                # Calibration criterion:
                #   Perp distance from c_B_predicted to the Joint-6 axis = 0
                #   i.e. find delta_6 s.t. c_B_pred lies on (c_A, n_A_norm)
                #
                # After calibration:
                #   r_A            = lateral marker offset from Joint 6 axis (bracket design check)
                #   axial_offset   = marker offset along Joint 6 axis        (bracket design check)

                def perp_dist_axis6(delta_deg):
                    """FK-predict c_B when Joint 6 is shifted by delta_deg,
                    return its perpendicular distance to the Joint-6 axis."""
                    pts_pred = []
                    for q_full, _ in dataset_B:
                        q_mod = np.array(q_full)
                        q_mod[arm_idx[6]] += np.radians(delta_deg)          # perturb Joint 6
                        T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                        T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker
                        pts_pred.append(T_t5_to_marker[:3, 3] * 1000.0)
                    c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_pred, robust=False)
                    v = c_fit - c_A
                    return float(np.linalg.norm(v - np.dot(v, n_A_norm) * n_A_norm))

                perp_before = perp_dist_axis6(0.0)
                res_opt = least_squares(
                    lambda delta: [perp_dist_axis6(delta[0])],
                    [0.0], bounds=([-15.0], [15.0])
                )
                optimal_offset_deg = +res_opt.x[0]
                perp_after = perp_dist_axis6(res_opt.x[0])

                if log_callback:
                    log_callback(f"  [v1.3 Joint 6 Calibration]")
                    log_callback(f"    perp_dist(δ=0)   = {perp_before:.4f} mm  (c_B ~ Joint6 axis, before)")
                    log_callback(f"    solved δ         = {res_opt.x[0]:.4f}°  →  applied offset = {optimal_offset_deg:.4f}°")
                    log_callback(f"    perp_dist(δ_opt) = {perp_after:.4f} mm  (after)")
                    log_callback(f"  [Bracket Design Verification]")
                    log_callback(f"    r_A  (Joint6 sweep radius, lateral offset from axis) = {r_A:.3f} mm")
                    log_callback(f"    axial offset (c_B along Joint6 axis)                 = {axial_offset:.3f} mm")

            else:  # wrist_pitch_v13
                # Sweep A = Joint 5 (WP)  → c_A, n_A_norm = J5 axis, r_A = radial marker offset
                # Sweep B = Joint 3 (Elbow) → c_B (cross-check only)
                #
                # DIRECT VECTOR-BASED J5 OFFSET COMPUTATION
                # ─────────────────────────────────────────────────────────────
                # Key insight (user formula):  vec(J5→Marker) = vec(c_A→p_marker)_perp
                #
                # At q5_commanded (= staged_offset), compare:
                #   v_fk  = FK-predicted marker vector in J5 perp-plane
                #   v_cam = camera-measured marker vector in J5 perp-plane
                #
                # signed_angle(v_fk → v_cam) = J5 zero offset (direct, no iteration)
                #
                # ALSO yields marker design values:
                #   r_A      = |v_cam_perp| (radial offset from J5 axis)
                #   axial    = (p_cam - c_A) · n_J5 (axial offset along J5)
                #   mounting = direction of v_cam_perp (bracket phase angle)
                #
                # Cross-check: minimize_scalar on center_match (J3 sweep center shift)
                # Expected sensitivity ~5.4mm/°. If scan is U-shaped → FK consistent.

                # ── 1. Reference frames at q5 ≈ q_cand (the sweep midpoint) ──────────
                q_ref = q_cand[cand_joint]   # commanded q5 during sweep (arm-local index)
                ref_window = np.radians(2.0)

                ref_frames = [(q, p) for q, p in dataset_A
                              if abs(q[arm_idx[cand_joint]] - q_ref) <= ref_window]
                if not ref_frames:
                    # fallback: closest single frame
                    idx_closest = int(np.argmin(
                        [abs(q[arm_idx[cand_joint]] - q_ref) for q, _ in dataset_A]))
                    ref_frames = [dataset_A[idx_closest]]

                # ── 2. Compute signed J5 offset for each reference frame ──────────────
                delta_list  = []
                r_A_list    = []
                axial_list  = []

                for q_f, pose_f in ref_frames:
                    # Camera-measured marker position (torso frame, mm)
                    p_cam = pose_f[:3, 3] * 1000.0

                    # FK-predicted marker position (torso frame, mm)
                    T_fk  = BaseCalibrator.compute_fk(self.robot, dyn_model, q_f, ee_name)
                    p_fk  = (T_fk @ T_ee_to_marker)[:3, 3] * 1000.0

                    # Project to J5 perpendicular plane (relative to c_A)
                    def _perp(p):
                        v = p - c_A
                        return v - np.dot(v, n_A_norm) * n_A_norm

                    v_cam_perp = _perp(p_cam)
                    v_fk_perp  = _perp(p_fk)

                    # Guard: skip if either vector is near-zero (shouldn't happen, r_A~161mm)
                    if np.linalg.norm(v_cam_perp) < 1.0 or np.linalg.norm(v_fk_perp) < 1.0:
                        continue

                    # Signed angle from FK-expected to camera-measured = J5 zero offset
                    cross = np.dot(np.cross(v_fk_perp, v_cam_perp), n_A_norm)
                    dot   = np.dot(v_fk_perp, v_cam_perp)
                    delta_list.append(np.degrees(np.arctan2(cross, dot)))
                    r_A_list.append(np.linalg.norm(v_cam_perp))
                    axial_list.append(float(np.dot(p_cam - c_A, n_A_norm)))

                # Robust aggregation (median over reference window)
                if delta_list:
                    delta_direct = float(np.median(delta_list))
                    r_A_direct   = float(np.median(r_A_list))
                    axial_direct = float(np.median(axial_list))
                    direct_ok    = True
                else:
                    delta_direct = 0.0
                    r_A_direct   = r_A
                    axial_direct = axial_offset
                    direct_ok    = False

                # ── 3. Cross-check: center-match scan on Sweep B (J3 circle center) ──
                c_J3_camera = c_B

                def center_match_residual(delta_deg):
                    pts_pred = []
                    for q_full, _ in dataset_B:
                        q_mod = np.array(q_full)
                        q_mod[arm_idx[5]] += np.radians(delta_deg)
                        T_ee  = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                        pts_pred.append((T_ee @ T_ee_to_marker)[:3, 3] * 1000.0)
                    c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_pred, robust=False)
                    return float(np.linalg.norm(c_fit - c_J3_camera))

                scan_pts = [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0]
                scan_vals = [(d, center_match_residual(d)) for d in scan_pts]
                best_scan  = min(scan_vals, key=lambda x: x[1])
                center_match_at0 = scan_vals[4][1]

                if log_callback:
                    log_callback("  [J5 center-match cross-check scan]")
                    for d, v in scan_vals:
                        tag = " <-min" if d == best_scan[0] else ""
                        log_callback(f"    delta={d:+6.1f}deg -> |c_FK-c_cam| = {v:.3f} mm{tag}")

                # ── 4. Primary result: direct vector method ──────────────────────────
                optimal_offset_deg = delta_direct if direct_ok else best_scan[0]
                converged_wp = direct_ok and (abs(delta_direct) < 0.2)

                # perp_before/after kept for UI compatibility (repurposed as center-match dist)
                perp_before = center_match_at0
                perp_after  = center_match_residual(optimal_offset_deg) if direct_ok else best_scan[1]

                if log_callback:
                    log_callback(f"  [v1.3 Joint 5 Calibration  — DIRECT vector method]")
                    log_callback(f"    reference frames used        : {len(ref_frames)} (±2° window)")
                    log_callback(f"    J5 offset (direct, primary)  : {delta_direct:.4f}deg")
                    log_callback(f"    J5 offset (scan minimum)     : {best_scan[0]:.1f}deg  (dist={best_scan[1]:.3f}mm)")
                    log_callback(f"  [Marker Design Values from Sweep A reference frame]")
                    log_callback(f"    r_A (sweep-fit)              : {r_A:.3f} mm")
                    log_callback(f"    r_A (direct, vec|cam-perp|)  : {r_A_direct:.3f} mm")
                    log_callback(f"    axial offset along J5 axis   : {axial_direct:.3f} mm  (sweep B: {axial_offset:.3f} mm)")

            sign = -1.0 if optimal_offset_deg < 0.0 else 1.0

            if save_debug:
                self.save_debug_orthogonal_plot(
                    arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam,
                    np.radians(optimal_offset_deg), ee_name, arm_idx, cand_joint,
                    angle_error_deg=angle_between_normals, log_callback=log_callback
                )

            return {
                'mode': mode,
                'optimal_offset': optimal_offset_deg,
                'converged': converged_wp if mode == "wrist_pitch_v13" else False,
                'angle_between_normals': angle_between_normals,
                'sign': sign,
                'perp_dist_before': perp_before,
                'perp_dist_after': perp_after,
                'axial_offset_mm': axial_offset,
                'lateral_offset_mm': lateral_offset,
                'r_A': r_A,
                'r_B': r_B,
                # plot 전용 내부 필드 (UI final_output에는 포함되지 않음)
                '_plot_data': {
                    'pts_a_cam': np.array([p[:3, 3] * 1000.0 for p in poses_a_t5]),
                    'pts_b_cam': np.array([p[:3, 3] * 1000.0 for p in poses_b_t5]),
                    'c_A': c_A, 'c_B': c_B,
                    'n_A': n_A, 'n_B': n_B,
                    'r_A': r_A, 'r_B': r_B,
                    'angle_between_normals': angle_between_normals,
                    'center_dist': lateral_offset,
                },
            }

        poses_A = [pose for q_full, pose in dataset_A]
        angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, pose in dataset_A]
        
        poses_B = [pose for q_full, pose in dataset_B]
        angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, pose in dataset_B]
        
        # Calculate nominal axes in camera frame at nominal ready pose (without current_offset_deg)
        q_ready = np.array(dataset_A[0][0])
        active_offset = self.joint_offsets.get(mode, 0.0)
        q_ready[arm_idx[cand_joint]] = initial_joint_pos[cand_joint] - np.radians(active_offset)
        
        # Define fixed camera-to-robot rotation relationship (ZYX Euler: [-90.0, 0.0, -90.0])
        # R_rob_to_cam represents the rotation from robot torso (base) to camera frame
        R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()

        # Calculate nominal axes dynamically using robot kinematics (Dynamics model FK)
        # to account for intermediate joint rotations in the ready pose.
        try:
            if self.robot and self.robot != "mock_robot":
                # Determine parent links for candidate and sweep joints in URDF structure
                parent_cand = f"link_torso_5" if cand_joint == 0 else f"link_{arm_side}_arm_{cand_joint - 1}"
                parent_A = f"link_torso_5" if sweep_joint_A == 0 else f"link_{arm_side}_arm_{sweep_joint_A - 1}"
                parent_B = f"link_torso_5" if sweep_joint_B == 0 else f"link_{arm_side}_arm_{sweep_joint_B - 1}"

                # Compute FK relative to link_torso_5
                T_t5_to_parent_cand = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, parent_cand, "link_torso_5")
                T_t5_to_parent_A = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, parent_A, "link_torso_5")
                T_t5_to_parent_B = BaseCalibrator.compute_fk(self.robot, dyn_model, q_ready, parent_B, "link_torso_5")

                R_t5_to_parent_cand = T_t5_to_parent_cand[:3, :3]
                R_t5_to_parent_A = T_t5_to_parent_A[:3, :3]
                R_t5_to_parent_B = T_t5_to_parent_B[:3, :3]

                # Rotate nominal axes from local parent link frames to torso (link_torso_5) frame
                a_cand_t5 = R_t5_to_parent_cand @ np.array([0.0, 1.0, 0.0])
                a_A_t5 = R_t5_to_parent_A @ np.array([0.0, 0.0, 1.0])
                a_B_t5 = R_t5_to_parent_B @ np.array([0.0, 0.0, 1.0])
            else:
                a_cand_t5 = np.array([0.0, 1.0, 0.0])
                a_A_t5 = np.array([0.0, 0.0, 1.0])
                a_B_t5 = np.array([0.0, 0.0, 1.0])
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to compute nominal axes via FK: {e}. Falling back to hardcoded torso axes.")
            a_cand_t5 = np.array([0.0, 1.0, 0.0])
            a_A_t5 = np.array([0.0, 0.0, 1.0])
            a_B_t5 = np.array([0.0, 0.0, 1.0])

        # Define nominal axes in the camera frame using transpose of R_rob_to_cam (since R_rob_to_cam is R_cam_to_torso)
        a_cand_cam = R_rob_to_cam.T @ a_cand_t5
        a_A_cam = R_rob_to_cam.T @ a_A_t5
        a_B_cam_nom = R_rob_to_cam.T @ a_B_t5

        # Define T_t5_to_cam using fixed rotation and zero translation (strictly camera fixed, no FK)
        T_t5_to_cam = np.eye(4)
        T_t5_to_cam[:3, :3] = R_rob_to_cam
        
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
        
        pts_a_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_A])
        pts_b_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_B])
        
        # Compute center distance in camera frame using direct results from fit_circle_3d_and_6dof_misalignment
        try:
            c_A_c = res_A['c_opt']
            c_B_c = res_B['c_opt']
            n_A_c = res_A['axis_opt']
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
        
        # Project nominal and actual axes onto the plane perpendicular to the candidate joint axis
        a_A_proj = a_A_cam - np.dot(a_A_cam, a_cand_cam) * a_cand_cam
        a_B_proj = a_B_cam_nom - np.dot(a_B_cam_nom, a_cand_cam) * a_cand_cam
        if np.linalg.norm(a_A_proj) > 1e-6:
            a_A_proj /= np.linalg.norm(a_A_proj)
        if np.linalg.norm(a_B_proj) > 1e-6:
            a_B_proj /= np.linalg.norm(a_B_proj)
            
        nominal_angle = np.arctan2(np.dot(np.cross(a_A_proj, a_B_proj), a_cand_cam), np.dot(a_A_proj, a_B_proj))
        
        n_A_proj = n_A - np.dot(n_A, a_cand_cam) * a_cand_cam
        n_B_proj = n_B - np.dot(n_B, a_cand_cam) * a_cand_cam
        if np.linalg.norm(n_A_proj) > 1e-6:
            n_A_proj /= np.linalg.norm(n_A_proj)
        if np.linalg.norm(n_B_proj) > 1e-6:
            n_B_proj /= np.linalg.norm(n_B_proj)
            
        actual_angle = np.arctan2(np.dot(np.cross(n_A_proj, n_B_proj), a_cand_cam), np.dot(n_A_proj, n_B_proj))
        
        diff_angle = actual_angle - nominal_angle
        # Wrap diff_angle to [-pi, pi]
        diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Match the physical motor driver rotations (negative feedback loop)
        sign = -1.0 if diff_angle > 0.0 else 1.0
        
        if log_callback:
            log_callback(f"  [DEBUG] Physically aligned n_A: {np.round(n_A, 4)}")
            log_callback(f"  [DEBUG] Physically aligned n_B: {np.round(n_B, 4)}")
            log_callback(f"  [DEBUG] Nominal angles in perp plane (deg): nominal={np.degrees(nominal_angle):.4f}, actual={np.degrees(actual_angle):.4f}")
            log_callback(f"  [DEBUG] Perpendicular plane angle difference (deg): diff_angle={np.degrees(diff_angle):.4f}")
            log_callback(f"  [DEBUG] Mathematically resolved offset direction sign: {sign:.1f}")
        
        # Proportional correction based on circle size error (radii difference) and center distance error
        size_error = abs(r_A - r_B)
        
        # Safeguard: check if circle fitting failed or is completely invalid
        if center_dist > 100.0 or size_error > 100.0:
            if log_callback:
                log_callback("[ERROR] Circle fitting failed or error is too large. Aborting step adjustment.")
            optimal_offset_deg = 0.0
        else:
            if use_angle_based_fitting:
                optimal_offset_deg = sign * abs(np.degrees(diff_angle)) * 0.95
                if log_callback:
                    log_callback(f"  [ANGLE CONTROL] Using angle-based calibration error: {np.degrees(diff_angle):.4f}°. Applying damped correction step: {optimal_offset_deg:.4f}°")
            elif center_dist <= 1.0:
                optimal_offset_deg = sign * 0.5
                if log_callback:
                    log_callback(f"  [STEP CONTROL] Center distance {center_dist:.4f} mm is close (<= 1.0 mm). Applying 0.05° step correction.")
            else:
                error_metric = max(size_error, center_dist)
                optimal_offset_deg = sign * error_metric * 0.5
                
            # Maximum step size clamp to prevent excessive joint movements
            max_step_deg = 20.0
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
        if save_debug:
            self.save_debug_orthogonal_plot(
                arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam, 
                optimal_offset_rad, ee_name, arm_idx, cand_joint, 
                angle_error_deg=angle_between_normals, log_callback=log_callback
            )



        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
            'converged': False,
            'angle_between_normals': angle_between_normals,
            'sign': sign,
            'center_dist': center_dist,
            'r_A': r_A,
            'r_B': r_B,
            # plot 전용 내부 필드 (UI final_output에는 포함되지 않음)
            '_plot_data': {
                'pts_a_cam': pts_a_cam,
                'pts_b_cam': pts_b_cam,
                'c_A': c_A_c,
                'c_B': c_B_c,
                'n_A': n_A,
                'n_B': n_B,
                'r_A': r_A,
                'r_B': r_B,
                'angle_between_normals': angle_between_normals,
                'center_dist': center_dist,
            },
        }
