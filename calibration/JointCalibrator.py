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
    def perform_3step_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=60.0):
        if log_callback:
            log_callback("\n" + "="*60)
            log_callback("   STARTING MULTI-STEP JOINT CALIBRATION SEQUENCE")
            log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
            log_callback("="*60 + "\n")
            
        def run_single_sweep(offset):
            return self.perform_calibration_sweep_continuous(
                arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                current_offset_deg=offset, sweep_duration=sweep_duration
            )
                
        # --- STEP 1: Initial Sweep ---
        if log_callback:
            log_callback(f"[STEP 1/3] Executing initial sweep with baseline offset {current_offset_deg:.4f}°...")
        res_1 = run_single_sweep(current_offset_deg)
        if not res_1:
            if log_callback: log_callback("[ERROR] Step 1 sweep failed. Aborting 3-step calibration.")
            return None
            
        optimal_offset_1 = res_1['optimal_offset']
        if log_callback:
            log_callback(f"\n[STEP 1 COMPLETE] Initial relative offset: {optimal_offset_1:.4f}° calculated.")
            
        # --- STEP 2: Polarity / Direction Verification ---
        temp_offset_2 = current_offset_deg + optimal_offset_1
        if log_callback:
            log_callback(f"\n[STEP 2/3] Verifying polarity. Shifting joint angle by {temp_offset_2:.4f}°...")
            
        res_2 = run_single_sweep(temp_offset_2)
        if not res_2:
            if log_callback: log_callback("[ERROR] Step 2 verification sweep failed. Aborting.")
            return None
            
        optimal_offset_2 = res_2['optimal_offset']
        
        # Check convergence: has the relative offset error decreased to less than 70% of initial?
        if abs(optimal_offset_2) < abs(optimal_offset_1) * 0.7:
            # Polarity is correct!
            recommended = temp_offset_2 + optimal_offset_2
            if log_callback:
                log_callback(f"\n[STEP 2: SUCCESS] Direction verification CONVERGED successfully!")
                log_callback(f"  * Relative offset error reduced from {optimal_offset_1:.4f}° to {optimal_offset_2:.4f}°.")
                log_callback(f"  * Recommended Absolute Offset: {recommended:.4f}°")
            
            final_res = res_2
            final_res['recommended_joint_offset'] = recommended
            final_res['converged'] = True
            return final_res
        else:
            # Polarity is WRONG (error remained high or increased). Reverse polarity direction!
            if log_callback:
                log_callback(f"\n[STEP 2: FAIL] Direction verification failed (relative error: {optimal_offset_2:.4f}°).")
                log_callback(f"  * Direction polarity appears to be REVERSED.")
                
            # --- STEP 3: Fallback Polarity Correction Sweep ---
            temp_offset_3 = current_offset_deg - optimal_offset_1
            if log_callback:
                log_callback(f"\n[STEP 3/3] Reversing polarity. Shifting joint angle by {temp_offset_3:.4f}°...")
                
            res_3 = run_single_sweep(temp_offset_3)
            if not res_3:
                if log_callback: log_callback("[ERROR] Step 3 confirmation sweep failed. Aborting.")
                return None
                
            optimal_offset_3 = res_3['optimal_offset']
            recommended = temp_offset_3 + optimal_offset_3
            
            if log_callback:
                log_callback(f"\n[STEP 3 COMPLETE] Polarity corrected & final offset confirmed.")
                log_callback(f"  * Recommended Absolute Offset: {recommended:.4f}°")
                
            final_res = res_3
            final_res['recommended_joint_offset'] = recommended
            final_res['converged'] = True
            return final_res

    def perform_move_to_ready_pose(self, arm_side, mode, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving to {arm_side} Pitch/Head Ready Pose (Mode: {mode})...")
        torso = [0, 0, 0, 0, 0, 0]
        
        if mode == "wrist_pitch":
            if arm_side == "right":
                right_arm = np.deg2rad([-55, -45, 25, -127, 90, 0, 0])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-55, 45, -25, -127, -90, 0, 0])
        elif mode == "elbow":
            if arm_side == "right":
                right_arm = np.deg2rad([-107, -17, 0, 0, 73, -90, -107])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-107, 17, 0, 0, -73, -90, 107])
        else: # head mode
            if arm_side == "right":
                right_arm = np.deg2rad([-90, -45, 73, -107, 90, 90, 0])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-90, 45, -73, -107, -90, 90, 0])

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=[0, 0], minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def save_debug_orthogonal_plot(self, arm_side, frame, dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, log_callback=None):
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
            
            c_A, R_c_A, r_A, rmse_A, _, _, _ = BaseCalibrator.fit_circle_3d(pts_a)
            c_B, R_c_B, r_B, rmse_B, _, _, _ = BaseCalibrator.fit_circle_3d(pts_b)
            
            n_A = R_c_A[:, 2]
            n_B = R_c_B[:, 2]
            
            u_A = R_c_A[:, 0]
            v_A = R_c_A[:, 1]
            u_B = R_c_B[:, 0]
            v_B = R_c_B[:, 1]

            center_dist = np.linalg.norm(c_A - c_B)
            angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            def generate_circle_pts(center, normal, radius, u, v, num_points=100):
                theta = np.linspace(0, 2*np.pi, num_points)
                circle_pts = []
                for t in theta:
                    p = center + radius * (np.cos(t) * u + np.sin(t) * v)
                    circle_pts.append(p)
                return np.array(circle_pts)

            circle_pts_a = generate_circle_pts(c_A, n_A, r_A, u_A, v_A)
            circle_pts_b = generate_circle_pts(c_B, n_B, r_B, u_B, v_B)
            
            scale = min(r_A, r_B) * 0.4
            
            # Subplot 1: XY Projection (Top View)
            axes[0].scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.7, label=f'Axis A Raw ({len(pts_a)} pts)')
            axes[0].scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.7, label=f'Axis B Raw ({len(pts_b)} pts)')
            axes[0].plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r--', linewidth=2, label='Axis A Fit')
            axes[0].plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b--', linewidth=2, label='Axis B Fit')
            axes[0].scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
            axes[0].scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
            axes[0].plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label=f'Center Dist: {center_dist:.2f}mm')
            axes[0].arrow(c_A[0], c_A[1], n_A[0]*scale, n_A[1]*scale, color='darkred', head_width=2, width=0.5)
            axes[0].arrow(c_B[0], c_B[1], n_B[0]*scale, n_B[1]*scale, color='darkblue', head_width=2, width=0.5)
            axes[0].set_xlabel('X (mm)')
            axes[0].set_ylabel('Y (mm)')
            axes[0].set_title('X-Y Projection (Top View)')
            axes[0].set_aspect('equal')
            axes[0].grid(True)
            axes[0].legend(fontsize=9)
            
            # Subplot 2: YZ Projection (Side View)
            axes[1].scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.7, label='Axis A Raw')
            axes[1].scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.7, label='Axis B Raw')
            axes[1].plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r--', linewidth=2, label='Axis A Fit')
            axes[1].plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b--', linewidth=2, label='Axis B Fit')
            axes[1].scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
            axes[1].scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
            axes[1].plot([c_A[1], c_B[1]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2)
            axes[1].arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5)
            axes[1].arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5)
            axes[1].set_xlabel('Y (mm)')
            axes[1].set_ylabel('Z (mm)')
            axes[1].set_title('Y-Z Projection (Side View)')
            axes[1].set_aspect('equal')
            axes[1].grid(True)
            axes[1].legend(fontsize=9)
            
            # Subplot 3: XZ Projection (Front View)
            axes[2].scatter(pts_a[:, 0], pts_a[:, 2], c='red', s=15, alpha=0.7, label='Axis A Raw')
            axes[2].scatter(pts_b[:, 0], pts_b[:, 2], c='blue', s=15, alpha=0.7, label='Axis B Raw')
            axes[2].plot(circle_pts_a[:, 0], circle_pts_a[:, 2], 'r--', linewidth=2, label='Axis A Fit')
            axes[2].plot(circle_pts_b[:, 0], circle_pts_b[:, 2], 'b--', linewidth=2, label='Axis B Fit')
            axes[2].scatter([c_A[0]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
            axes[2].scatter([c_B[0]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
            axes[2].plot([c_A[0], c_B[0]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2)
            axes[2].arrow(c_A[0], c_A[2], n_A[0]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5)
            axes[2].arrow(c_B[0], c_B[2], n_B[0]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5)
            axes[2].set_xlabel('X (mm)')
            axes[2].set_ylabel('Z (mm)')
            axes[2].set_title('X-Z Projection (Front View)')
            axes[2].set_aspect('equal')
            axes[2].grid(True)
            axes[2].legend(fontsize=9)
            
            fig.suptitle(f"Orthogonal Multi-View Projections ({arm_side.upper()} Arm, {frame.upper()} Frame)\nCenter Distance: {center_dist:.3f} mm | Axis Normals Angle: {angle_between_normals:.2f}°", fontsize=13, fontweight='bold')
            plt.tight_layout()
            
            result_dir = os.path.join(os.path.dirname(__file__), "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"debug_orthogonal_circles_{arm_side}_{frame}.png"))
            plt.savefig(plot_save_path, dpi=150)
            plt.close()
            if log_callback:
                log_callback(f"[SUCCESS] Orthogonal debug plot saved to: {plot_save_path}")
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to save orthogonal debug plot for {frame}: {e}")

    def perform_calibration_sweep_5_or_3(self, arm_side, mode, log_callback=None, status_callback=None, use_head_tracking=True, current_offset_deg=0.0):
        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} OFFSET CALIBRATION SWEEP (ITERATIVE BRENT OPTIMIZATION)")
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

        # 0.5 degree steps from -20 to 20 deg (81 steps)
        sweep_angles = np.arange(-20.0, 20.01, 0.5)
        
        # Prepare Active Head/Camera Tracking
        head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
        q_head_0 = state.position[head_idx].copy() if (head_idx is not None and use_head_tracking and mode == "wrist_pitch") else None
        dyn_model = self.robot.get_dynamics()
        
        try:
            T_neck = self.compute_fk(self.robot, dyn_model, state.position, "link_head_2", "link_torso_5")
            p_neck = T_neck[:3, 3] if (use_head_tracking and mode == "wrist_pitch") else None
        except Exception:
            p_neck = None
            
        ee_name = f"ee_{arm_side}"
        try:
            T_ee_0 = self.compute_fk(self.robot, dyn_model, state.position, ee_name, "link_torso_5")
            p_marker_0 = T_ee_0[:3, 3] if (use_head_tracking and mode == "wrist_pitch") else None
        except Exception:
            p_marker_0 = None

        # Arm cand baseline pose (shifted by current offset)
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = initial_joint_pos[cand_joint] + np.radians(current_offset_deg)

        # 1. PHYSICAL SWEEP JOINT A
        if log_callback: log_callback(f"\n--- [1/2] Physically Sweeping Joint A (Index {sweep_joint_A}) ---")
        dataset_A = []
        for sa in sweep_angles:
            q_sweep = list(q_cand)
            q_sweep[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(sa)
            
            # Head tracking computation
            head_q_step = None
            if use_head_tracking and mode == "wrist_pitch" and q_head_0 is not None and p_neck is not None and p_marker_0 is not None:
                q_full_temp = np.array(state.position)
                if arm_side == "left":
                    q_full_temp[model.left_arm_idx] = q_sweep
                else:
                    q_full_temp[model.right_arm_idx] = q_sweep
                try:
                    T_ee_temp = self.compute_fk(self.robot, dyn_model, q_full_temp, ee_name, "link_torso_5")
                    p_marker_i = T_ee_temp[:3, 3]
                    v_0 = p_marker_0 - p_neck
                    v_i = p_marker_i - p_neck
                    yaw_geo_0 = np.arctan2(v_0[1], v_0[0])
                    pitch_geo_0 = np.arctan2(v_0[2], np.sqrt(v_0[0]**2 + v_0[1]**2))
                    yaw_geo_i = np.arctan2(v_i[1], v_i[0])
                    pitch_geo_i = np.arctan2(v_i[2], np.sqrt(v_i[0]**2 + v_i[1]**2))
                    yaw_diff = yaw_geo_i - yaw_geo_0
                    pitch_diff = pitch_geo_i - pitch_geo_0
                    yaw_target = q_head_0[0] + yaw_diff
                    pitch_target = q_head_0[1] - pitch_diff
                    yaw_target = np.clip(yaw_target, -25.0 * np.pi / 180.0, 25.0 * np.pi / 180.0)
                    pitch_target = np.clip(pitch_target, -20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0)
                    head_q_step = np.array([yaw_target, pitch_target])
                except Exception:
                    pass

            if arm_side == "left":
                self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=0.3, apply_offsets=False)
            else:
                self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=0.3, apply_offsets=False)
            time.sleep(0.1)
            
            res = self.marker_st.get_marker_transform(sampling_time=0.5, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                dataset_A.append((q_full_captured, pose))
                if log_callback: log_callback(f"  Captured Point {len(dataset_A)}/{len(sweep_angles)}: sa={sa:.1f}°")

        # Return to baseline
        head_q_step_cand = None
        if use_head_tracking and mode == "wrist_pitch" and q_head_0 is not None and p_neck is not None and p_marker_0 is not None:
            q_full_temp = np.array(state.position)
            if arm_side == "left":
                q_full_temp[model.left_arm_idx] = q_cand
            else:
                q_full_temp[model.right_arm_idx] = q_cand
            try:
                T_ee_temp = self.compute_fk(self.robot, dyn_model, q_full_temp, ee_name, "link_torso_5")
                p_marker_i = T_ee_temp[:3, 3]
                v_0 = p_marker_0 - p_neck
                v_i = p_marker_i - p_neck
                yaw_geo_0 = np.arctan2(v_0[1], v_0[0])
                pitch_geo_0 = np.arctan2(v_0[2], np.sqrt(v_0[0]**2 + v_0[1]**2))
                yaw_geo_i = np.arctan2(v_i[1], v_i[0])
                pitch_geo_i = np.arctan2(v_i[2], np.sqrt(v_i[0]**2 + v_i[1]**2))
                yaw_diff = yaw_geo_i - yaw_geo_0
                pitch_diff = pitch_geo_i - pitch_geo_0
                yaw_target = q_head_0[0] + yaw_diff
                pitch_target = q_head_0[1] - pitch_diff
                yaw_target = np.clip(yaw_target, -25.0 * np.pi / 180.0, 25.0 * np.pi / 180.0)
                pitch_target = np.clip(pitch_target, -20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0)
                head_q_step_cand = np.array([yaw_target, pitch_target])
            except Exception:
                pass

        if log_callback: log_callback("\n[INFO] Returning to candidate baseline...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=q_cand, head=head_q_step_cand, minimum_time=1.0, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=q_cand, head=head_q_step_cand, minimum_time=1.0, apply_offsets=False)
        time.sleep(0.1)

        # 2. PHYSICAL SWEEP JOINT B
        if log_callback: log_callback(f"\n--- [2/2] Physically Sweeping Joint B (Index {sweep_joint_B}) ---")
        dataset_B = []
        for sb in sweep_angles:
            q_sweep = list(q_cand)
            q_sweep[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(sb)
            
            # Head tracking computation
            head_q_step = None
            if use_head_tracking and mode == "wrist_pitch" and q_head_0 is not None and p_neck is not None and p_marker_0 is not None:
                q_full_temp = np.array(state.position)
                if arm_side == "left":
                    q_full_temp[model.left_arm_idx] = q_sweep
                else:
                    q_full_temp[model.right_arm_idx] = q_sweep
                try:
                    T_ee_temp = self.compute_fk(self.robot, dyn_model, q_full_temp, ee_name, "link_torso_5")
                    p_marker_i = T_ee_temp[:3, 3]
                    v_0 = p_marker_0 - p_neck
                    v_i = p_marker_i - p_neck
                    yaw_geo_0 = np.arctan2(v_0[1], v_0[0])
                    pitch_geo_0 = np.arctan2(v_0[2], np.sqrt(v_0[0]**2 + v_0[1]**2))
                    yaw_geo_i = np.arctan2(v_i[1], v_i[0])
                    pitch_geo_i = np.arctan2(v_i[2], np.sqrt(v_i[0]**2 + v_i[1]**2))
                    yaw_diff = yaw_geo_i - yaw_geo_0
                    pitch_diff = pitch_geo_i - pitch_geo_0
                    yaw_target = q_head_0[0] + yaw_diff
                    pitch_target = q_head_0[1] - pitch_diff
                    yaw_target = np.clip(yaw_target, -25.0 * np.pi / 180.0, 25.0 * np.pi / 180.0)
                    pitch_target = np.clip(pitch_target, -20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0)
                    head_q_step = np.array([yaw_target, pitch_target])
                except Exception:
                    pass

            if arm_side == "left":
                self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=0.3, apply_offsets=False)
            else:
                self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=0.3, apply_offsets=False)
            time.sleep(0.1)
            
            res = self.marker_st.get_marker_transform(sampling_time=0.5, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                dataset_B.append((q_full_captured, pose))
                if log_callback: log_callback(f"  Captured Point {len(dataset_B)}/{len(sweep_angles)}: sb={sb:.1f}°")

        # Return arm and head to original ready pose
        if log_callback: log_callback("\n[INFO] Sweep finished. Returning arm and head to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0, apply_offsets=False)

        if len(dataset_A) < 6 or len(dataset_B) < 6:
            if log_callback: log_callback("[ERROR] Too few valid captured points. Calibration failed.")
            return None

        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # Save captured sweep points to debug txt files before offline optimization
        self.save_debug_points(
            arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, 
            cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback
        )

        # 3. OFFLINE DIRECT ANGLE OPTIMIZATION (Fitted Circle Normal Orthogonality in Torso Frame)
        if log_callback: log_callback("\n--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Torso Frame) ---")
        
        # Project raw captured points of Sweep A and Sweep B to torso frame (independent of arm joint offsets)
        pts_torso_A = []
        for q_full, pose in dataset_A:
            p_cam = pose[:3, 3]
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            pts_torso_A.append(p_meas_t5 * 1000.0) # mm
            
        pts_torso_B = []
        for q_full, pose in dataset_B:
            p_cam = pose[:3, 3]
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            pts_torso_B.append(p_meas_t5 * 1000.0) # mm
            
        # Fit 3D circles in torso frame
        c_A, R_circle_A, r_A, rmse_A, _, _, _ = BaseCalibrator.fit_circle_3d(pts_torso_A)
        c_B, R_circle_B, r_B, rmse_B, _, _, _ = BaseCalibrator.fit_circle_3d(pts_torso_B)
        
        n_A = R_circle_A[:, 2] # Circle A Normal vector (sweep rotation axis A)
        n_B = R_circle_B[:, 2] # Circle B Normal vector (sweep rotation axis B)
        
        # Calculate angle between normals (exactly as in debug_joint_plotter.py)
        angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
        offset_magnitude_deg = angle_between_normals
        
        # Evaluate direction/sign in ee frame
        def evaluate_ee_ortho_error(offset_deg):
            offset_r = np.radians(offset_deg)
            pts_ee_A_eval = []
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_r
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                pts_ee_A_eval.append(p_ee * 1000.0)
                
            pts_ee_B_eval = []
            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_r
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                pts_ee_B_eval.append(p_ee * 1000.0)
                
            _, R_c_A, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_A_eval)
            _, R_c_B, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_B_eval)
            dot_v = np.dot(R_c_A[:, 2], R_c_B[:, 2])
            ang_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_v), -1.0, 1.0))))
            return ang_err
            
        error_plus = evaluate_ee_ortho_error(1.0)
        error_minus = evaluate_ee_ortho_error(-1.0)
        
        sign = 1.0 if error_plus < error_minus else -1.0
        optimal_offset_deg = sign * offset_magnitude_deg
        if mode == "elbow":
            optimal_offset_deg = -abs(optimal_offset_deg)
        optimal_offset_rad = np.radians(optimal_offset_deg)

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()})")
            log_callback("="*50)
            log_callback(f"  * Torso Circle Normals Angle : {angle_between_normals:.4f} deg")
            log_callback(f"  * Estimated Optimal Offset   : {optimal_offset_deg:.6f} deg")
            log_callback("="*50)

        # Prepare final visual projection datasets
        pts_ee_A_best = []
        for q_full, pose in dataset_A:
            p_cam = pose[:3, 3]
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_A_best.append(p_ee * 1000.0)
            
        pts_ee_B_best = []
        for q_full, pose in dataset_B:
            p_cam = pose[:3, 3]
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_B_best.append(p_ee * 1000.0)

        # High-precision 3D fit circles for plotting
        c3d_A, R_circle_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A = BaseCalibrator.fit_circle_3d(pts_ee_A_best)
        c3d_B, R_circle_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B = BaseCalibrator.fit_circle_3d(pts_ee_B_best)

        # Simultaneously generate and save orthogonal debug plot
        self.save_debug_orthogonal_plot(arm_side, "torso", dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, log_callback)
        self.save_debug_orthogonal_plot(arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, log_callback)

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
                    
                marker_6_res = MarkerCalibrator.fit_circle_3d_and_6dof_misalignment(
                    captured_poses_torso, 
                    sweep_angles, 
                    axis_prior=[1.0, 0.0, 0.0]
                )
                if marker_6_res:
                    marker_6_res['axis'] = marker_6_res['axis_opt']
                
                if log_callback and marker_6_res:
                    log_callback("\n" + "-"*50)
                    log_callback("  [SIMULTANEOUS MARKER AXIS 6 ESTIMATION RESULTS]")
                    log_callback("-"*50)
                    log_callback(f"    - Fitted Sweep Radius    : {marker_6_res['radius']:.3f} mm")
                    log_callback(f"    - Axis 6 Fitting RMSE    : {marker_6_res['rmse']:.3f} mm")
                    log_callback(f"    - Axis Direction Vector  : {np.round(marker_6_res['axis_opt'], 4)}")
                    log_callback(f"    - Jitter StdDev (Tilt)   : {np.std(marker_6_res.get('tilt_list', [0.0])):.3f} deg")
                    log_callback("-"*50 + "\n")
            except Exception as e:
                if log_callback:
                    log_callback(f"\n[WARN] Failed simultaneous Marker Axis 6 calculation: {e}\n")

        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
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
            'marker_6_res': marker_6_res
        }

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=30.0):
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
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = initial_joint_pos[cand_joint] + np.radians(current_offset_deg)

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
            time.sleep(0.03) # 30Hz polling
            
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
        self.save_debug_points(
            arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, 
            cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback
        )

        # Clean/Downsample dataset to make optimization fast but robust
        step_A = max(1, len(dataset_A) // 60)
        step_B = max(1, len(dataset_B) // 60)
        dataset_A = dataset_A[::step_A]
        dataset_B = dataset_B[::step_B]

        # 3. OFFLINE DIRECT ANGLE OPTIMIZATION (Fitted Circle Normal Orthogonality in Torso Frame)
        if log_callback: log_callback("\n--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Torso Frame) ---")
        
        # Project raw captured points of Sweep A and Sweep B to torso frame (independent of arm joint offsets)
        pts_torso_A = []
        for q_full, pose in dataset_A:
            p_cam = pose[:3, 3]
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            pts_torso_A.append(p_meas_t5 * 1000.0) # mm
            
        pts_torso_B = []
        for q_full, pose in dataset_B:
            p_cam = pose[:3, 3]
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            pts_torso_B.append(p_meas_t5 * 1000.0) # mm
            
        # Fit 3D circles in torso frame
        c_A, R_circle_A, r_A, rmse_A, _, _, _ = BaseCalibrator.fit_circle_3d(pts_torso_A)
        c_B, R_circle_B, r_B, rmse_B, _, _, _ = BaseCalibrator.fit_circle_3d(pts_torso_B)
        
        n_A = R_circle_A[:, 2] # Circle A Normal vector (sweep rotation axis A)
        n_B = R_circle_B[:, 2] # Circle B Normal vector (sweep rotation axis B)
        
        # Calculate angle between normals (exactly as in debug_joint_plotter.py)
        angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
        offset_magnitude_deg = angle_between_normals
        
        # Evaluate direction/sign in ee frame
        def evaluate_ee_ortho_error(offset_deg):
            offset_r = np.radians(offset_deg)
            pts_ee_A_eval = []
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_r
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                pts_ee_A_eval.append(p_ee * 1000.0)
                
            pts_ee_B_eval = []
            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_r
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                pts_ee_B_eval.append(p_ee * 1000.0)
                
            _, R_c_A, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_A_eval)
            _, R_c_B, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_B_eval)
            dot_v = np.dot(R_c_A[:, 2], R_c_B[:, 2])
            ang_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_v), -1.0, 1.0))))
            return ang_err
            
        error_plus = evaluate_ee_ortho_error(1.0)
        error_minus = evaluate_ee_ortho_error(-1.0)
        
        sign = 1.0 if error_plus < error_minus else -1.0
        optimal_offset_deg = sign * offset_magnitude_deg
        if mode == "elbow":
            optimal_offset_deg = -abs(optimal_offset_deg)
        optimal_offset_rad = np.radians(optimal_offset_deg)

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()} - CONTINUOUS)")
            log_callback("="*50)
            log_callback(f"  * Torso Circle Normals Angle : {angle_between_normals:.4f} deg")
            log_callback(f"  * Estimated Optimal Offset   : {optimal_offset_deg:.6f} deg")
            log_callback("="*50)

        # Prepare final visual projection datasets
        pts_ee_A_best = []
        for q_full, pose in dataset_A:
            p_cam = pose[:3, 3]
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_A_best.append(p_ee * 1000.0)
            
        pts_ee_B_best = []
        for q_full, pose in dataset_B:
            p_cam = pose[:3, 3]
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_B_best.append(p_ee * 1000.0)

        # High-precision 3D fit circles for plotting
        c3d_A, R_circle_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A = BaseCalibrator.fit_circle_3d(pts_ee_A_best)
        c3d_B, R_circle_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B = BaseCalibrator.fit_circle_3d(pts_ee_B_best)

        # Simultaneously generate and save orthogonal debug plot
        self.save_debug_orthogonal_plot(arm_side, "torso", dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, log_callback)
        self.save_debug_orthogonal_plot(arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, log_callback)

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
                    
                from MarkerCalibrator import MarkerCalibrator
                marker_6_res = MarkerCalibrator.fit_circle_3d_and_6dof_misalignment(
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
                    log_callback(f"    - Axis 6 Fitting RMSE    : {marker_6_res['rmse']:.3f} mm")
                    log_callback(f"    - Axis Direction Vector  : {np.round(marker_6_res['axis_opt'], 4)}")
                    log_callback(f"    - Jitter StdDev (Tilt)   : {np.std(marker_6_res.get('tilt_list', [0.0])):.3f} deg")
                    log_callback("-"*50 + "\n")
            except Exception as e:
                if log_callback:
                    log_callback(f"\n[WARN] Failed simultaneous Marker Axis 6 calculation: {e}\n")

        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
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
            'marker_6_res': marker_6_res
        }

    def perform_head_calibration_sweep(self, arm_side, log_callback=None, status_callback=None):
        if log_callback:
            log_callback("\n" + "="*50)
            log_callback("   STARTING HEAD YAW/PITCH CALIBRATION SWEEP")
            log_callback("="*50)

        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system (marker_st) is not initialized.")
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

        dyn_model = self.robot.get_dynamics()
        q_current = self.robot.get_state().position
        
        # Load calibrated Tf_to_marker
        Tf_to_marker_val = self.camera_config.get(f"Tf_to_marker_{arm_side}", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T_ee_to_marker = self.make_transform(Tf_to_marker_val)

        ee_link = f"ee_{arm_side}"
        T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_current, ee_link, base_link="link_torso_5")
        
        # Fixed marker position in link_torso_5 base frame
        T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker
        p_marker_t5 = T_t5_to_marker[:3, 3]

        # Load mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # Cross Sweep angles
        sweep_deg = list(range(-10, 11))
        captured_data = []

        # Part A: Yaw sweep
        if log_callback: log_callback("\n[Part A] Sweeping Head Yaw (-10 to 10 deg)...")
        for yaw in sweep_deg:
            q_head = np.deg2rad([yaw, 0.0])
            self.movej(self.robot, head=q_head, minimum_time=1.0)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_meas = pose[:3, 3]
                captured_data.append((yaw, 0.0, p_meas))

        # Return head to zero
        self.movej(self.robot, head=[0, 0], minimum_time=1.5)
        time.sleep(0.5)

        # Part B: Pitch sweep
        if log_callback: log_callback("\n[Part B] Sweeping Head Pitch (-10 to 10 deg)...")
        for pitch in sweep_deg:
            if pitch == 0: continue
            q_head = np.deg2rad([0.0, pitch])
            self.movej(self.robot, head=q_head, minimum_time=1.0)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_meas = pose[:3, 3]
                captured_data.append((0.0, pitch, p_meas))

        # Return to base pose
        self.movej(self.robot, head=[0, 0], minimum_time=1.5)

        if len(captured_data) < 10:
            if log_callback: log_callback("[ERROR] Too few valid frames captured. Head calibration failed.")
            return None

        # 3. Solver Optimization for Head Offsets
        if log_callback: log_callback("\n[INFO] Performing Levenberg-Marquardt optimizer...")
        head_names = ["joint_head_1", "joint_head_2"]
        
        # Initial guess (zero offsets)
        init_guess = [0.0, 0.0]

        def loss_func(params):
            yaw_off, pitch_off = params
            residuals = []
            for yaw_enc, pitch_enc, p_cam in captured_data:
                q_head = np.deg2rad([yaw_enc + yaw_off, pitch_enc + pitch_off])
                
                state_head = dyn_model.make_state(["link_torso_5", "link_head_2"], head_names)
                state_head.set_q(q_head)
                dyn_model.compute_forward_kinematics(state_head)
                T_t5_to_h2 = dyn_model.compute_transformation(state_head, 0, 1)
                
                T_t5_to_cam = T_t5_to_h2 @ T_mount_to_cam
                p_marker_pred = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                
                err = p_marker_t5 - p_marker_pred
                residuals.extend(err * 1000.0) # in mm
            return np.array(residuals)

        res_opt = least_squares(loss_func, init_guess, loss="huber")
        opt_yaw_off, opt_pitch_off = res_opt.x
        rmse = np.sqrt(np.mean(res_opt.fun**2))

        # Generate plot datasets
        meas_pts_yaw = []
        pred_pts_yaw = []
        meas_pts_pitch = []
        pred_pts_pitch = []

        for y_enc, p_enc, p_cam in captured_data:
            q_head_opt = np.deg2rad([y_enc + opt_yaw_off, p_enc + opt_pitch_off])
            state_head = dyn_model.make_state(["link_torso_5", "link_head_2"], head_names)
            state_head.set_q(q_head_opt)
            dyn_model.compute_forward_kinematics(state_head)
            T_t5_to_h2 = dyn_model.compute_transformation(state_head, 0, 1)
            T_t5_to_cam = T_t5_to_h2 @ T_mount_to_cam
            p_marker_pred = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            
            if y_enc != 0.0:
                meas_pts_yaw.append([y_enc, p_marker_t5[0]*1000, p_marker_t5[1]*1000, p_marker_t5[2]*1000])
                pred_pts_yaw.append([y_enc, p_marker_pred[0]*1000, p_marker_pred[1]*1000, p_marker_pred[2]*1000])
            else:
                meas_pts_pitch.append([p_enc, p_marker_t5[0]*1000, p_marker_t5[1]*1000, p_marker_t5[2]*1000])
                pred_pts_pitch.append([p_enc, p_marker_pred[0]*1000, p_marker_pred[1]*1000, p_marker_pred[2]*1000])

        return {
            'mode': 'head',
            'opt_yaw': opt_yaw_off,
            'opt_pitch': opt_pitch_off,
            'rmse': rmse,
            'meas_pts_yaw': np.array(meas_pts_yaw) if meas_pts_yaw else np.empty((0, 4)),
            'pred_pts_yaw': np.array(pred_pts_yaw) if pred_pts_yaw else np.empty((0, 4)),
            'meas_pts_pitch': np.array(meas_pts_pitch) if meas_pts_pitch else np.empty((0, 4)),
            'pred_pts_pitch': np.array(pred_pts_pitch) if pred_pts_pitch else np.empty((0, 4))
        }
