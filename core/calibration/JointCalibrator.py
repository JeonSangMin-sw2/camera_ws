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
class DebugLogger:
    def __init__(self, original_log_callback, file_path):
        self.original_log_callback = original_log_callback
        self.file_path = file_path
        self.buffer = []
        
    def log(self, msg):
        self.buffer.append(msg)
        msg_upper = msg.upper()
        if (
            "[SAFETY WARNING]" in msg_upper or
            "[SUCCESS]" in msg_upper or
            "[ERROR]" in msg_upper or
            "[WARN]" in msg_upper or
            "[INFO]" in msg_upper or
            "[VALIDATION SWEEP]" in msg_upper or
            "[ITERATION" in msg_upper or
            "RECOMMENDED ABSOLUTE OFFSET" in msg_upper or
            "STEP CORRECTION" in msg_upper or
            "COMMENCING" in msg_upper or
            "SWEPT" in msg_upper or
            "SWEEP COMPLETE" in msg_upper or
            "STARTING" in msg_upper
        ):
            if self.original_log_callback:
                self.original_log_callback(msg)
                
    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.buffer) + "\n")
        except Exception:
            pass

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.use_angle_based_fitting = True

    def perform_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None, save_debug=False):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        config_dir = os.path.abspath(os.path.dirname(__file__))
        debug_file_path = os.path.join(config_dir, f"joint_calib_debug_{arm_side}_{mode}.txt")
        logger = DebugLogger(log_callback, debug_file_path)
        original_log = log_callback
        log_callback = logger.log

        try:
            self.last_staged_offset = None
            self.last_diff_angle = None
            if original_log:
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
                
            max_iterations = 10
            staged_offset = current_offset_deg
            final_res = None
            first_res = None
            converged = False
            
            # Sign-reversal tracking state
            prev_error = None
            prev_step_correction = 0.0
            direction_multiplier = 1.0
            
            for i in range(1, max_iterations + 1):
                if getattr(self, 'stop_requested', False):
                    if log_callback: log_callback("[INFO] Joint calibration aborted due to stop request.")
                    return None
                    
                if log_callback:
                    log_callback(f"\n[ITERATION {i}/{max_iterations}] Sweeping physically with staged offset {staged_offset:.4f}°...")
                
                # Perform physical sweep (or simulated sweep in mock mode) at the current staged offset
                res = run_single_sweep(staged_offset)
                if not res:
                    if log_callback: log_callback(f"[ERROR] Iteration {i} sweep failed. Aborting calibration.")
                    return None
                
                if i == 1:
                    first_res = res
                final_res = res
        
                angle_error = res.get('angle_between_normals', 0.0)
                
                # Define angle deviation and center distance metric based on the mode
                if mode == "wrist_roll_v13":
                    angle_dev = abs(res.get('diff_angle_deg', 0.0))
                    center_dist = res.get('perp_dist_after', 999.0)
                    current_error = center_dist
                elif mode == "wrist_yaw2":
                    angle_dev = abs(res.get('diff_angle_deg', 0.0))
                    center_dist = res.get('center_dist', 999.0)
                    r_A = res.get('r_A', 0.0)
                    r_B = res.get('r_B', 0.0)
                    size_error = abs(r_A - r_B)
                    current_error = center_dist
                elif mode == "wrist_pitch_v13":
                    angle_dev = angle_error
                    # For wrist_pitch_v13, center_dist is perp_dist_after (approx forearm length)
                    perp_dist_val = res.get('perp_dist_after', 999.0)
                    nominal_dist_35 = res.get('nominal_dist_35', 235.0)
                    center_dist = abs(perp_dist_val - nominal_dist_35)
                    r_A = res.get('r_A', 0.0)
                    r_B = res.get('r_B', 0.0)
                    size_error = abs(r_A - r_B)
                    current_error = max(size_error, center_dist)
                else: # wrist_pitch (v1.2) or elbow
                    angle_dev = angle_error
                    center_dist = res.get('center_dist', 999.0)
                    r_A = res.get('r_A', 0.0)
                    r_B = res.get('r_B', 0.0)
                    size_error = abs(r_A - r_B)
                    current_error = max(size_error, center_dist)
                
                # Print iteration summary
                if log_callback:
                    if mode == "wrist_roll_v13":
                        log_callback(f"  * Angle Error (Deviation)          : {angle_dev:.4f}°")
                        log_callback(f"  * Perpendicular Distance (After)   : {center_dist:.4f} mm")
                        log_callback(f"  * Perpendicular Distance (Before)  : {res.get('perp_dist_before', 999.0):.4f} mm")
                    elif mode == "wrist_pitch_v13":
                        log_callback(f"  * Angle Error (Deviation)          : {angle_dev:.4f}°")
                        log_callback(f"  * Forearm Length (Center Dist)     : {perp_dist_val:.4f} mm (Error: {center_dist:.4f} mm)")
                        log_callback(f"  * Radii Difference (r3 - r5)       : {size_error:.4f} mm")
                        log_callback(f"  * Max Fitting Error Metric         : {current_error:.4f} mm")
                    else:
                        log_callback(f"  * Angle Error (Deviation)     : {angle_dev:.4f}°")
                        log_callback(f"  * Circle Size Error (r_A-r_B) : {size_error:.4f} mm")
                        log_callback(f"  * Center Distance Error       : {center_dist:.4f} mm")
                        log_callback(f"  * Max Fitting Error Metric    : {current_error:.4f} mm")
                
                # Use the pre-calculated damped optimal offset correction to ensure convergence
                raw_optimal_offset = res.get('optimal_offset', 0.0)
                step_correction = direction_multiplier * raw_optimal_offset
                
                # Convergence check:
                # For orthogonal modes (wrist_roll_v13, wrist_yaw2), angle_dev is mechanically locked close to 0
                # and cannot be used as a convergence metric. We only check step correction.
                if mode in ("wrist_roll_v13", "wrist_yaw2"):
                    converged_criteria = (abs(step_correction) < 0.05)
                else:
                    converged_criteria = (abs(step_correction) < 0.05 or angle_dev <= 0.1 or center_dist <= 0.1)
                
                if converged_criteria:
                    converged = True
                    if log_callback:
                        log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                        if abs(step_correction) < 0.05:
                            log_callback(f"  * Step Correction: {step_correction:.4f}° < 0.05° (reached resolution limit)")
                        else:
                            log_callback(f"  * Circle Normals Angle Error: {angle_dev:.4f}° <= 0.1°")
                            log_callback(f"  * Center Distance Error: {center_dist:.4f} mm <= 0.1 mm")
                        log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                    break
                
                # Sign-reversal fail-safe check:
                # If error increased compared to previous iteration OR if the candidate offset exceeds limits,
                # backtrack and flip sign.
                candidate_staged_offset = staged_offset + step_correction
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
                bounds_breached = (candidate_staged_offset < off_min - 0.01 or candidate_staged_offset > off_max + 0.01)

                current_error_metric = angle_dev
                if (prev_error is not None and current_error_metric > prev_error + 0.05) or bounds_breached:
                    if log_callback:
                        if bounds_breached:
                            log_callback(f"  [SAFETY WARNING] Candidate offset {candidate_staged_offset:.4f}° exceeds bounds [{off_min}°, {off_max}°]!")
                        else:
                            log_callback(f"  [SAFETY WARNING] Error increased from {prev_error:.4f}° to {current_error_metric:.4f}°!")
                        log_callback(f"                   Reversing calibration direction and reverting offset from {staged_offset:.4f}°...")
                    
                    # Backtrack (revert previous step correction if one was applied)
                    if prev_step_correction != 0.0:
                        staged_offset -= prev_step_correction
                        step_correction = -prev_step_correction
                    else:
                        step_correction = -step_correction
                        
                    direction_multiplier *= -1.0
                    prev_step_correction = step_correction
                    staged_offset += step_correction
                    
                    if log_callback:
                        log_callback(f"                   New staged offset: {staged_offset:.4f}° (direction_multiplier={direction_multiplier})")
                else:
                    # Normal update: save current error for future comparison, apply correction
                    prev_error = current_error_metric
                    prev_step_correction = step_correction
                    staged_offset += step_correction
                
                # Safety: clamp staged_offset to the joint's configured offset range
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
                if staged_offset < off_min or staged_offset > off_max:
                    if log_callback:
                        log_callback(f"  [SAFETY WARNING] Staged offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                    staged_offset = float(np.clip(staged_offset, off_min, off_max))
                    
                if log_callback:
                    log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                    
            # Final range safety: clamp to configured offset_range
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
            if staged_offset < off_min or staged_offset > off_max:
                if log_callback:
                    log_callback(f"  [SAFETY WARNING] Recommended final offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                staged_offset = float(np.clip(staged_offset, off_min, off_max))
        
            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Joint calibration aborted before final report.")
                return None
        
            # Build clean final output dict — UI only needs these fields
            final_output = {
                'mode': mode,
                'recommended_joint_offset': staged_offset,
                'optimal_offset': staged_offset,
                'converged': converged,
                'perp_dist_before': final_res.get('perp_dist_before', float('nan')) if final_res else float('nan'),
                'perp_dist_after': final_res.get('perp_dist_after', float('nan')) if final_res else float('nan'),
                'axial_offset_mm': final_res.get('axial_offset_mm', float('nan')) if final_res else float('nan'),
                'lateral_offset_mm': final_res.get('lateral_offset_mm', float('nan')) if final_res else float('nan'),
                'r_A': final_res.get('r_A', float('nan')) if final_res else float('nan'),
                'r_B': final_res.get('r_B', float('nan')) if final_res else float('nan'),
                'x_cal': float('nan'),
                'y_cal': float('nan'),
                'z_cal': float('nan'),
            }
        
            # Since the final iteration ran physically at (or very close to) the final offset,
            # we reuse final_res as the validation sweep to avoid a redundant extra sweep.
            validation_res = final_res
            if validation_res and first_res:
                plot_path = self.save_calibration_comparison_plot(arm_side, mode, first_res, validation_res, log_callback=log_callback)
                final_output['plot_path_combined'] = plot_path
            
            if final_res and mode in ("wrist_roll_v13", "wrist_yaw2"):
                x_c, y_c, z_c, roll_c, pitch_c, yaw_c = self.calculate_nominal_marker_coordinates(arm_side, mode, final_res, staged_offset, log_callback)
                final_output['x_cal'] = x_c
                final_output['y_cal'] = y_c
                final_output['z_cal'] = z_c
                final_output['roll_cal'] = roll_c
                final_output['pitch_cal'] = pitch_c
                final_output['yaw_cal'] = yaw_c
            
            return final_output
        finally:
            logger.save()

    def calculate_nominal_marker_coordinates(self, arm_side, mode, res, optimal_offset_deg, log_callback=None):
        if not self.robot or not res:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        try:
            # Retrieve wrist link length dynamically from kinematics
            try:
                dyn_model_link = self.robot.get_dynamics()
                names_link = self.robot.model().robot_joint_names
                state_link = dyn_model_link.make_state(
                    [f"link_{arm_side}_arm_5", f"ee_{arm_side}"],
                    names_link
                )
                state_link.set_q(self.robot.get_state().position)
                dyn_model_link.compute_forward_kinematics(state_link)
                T_link = dyn_model_link.compute_transformation(state_link, 0, 1)
                L_5_ee = np.linalg.norm(T_link[:3, 3]) * 1000.0 # m to mm
            except Exception:
                L_5_ee = 300.0 if getattr(self.robot, "is_pure_mock", False) else 126.1

            r_A = res.get('r_A', 0.0)
            r_B = res.get('r_B', 0.0)
            
            dataset_A = res.get('_dataset_A')
            dataset_B = res.get('_dataset_B')
            pts_a_cam = res['_plot_data']['pts_a_cam']
            n_A = res['_plot_data']['n_A']
            n_B = res['_plot_data']['n_B']
            c_B_c = res['_plot_data']['c_B']
            c_A_c = res['_plot_data']['c_A']

            state = self.robot.get_state()
            model = self.robot.model()
            arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
            cand_joint = 6 # Joint 7 for wrist_roll_v13 and wrist_yaw2
            sweep_joint_B = 5 # Joint 6 for Sweep B
            
            # Re-calculate idx_A_mid and idx_B_mid
            angles_A = [dataset_A[k][0][arm_idx[cand_joint]] for k in range(len(dataset_A))]
            idx_A_mid = np.argmin(np.abs(angles_A))
            
            angles_B = [dataset_B[k][0][arm_idx[sweep_joint_B]] for k in range(len(dataset_B))]
            idx_B_mid = np.argmin(np.abs(angles_B))
            
            p_A_ready = pts_a_cam[idx_A_mid]

            # 1. Get ready pose marker rotation in camera frame from Sweep B (which preserves Joint 7 offset)
            R_cam_to_marker_ready = dataset_B[idx_B_mid][1][:3, :3]
            
            # 2. Transform circle normals (which are in camera frame) to marker frame
            n_A_marker = R_cam_to_marker_ready.T @ n_A
            n_B_marker = R_cam_to_marker_ready.T @ n_B
            
            # Retrieve nominal templates
            ver_key = "1.3" if self.is_v13() else "1.2"
            nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
            nominal_rpy = nominal_vec[3:6]
            
            R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
            z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
            y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
            x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
            
            if mode == "wrist_yaw2":
                # Analytical radii-based calibration for v1.2
                y_cal = (r_A / 1000.0) if arm_side == "left" else (-r_A / 1000.0)
                z_cal = -(L_5_ee - r_B) / 1000.0
                x_cal = 0.0
                
                if np.dot(n_A_marker, z_ee_m_ideal) < 0:
                    n_A_marker = -n_A_marker
                if np.dot(n_B_marker, y_ee_m_ideal) < 0:
                    n_B_marker = -n_B_marker
                    
                z_col = n_A_marker
                y_col = n_B_marker - np.dot(n_B_marker, z_col) * z_col
                y_col /= np.linalg.norm(y_col)
                x_col = np.cross(y_col, z_col)
            else:
                # FK-based calibration for v1.3
                dyn_model = self.robot.get_dynamics()
                initial_joint_pos = res.get('_initial_joint_pos')
                
                q_ready_full = np.array(state.position)
                for idx, val in zip(arm_idx, initial_joint_pos):
                    q_ready_full[idx] = val
                
                # Apply the current calibrated joint offset (optimal_offset_deg) to the candidate joint
                q_ready_full[arm_idx[cand_joint]] += np.radians(optimal_offset_deg)
                
                ee_name = f"ee_{arm_side}"
                T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_ready_full, ee_name)
                
                T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_ready_full, "link_head_2", "link_torso_5")
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                T_head_to_cam = self.make_transform(mount_to_cam)
                T_t5_to_cam = T_t5_to_head @ T_head_to_cam
                
                T_cam_to_ee = np.linalg.inv(T_t5_to_cam) @ T_t5_to_ee
                
                p_marker_cam = np.array([p_A_ready[0], p_A_ready[1], p_A_ready[2], 1.0]) / 1000.0
                p_marker_ee = np.linalg.inv(T_cam_to_ee) @ p_marker_cam
                x_cal, y_cal, z_cal = p_marker_ee[0], p_marker_ee[1], p_marker_ee[2]
                
                if np.dot(n_A_marker, x_ee_m_ideal) < 0:
                    n_A_marker = -n_A_marker
                if np.dot(n_B_marker, y_ee_m_ideal) < 0:
                    n_B_marker = -n_B_marker
                    
                x_col = n_A_marker
                y_col = n_B_marker - np.dot(n_B_marker, x_col) * x_col
                y_col /= np.linalg.norm(y_col)
                z_col = np.cross(x_col, y_col)
            
            # Construct R_ee_m_actual
            R_ee_m_actual = np.column_stack((x_col, y_col, z_col)).T
            euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
            yaw_e, pitch_e, roll_e = euler_deg
            if arm_side == "right" and yaw_e < 0:
                yaw_e += 360.0
                
            roll_cal = float(roll_e)
            pitch_cal = float(pitch_e)
            yaw_cal = float(yaw_e)
            
            # Format outputs ensuring safety bounds on signs for v1.2
            if mode == "wrist_yaw2":
                x_cal = 0.0
                y_cal = float(y_cal)
                z_cal = -float(abs(z_cal))
            else:
                x_cal = float(x_cal)
                y_cal = float(y_cal)
                z_cal = float(z_cal)

            if log_callback:
                log_callback(f"[INFO] Calculated nominal marker position in EE frame: X={x_cal*1000.0:.2f}, Y={y_cal*1000.0:.2f}, Z={z_cal*1000.0:.2f} mm")
                log_callback(f"[INFO] Calculated nominal marker angles in EE frame: R={roll_cal:.2f}, P={pitch_cal:.2f}, Y={yaw_cal:.2f} deg")
            
            return x_cal, y_cal, z_cal, roll_cal, pitch_cal, yaw_cal
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to calculate nominal marker coordinates: {e}")
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')



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
        
        if mode == "elbow":
            ready_mode = "elbow"
        elif mode == "wrist_yaw2":
            ready_mode = "wrist_yaw2"
        elif mode == "wrist_roll_v13":
            ready_mode = "wrist_roll_v13"
        else:
            ready_mode = "wrist_pitch"
        
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

    def compute_shortest_distance_between_lines(self, cA, nA, cB, nB):
        nA_norm = nA / np.linalg.norm(nA)
        nB_norm = nB / np.linalg.norm(nB)
        cross = np.cross(nA_norm, nB_norm)
        cross_norm = np.linalg.norm(cross)
        diff = cB - cA
        if cross_norm > 1e-4:
            return abs(np.dot(diff, cross)) / cross_norm
        else:
            return np.linalg.norm(diff - np.dot(diff, nA_norm) * nA_norm)

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
                    dist_before = self.compute_shortest_distance_between_lines(
                        first_pd['c_A'], first_pd['n_A'], first_pd['c_B'], first_pd['n_B']
                    )
                    before_dist_str = f" | Axis 3-5 Dist = {dist_before:.2f} mm"
                if all(k in final_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_after = self.compute_shortest_distance_between_lines(
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

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} CONTINUOUS OFFSET CALIBRATION SWEEP")
            if current_offset_deg != 0.0:
                log_callback(f"   [Baseline Shift (Current Applied Offset): {current_offset_deg:.4f}°]")
            log_callback("="*50)

        is_camera_mock = (self.marker_st is None or type(self.marker_st).__name__ == "SimulatedMarkerTransform")

        if not is_camera_mock:
            # Pre-check marker visibility
            initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not initial_check:
                if log_callback: log_callback("[ERROR] Marker is not visible.")
                if status_callback: status_callback(False)
                return None
            if status_callback: status_callback(True)
        else:
            if status_callback: status_callback(True)

        if getattr(self, 'stop_requested', False):
            return None

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

        # Define joint parameters from JOINT_CONFIGS
        jcfg = self.JOINT_CONFIGS[mode]
        cand_joint = jcfg["cand_joint"]
        sweep_joint_A = jcfg["sweep_joint_A"]
        sweep_joint_B = jcfg["sweep_joint_B"]

        dyn_model = self.robot.get_dynamics()
        ee_name = f"ee_{arm_side}"

        # Arm cand baseline pose (shifted by current offset)
        offset_key = jcfg.get("offset_key", mode)
        if arm_side in self.joint_offsets:
            active_offset = self.joint_offsets[arm_side].get(offset_key, 0.0)
        else:
            active_offset = self.joint_offsets.get(offset_key, 0.0)
        nominal_joint_pos = initial_joint_pos[cand_joint] - np.radians(active_offset)
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = nominal_joint_pos - np.radians(current_offset_deg)

        # Determine sweep ranges from JOINT_CONFIGS
        range_A = jcfg.get("sweep_range_A", 20.0)
        range_B = jcfg.get("sweep_range_B", 20.0)

        # 1. PHYSICAL SWEEP JOINT A
        if log_callback: log_callback(f"\n--- Commencing Continuous Sweep on Joint A (Index {sweep_joint_A}, duration={sweep_duration}s) ---")
        dataset_A = self.perform_single_joint_sweep(
            arm_side, sweep_joint_A, q_cand, -range_A, range_A, sweep_duration,
            q_head=None, label="Joint A", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint
        )
        if dataset_A is None:
            return None

        if getattr(self, 'stop_requested', False):
            return None
            
        if is_camera_mock:
            time.sleep(0.01)
        else:
            time.sleep(0.5)

        # 2. PHYSICAL SWEEP JOINT B
        if log_callback: log_callback(f"\n--- Commencing Continuous Sweep on Joint B (Index {sweep_joint_B}, duration={sweep_duration}s) ---")
        dataset_B = self.perform_single_joint_sweep(
            arm_side, sweep_joint_B, q_cand, -range_B, range_B, sweep_duration,
            q_head=None, label="Joint B", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint
        )
        if dataset_B is None:
            return None

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

        # Save FULL captured continuous sweep points to debug txt files before downsampling
        if save_debug:
            self.save_debug_points(
                arm_side, sweep_joint_A, dataset_A, initial_joint_pos, ee_name, dyn_model, None, "joint_A", log_callback
            )
            self.save_debug_points(
                arm_side, sweep_joint_B, dataset_B, initial_joint_pos, ee_name, dyn_model, None, "joint_B", log_callback
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
            log_callback=log_callback,
            cand_joint=cand_joint,
            sweep_joint_A=sweep_joint_A,
            sweep_joint_B=sweep_joint_B
        )

    def compute_calibration_results(self, arm_side, mode, dataset_A, dataset_B, initial_joint_pos, current_offset_deg=0.0, use_angle_based_fitting=None, save_debug=False, log_callback=None, cand_joint=None, sweep_joint_A=None, sweep_joint_B=None):
        from scipy.spatial.transform import Rotation as R_scipy
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        # Define nominal axes in parent link frame for each mode
        if mode == "wrist_roll_v13":
            a_cand_local = np.array([1.0, 0.0, 0.0])
            a_A_local = np.array([1.0, 0.0, 0.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        elif mode == "wrist_yaw2":
            a_cand_local = np.array([0.0, 0.0, 1.0])
            a_A_local = np.array([0.0, 0.0, 1.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        elif mode in ("wrist_pitch_v13", "elbow"):
            a_cand_local = np.array([0.0, 1.0, 0.0])
            a_A_local = np.array([0.0, 1.0, 0.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        elif mode in ("wrist_pitch", "elbow"):
            a_cand_local = np.array([0.0, 1.0, 0.0])
            a_A_local = np.array([0.0, 0.0, 1.0])
            a_B_local = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Get joint index configs
        jcfg = self.JOINT_CONFIGS[mode]
        cand_joint = jcfg["cand_joint"]
        sweep_joint_A = jcfg["sweep_joint_A"]
        sweep_joint_B = jcfg["sweep_joint_B"]
        
        state = self.robot.get_state() if self.robot else None
        model = self.robot.model() if self.robot else None
        arm_idx = (model.left_arm_idx if arm_side == "left" else model.right_arm_idx) if model else list(range(7))
        dyn_model = self.robot.get_dynamics() if self.robot else None
        ee_name = f"ee_{arm_side}"

        # Compute dynamic nominal axes using forward kinematics (FK) at the ready pose
        is_pure_mock = getattr(self.robot, "is_pure_mock", False) if self.robot else True
        if self.robot and not is_pure_mock:
            try:
                # Construct full q array at ready pose
                q_ready_full = np.array(state.position)
                for idx, val in zip(arm_idx, initial_joint_pos):
                    q_ready_full[idx] = val

                def get_link_name(j_idx):
                    return f"link_{arm_side}_arm_{j_idx}" if j_idx >= 0 else "link_torso_5"

                T_cand = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(cand_joint - 1))
                T_A = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(sweep_joint_A - 1))
                T_B = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(sweep_joint_B - 1))

                a_cand_t5 = T_cand[:3, :3] @ a_cand_local
                a_A_t5 = T_A[:3, :3] @ a_A_local
                a_B_t5 = T_B[:3, :3] @ a_B_local
                
                if log_callback:
                    log_callback(f"[INFO] Dynamically calculated nominal axes from FK (Arm: {arm_side}, Mode: {mode}):")
                    log_callback(f"       a_cand_t5 = {a_cand_t5.tolist()}")
                    log_callback(f"       a_A_t5    = {a_A_t5.tolist()}")
                    log_callback(f"       a_B_t5    = {a_B_t5.tolist()}")
            except Exception as e:
                if log_callback:
                    log_callback(f"[WARN] Failed to compute dynamic nominal axes from FK, falling back to constant defaults: {e}")
                a_cand_t5 = a_cand_local
                a_A_t5 = a_A_local
                a_B_t5 = a_B_local
        else:
            # Fallback for mock simulation mode
            a_cand_t5 = a_cand_local
            a_A_t5 = a_A_local
            a_B_t5 = a_B_local

        # 1. Use nominal fixed camera rotation relative to torso (ZYX [-90, 0, -90])
        # to avoid using uncalibrated mount_to_cam values or head kinematics.
        R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()

        # Define nominal axes in the camera frame using transpose of R_rob_to_cam (since R_rob_to_cam is R_cam_to_torso)
        a_cand_cam = R_rob_to_cam.T @ a_cand_t5
        a_A_cam = R_rob_to_cam.T @ a_A_t5
        a_B_cam_nom = R_rob_to_cam.T @ a_B_t5

        a_cand_cam /= np.linalg.norm(a_cand_cam)
        a_A_cam /= np.linalg.norm(a_A_cam)
        a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)

        # 2. Extract poses and angles in the camera frame
        poses_A = [pose for _, pose in dataset_A]
        angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, _ in dataset_A]
        poses_B = [pose for _, pose in dataset_B]
        angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, _ in dataset_B]

        # 3. Fit Sweep A and B axes in the camera frame
        res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_A, angles_A, axis_prior=a_A_cam)
        res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_B, angles_B, axis_prior=a_B_cam_nom)

        n_A = res_A['axis_opt']
        n_B = res_B['axis_opt']
        r_A = res_A['radius']
        r_B = res_B['radius']
        rmse_A = res_A['rmse']
        rmse_B = res_B['rmse']
        c_A_c = res_A['c_opt']
        c_B_c = res_B['c_opt']

        pts_a_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_A])
        pts_b_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_B])

        # Compute center distance in camera frame
        diff_centers = c_B_c - c_A_c
        center_dist = np.linalg.norm(diff_centers - np.dot(diff_centers, n_A) * n_A)
        
        # Enforce n_A and n_B direction using the time-series trajectory direction (start -> end sweep)
        # to ensure that the fitted normal vector aligns with the positive joint rotation direction.
        v_A = pts_a_cam - c_A_c
        u_A = pts_a_cam[1:] - pts_a_cam[:-1]
        mean_cross_A = np.mean(np.cross(v_A[:-1], u_A), axis=0)
        n_A = n_A if np.dot(n_A, mean_cross_A) > 0 else -n_A

        v_B = pts_b_cam - c_B_c
        u_B = pts_b_cam[1:] - pts_b_cam[:-1]
        mean_cross_B = np.mean(np.cross(v_B[:-1], u_B), axis=0)
        n_B = n_B if np.dot(n_B, mean_cross_B) > 0 else -n_B

        angle_between_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))

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
            
        if mode in ("wrist_roll_v13", "wrist_yaw2"):
            # Find index in dataset_A where candidate joint (e.g. J7) is closest to 0.0 (ready pose)
            idx_A_mid = np.argmin([abs(q_full[arm_idx[cand_joint]]) for q_full, _ in dataset_A])
            p_A_ready = pts_a_cam[idx_A_mid]
            
            # Find index in dataset_B where J6 (sweep_joint_B) is closest to 0.0 (ready pose at staged offset)
            idx_B_mid = np.argmin([abs(q_full[arm_idx[sweep_joint_B]]) for q_full, _ in dataset_B])
            R_cam_to_marker_ready = dataset_B[idx_B_mid][1][:3, :3]
            n_A_marker = R_cam_to_marker_ready.T @ n_A
            n_B_marker = R_cam_to_marker_ready.T @ n_B
            
            ver_key = "1.3" if self.is_v13() else "1.2"
            nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
            nominal_rpy = nominal_vec[3:6]
            
            R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
            z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
            y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
            x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
            
            if mode == "wrist_yaw2":
                if np.dot(n_A_marker, z_ee_m_ideal) < 0:
                    n_A_marker = -n_A_marker
                if np.dot(n_B_marker, y_ee_m_ideal) < 0:
                    n_B_marker = -n_B_marker
                z_col = n_A_marker
                y_col = n_B_marker - np.dot(n_B_marker, z_col) * z_col
                y_col /= np.linalg.norm(y_col)
                x_col = np.cross(y_col, z_col)
            else:
                if np.dot(n_A_marker, x_ee_m_ideal) < 0:
                    n_A_marker = -n_A_marker
                if np.dot(n_B_marker, y_ee_m_ideal) < 0:
                    n_B_marker = -n_B_marker
                x_col = n_A_marker
                y_col = n_B_marker - np.dot(n_B_marker, x_col) * x_col
                y_col /= np.linalg.norm(y_col)
                z_col = np.cross(x_col, y_col)
                
            R_ee_m_actual = np.column_stack((x_col, y_col, z_col)).T
            euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
            yaw_e, pitch_e, roll_e = euler_deg
            if arm_side == "right" and yaw_e < 0:
                yaw_e += 360.0
                
            if mode == "wrist_yaw2":
                diff_angle = np.radians(nominal_rpy[2] - yaw_e)
            else:
                diff_angle = np.radians(nominal_rpy[0] - roll_e)
                
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            
            v_marker = p_A_ready - c_B_c
            axial_offset_mm = np.dot(v_marker, n_A)
            lateral_offset_mm = np.linalg.norm(v_marker - axial_offset_mm * n_A)
        else:
            actual_angle = np.arctan2(np.dot(np.cross(n_A_proj, n_B_proj), a_cand_cam), np.dot(n_A_proj, n_B_proj))
            diff_angle = actual_angle - nominal_angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            axial_offset_mm = float('nan')
            lateral_offset_mm = float('nan')

        # Compute 3D shortest distance between axes
        perp_dist = self.compute_shortest_distance_between_lines(c_A_c, n_A, c_B_c, n_B)

        nominal_dist_35 = 235.0
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

        size_error = abs(r_A - r_B)
        is_v13 = self.is_v13()
        # For wrist_pitch_v13/v1.3-elbow, center_dist is around forearm length (~235mm), not 0.
        # Since sweep A and B are at different distances from the marker in most modes, size_error is expected to be large.
        if mode in ("wrist_pitch_v13", "wrist_roll_v13", "elbow_v13") or (mode == "elbow" and is_v13):
            is_large_error = (center_dist > 350.0)
        elif mode == "wrist_yaw2":
            is_large_error = (center_dist > 150.0)
        elif mode == "elbow": # v1.2 elbow
            is_large_error = (center_dist > 150.0 or size_error > 100.0)
        else:
            is_large_error = (center_dist > 150.0)
        
        if is_large_error:
            if log_callback:
                log_callback(f"[ERROR] Circle fitting failed or error is too large (center_dist={center_dist:.2f} mm, size_error={size_error:.2f} mm). Aborting step adjustment.")
            optimal_offset_deg = 0.0
        else:
            damping = 0.95
            if mode in ("wrist_roll_v13", "wrist_yaw2"):
                chirality_sign = 1.0
            else:
                R_epsilon = R_scipy.from_rotvec(np.radians(1.0) * a_cand_cam).as_matrix()
                a_B_cam_eps = R_epsilon @ a_B_cam_nom
                a_B_proj_eps = a_B_cam_eps - np.dot(a_B_cam_eps, a_cand_cam) * a_cand_cam
                if np.linalg.norm(a_B_proj_eps) > 1e-6:
                    a_B_proj_eps /= np.linalg.norm(a_B_proj_eps)
                cross_nominal = np.cross(a_A_proj, a_B_proj_eps)
                chirality_sign = np.sign(np.dot(cross_nominal, a_cand_cam))
                if chirality_sign == 0.0:
                    chirality_sign = 1.0
            optimal_offset_deg = chirality_sign * np.degrees(diff_angle) * damping

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()})")
            log_callback("="*50)
            log_callback(f"  * Camera Circle Normals Angle (Reference): {angle_between_normals:.4f} deg")
            log_callback(f"  * Circle Size Error (abs: r_A - r_B)     : {size_error:.4f} mm")
            if mode == "wrist_pitch_v13":
                log_callback(f"  * Forearm Length (Shortest 3D Dist)      : {perp_dist:.4f} mm (Nom: {nominal_dist_35:.2f} mm)")
            else:
                log_callback(f"  * Estimated Circle Center Distance       : {center_dist:.4f} mm")
            log_callback(f"  * Calculated Offset Correction           : {optimal_offset_deg:.6f} deg")
            log_callback("="*50)

        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
            'recommended_joint_offset': optimal_offset_deg,
            'diff_angle_deg': np.degrees(diff_angle),
            'converged': False,
            '_dataset_A': dataset_A,
            '_dataset_B': dataset_B,
            '_initial_joint_pos': initial_joint_pos,
            'angle_between_normals': angle_between_normals,
            'center_dist': center_dist,
            'perp_dist_before': perp_dist,
            'perp_dist_after': perp_dist,
            'axial_offset_mm': axial_offset_mm,
            'lateral_offset_mm': lateral_offset_mm,
            'nominal_dist_35': nominal_dist_35,
            'r_A': r_A,
            'r_B': r_B,
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
                'perp_dist_before': perp_dist,
                'perp_dist_after': perp_dist,
            },
        }
