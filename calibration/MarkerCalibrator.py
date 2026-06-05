import time
import logging
import os
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from CalibratorBase import BaseCalibrator

class MarkerCalibrator(BaseCalibrator):
    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False):
        points = np.array([T[:3, 3] * 1000.0 for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # Initial Center and Normal estimation using unified circle fit
        c_fit, R_fit, radius_fit, rmse_fit, _, _, _ = BaseCalibrator.fit_circle_3d(points)
        
        # Nominal Normal
        if axis_prior is not None:
            n_nominal = np.array(axis_prior, dtype=float)
            n_nominal /= np.linalg.norm(n_nominal)
        else:
            n_nominal = np.array([0.0, 0.0, 1.0])
            
        # Initial Normal
        if axis_prior is not None:
            best_normal = n_nominal.copy()
        else:
            best_normal = R_fit[:, 2] # Normal is the Z-axis of R_fit
            
        centroid = np.mean(points, axis=0)
        ex = R_fit[:, 0]
        ey = R_fit[:, 1]

        # Sagitta formula
        C_chord_vec = points[-1] - points[0]
        C_chord = np.linalg.norm(C_chord_vec)
        p_mid_chord = (points[0] + points[-1]) / 2.0
        p_mid_arc = points[len(points) // 2]
        v_sag = p_mid_arc - p_mid_chord
        H_sag = np.linalg.norm(v_sag)
        
        if H_sag > 0.05 and C_chord > 1.0:
            R_geom = (C_chord ** 2) / (8.0 * H_sag) + H_sag / 2.0
        else:
            R_geom = 280.0 if (axis_prior is not None and abs(axis_prior[2]) > 0.8) else 75.0
            
        R_init = np.clip(R_geom, 50.0, 450.0)
        
        if 50.0 <= R_geom <= 450.0 and H_sag > 0.05:
            u_sag = v_sag / H_sag
            c_init = p_mid_arc - R_init * u_sag
        else:
            R_init = np.clip(radius_fit, 50.0, 450.0)
            c_init = c_fit
        
        best_opt = None
        best_rmse = float('inf')
        best_sign = 1
        
        for sign in [1, -1]:
            angles_rad = angles_rad_base * sign
            r_dir_init = points[0] - c_init
            r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
            if np.linalg.norm(r_dir_init) > 1e-6:
                r_dir_init /= np.linalg.norm(r_dir_init)
            
            init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-1.5, -1.5, -1.5], [-1.5, -1.5, -1.5], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [450.0]])
            
            def total_residuals(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    
                r_init = params[6:9]
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                residuals = []
                for pt, theta in zip(points, angles_rad):
                    pred_pt = c + R * MarkerCalibrator.rodrigues_rotation(r_init, axis, theta)
                    residuals.extend(pt - pred_pt)
                return np.array(residuals)
                
            opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            rmse = np.sqrt(np.mean(opt_res.fun**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_opt = opt_res
                best_sign = sign

        # Extract optimal
        c_init = best_opt.x[0:3]
        best_normal = best_opt.x[3:6]
        best_normal /= np.linalg.norm(best_normal)
        r_final_dir = best_opt.x[6:9]
        r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
        if np.linalg.norm(r_final_dir) > 1e-6:
            r_final_dir /= np.linalg.norm(r_final_dir)
        R_init = best_opt.x[9]
        
        # Worst-outlier rejection
        inlier_mask = np.ones(len(points), dtype=bool)
        for out_iter in range(3):
            angles_rad = angles_rad_base * best_sign
            pts_in = points[inlier_mask]
            rad_in = angles_rad[inlier_mask]
            
            if len(pts_in) < 6:
                break
                
            init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-1.5, -1.5, -1.5], [-1.5, -1.5, -1.5], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [450.0]])
            
            def total_residuals_in(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                r_init = params[6:9]
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                residuals = []
                for pt, theta in zip(pts_in, rad_in):
                    pred_pt = c + R * MarkerCalibrator.rodrigues_rotation(r_init, axis, theta)
                    residuals.extend(pt - pred_pt)
                return np.array(residuals)
                
            opt_res = least_squares(total_residuals_in, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            c_init = opt_res.x[0:3]
            best_normal = opt_res.x[3:6]
            best_normal /= np.linalg.norm(best_normal)
            r_final_dir = opt_res.x[6:9]
            r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
            if np.linalg.norm(r_final_dir) > 1e-6:
                r_final_dir /= np.linalg.norm(r_final_dir)
            R_init = opt_res.x[9]
            
            all_errors = []
            for pt, theta in zip(points, angles_rad):
                pred_pt = c_init + R_init * MarkerCalibrator.rodrigues_rotation(r_final_dir, best_normal, theta)
                all_errors.append(np.linalg.norm(pt - pred_pt))
            all_errors = np.array(all_errors)
            
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.5:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final Optimization
        pts_in = points[inlier_mask]
        rad_in = angles_rad_base[inlier_mask] * best_sign
        init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
        
        def total_residuals_final(params):
            c = params[0:3]
            axis = params[3:6]
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            r_init = params[6:9]
            r_init -= np.dot(r_init, axis) * axis
            r_init_norm = np.linalg.norm(r_init)
            if r_init_norm > 1e-6:
                r_init = r_init / r_init_norm
            R = params[9]
            
            residuals = []
            for pt, theta in zip(pts_in, rad_in):
                pred_pt = c + R * MarkerCalibrator.rodrigues_rotation(r_init, axis, theta)
                residuals.extend(pt - pred_pt)
            return np.array(residuals)
            
        lower_bounds = np.hstack([c_init - 200.0, [-1.5, -1.5, -1.5], [-1.5, -1.5, -1.5], [50.0]])
        upper_bounds = np.hstack([c_init + 200.0, [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [450.0]])
        
        opt_res = least_squares(total_residuals_final, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
        c_opt = opt_res.x[0:3]
        axis_opt = opt_res.x[3:6]
        axis_opt /= np.linalg.norm(axis_opt)
        
        r_init_opt = opt_res.x[6:9]
        r_init_opt -= np.dot(r_init_opt, axis_opt) * axis_opt
        r_init_opt /= np.linalg.norm(r_init_opt)
        radius_opt = opt_res.x[9]
        
        rmse = np.sqrt(np.mean(opt_res.fun**2))
        
        # Coordinate frames for plotting
        if axis_opt[0] < 0.9:
            ex = np.cross(axis_opt, [1, 0, 0])
        else:
            ex = np.cross(axis_opt, [0, 1, 0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(axis_opt, ex)
        
        pts_centered = points - c_opt
        pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
        uc_opt = 0.0
        vc_opt = 0.0
        
        # Calculate 6-DOF misalignment
        tilt_angle = np.rad2deg(np.arccos(np.clip(np.dot(axis_opt, n_nominal), -1.0, 1.0)))
        
        # Projection for yaw
        n_proj = axis_opt - np.dot(axis_opt, n_nominal) * n_nominal
        if np.linalg.norm(n_proj) > 1e-6:
            n_proj /= np.linalg.norm(n_proj)
            if n_nominal[2] > 0.8:
                yaw_angle = np.rad2deg(np.arctan2(n_proj[1], n_proj[0]))
            else:
                yaw_angle = 0.0
        else:
            yaw_angle = 0.0
            
        # Calculate individual tilt angles for each pose to compute jitter/stddev
        tilt_list = []
        for T in relative_poses:
            if axis_prior is not None:
                if abs(axis_prior[0]) > 0.8:
                    axis_i = T[:3, 0]
                elif abs(axis_prior[1]) > 0.8:
                    axis_i = T[:3, 1]
                else:
                    axis_i = T[:3, 2]
            else:
                axis_i = T[:3, 2]
            axis_norm = np.linalg.norm(axis_i)
            if axis_norm > 1e-6:
                axis_i /= axis_norm
            tilt_i = np.rad2deg(np.arccos(np.clip(np.dot(axis_i, n_nominal), -1.0, 1.0)))
            tilt_list.append(tilt_i)
            
        res_dict = {
            'c_opt': c_opt,
            'axis_opt': axis_opt,
            'radius': radius_opt,
            'rmse': rmse,
            'tilt': tilt_angle,
            'yaw': yaw_angle,
            'pts_2d': pts_2d,
            'uc_opt': uc_opt,
            'vc_opt': vc_opt,
            'inlier_mask': inlier_mask,
            'tilt_list': tilt_list
        }
        return res_dict

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm)...")
        
        # Load mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_rob_to_cam = self.make_transform(mount_to_cam)
        T_target_cam = np.eye(4)
        T_target_cam[2, 3] = target_dist / 1000.0 # to meters

        for attempt in range(5):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move canceled by user.")
                self.robot.cancel_control()
                break
                
            if log_callback: log_callback(f"[Attempt {attempt + 1}/5] Capturing marker pose...")
            time.sleep(1.0)
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not res:
                if log_callback: log_callback("  [ERROR] Marker not visible.")
                return False
            
            if isinstance(res, list):
                T_cam_to_marker = np.array(res[0]).reshape(4, 4)
            else:
                T_cam_to_marker = np.array(list(res.values())[0]).reshape(4, 4)
                
            cam_pos = T_cam_to_marker[:3, 3]
            cam_rot = T_cam_to_marker[:3, :3]
            
            pos_err_mm = np.linalg.norm(cam_pos - T_target_cam[:3, 3]) * 1000.0
            rot_err_mat = cam_rot.T @ T_target_cam[:3, :3]
            rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
            err_norm = np.linalg.norm([pos_err_mm, rot_err_deg])

            if log_callback:
                log_callback(f"  Current: X={cam_pos[0]*1000:.1f}, Y={cam_pos[1]*1000:.1f}, Z={cam_pos[2]*1000:.1f} mm")
                log_callback(f"  Error Norm: {err_norm:.2f} (Pos:{pos_err_mm:.1f}mm, Ang:{rot_err_deg:.1f}deg)")

            if err_norm <= 0.5:
                if log_callback: log_callback(f"  [SUCCESS] Reached center aligned pose! (Norm: {err_norm:.2f})")
                break

            if log_callback: log_callback("  Calculating joint command and moving...")
            T_rob_to_marker = T_rob_to_cam @ T_cam_to_marker
            T_rob_to_marker_target = T_rob_to_cam @ T_target_cam
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            T_rob_to_ee_new = T_rob_to_marker_target @ np.linalg.inv(T_rob_to_marker) @ T_rob_to_ee
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("link_torso_5", ee_name, T_rob_to_ee_new, 0.2, 0.5, 1.0)
            cb.set_stop_orientation_tracking_error(1e-4)
            cb.set_stop_position_tracking_error(1e-3)
            
            body_cmd = rby.BodyComponentBasedCommandBuilder()
            if arm_side == "right":
                body_cmd.set_right_arm_command(cb)
            else:
                body_cmd.set_left_arm_command(cb)
                
            rc = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
            )
            rv = self.robot.send_command(rc, 10).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                if log_callback: log_callback(f"  [ERROR] Failed to move: {rv.finish_code}")
                return False
            time.sleep(0.5)
        return True

    def perform_move_to_ready_pose(self, arm_side, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to Marker Ready Pose...")
        torso = [0, 0, 0, 0, 0, 0]
        
        # 1. First move the inactive arm to zero pose to avoid collision
        if log_callback: log_callback("[INFO] Moving inactive arm to zero pose first...")
        if arm_side == "right":
            success_other = self.movej(self.robot, left_arm=[0.0]*7, head=None, minimum_time=3.0)
        else:
            success_other = self.movej(self.robot, right_arm=[0.0]*7, head=None, minimum_time=3.0)
            
        if not success_other:
            if log_callback: log_callback("[ERROR] Failed to move inactive arm to zero pose.")
            return False
            
        # 2. Move active arm and head/torso to ready pose
        if log_callback: log_callback("[INFO] Moving active arm, torso, and head to ready pose...")
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

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True):
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
            log_callback(f"   STARTING {str(axis_mode).upper()} CONTINUOUS MARKER SWEEP")
            log_callback("="*50)
            
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return None
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return None

        # Pre-check marker visibility
        initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
        if not initial_check:
            if log_callback: log_callback("[ERROR] Marker is not visible in ready pose.")
            if status_callback: status_callback(False)
            return None
        if status_callback: status_callback(True)

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

        # Sweep configuration
        axis_str = str(axis_mode).lower()
        if "6" in axis_str:
            start_deg = -20.0
            end_deg = 20.0
            joint_i = 6
        else:
            start_deg = -10.0
            end_deg = 10.0
            joint_i = 5

        # Head index and active head tracking setup
        head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
        q_head_0 = state.position[head_idx].copy() if head_idx is not None else None
        dyn_model = self.robot.get_dynamics()
        
        # Compute start and end arm poses
        q_start_arm = list(initial_joint_pos)
        q_start_arm[joint_i] = initial_joint_pos[joint_i] + np.radians(start_deg)
        q_end_arm = list(initial_joint_pos)
        q_end_arm[joint_i] = initial_joint_pos[joint_i] + np.radians(end_deg)
        
        q_full_start = np.array(state.position)
        q_full_end = np.array(state.position)
        if arm_side == "left":
            q_full_start[model.left_arm_idx] = q_start_arm
            q_full_end[model.left_arm_idx] = q_end_arm
        else:
            q_full_start[model.right_arm_idx] = q_start_arm
            q_full_end[model.right_arm_idx] = q_end_arm

        ee_name = f"ee_{arm_side}"
        q_head_start = None
        q_head_end = None
        
        if use_head_tracking and head_idx is not None and q_head_0 is not None:
            try:
                T_ee_start = self.compute_fk(self.robot, dyn_model, q_full_start, ee_name, "link_torso_5")
                p_marker_start = T_ee_start[:3, 3]
                
                T_neck = self.compute_fk(self.robot, dyn_model, state.position, "link_head_2", "link_torso_5")
                p_neck = T_neck[:3, 3]
                
                T_ee_0 = self.compute_fk(self.robot, dyn_model, state.position, ee_name, "link_torso_5")
                p_marker_0 = T_ee_0[:3, 3]
                
                v_0 = p_marker_0 - p_neck
                v_start = p_marker_start - p_neck
                
                yaw_geo_0 = np.arctan2(v_0[1], v_0[0])
                pitch_geo_0 = np.arctan2(v_0[2], np.sqrt(v_0[0]**2 + v_0[1]**2))
                
                yaw_geo_start = np.arctan2(v_start[1], v_start[0])
                pitch_geo_start = np.arctan2(v_start[2], np.sqrt(v_start[0]**2 + v_start[1]**2))
                
                yaw_target_start = np.clip(q_head_0[0] + (yaw_geo_start - yaw_geo_0), -np.radians(25.0), np.radians(25.0))
                pitch_target_start = np.clip(q_head_0[1] - (pitch_geo_start - pitch_geo_0), -np.radians(20.0), np.radians(20.0))
                q_head_start = np.array([yaw_target_start, pitch_target_start])

                T_ee_end = self.compute_fk(self.robot, dyn_model, q_full_end, ee_name, "link_torso_5")
                p_marker_end = T_ee_end[:3, 3]
                v_end = p_marker_end - p_neck
                
                yaw_geo_end = np.arctan2(v_end[1], v_end[0])
                pitch_geo_end = np.arctan2(v_end[2], np.sqrt(v_end[0]**2 + v_end[1]**2))
                
                yaw_target_end = np.clip(q_head_0[0] + (yaw_geo_end - yaw_geo_0), -np.radians(25.0), np.radians(25.0))
                pitch_target_end = np.clip(q_head_0[1] - (pitch_geo_end - pitch_geo_0), -np.radians(20.0), np.radians(20.0))
                q_head_end = np.array([yaw_target_end, pitch_target_end])
            except Exception as e:
                if log_callback: log_callback(f"[WARN] Failed to compute target head angles: {e}")
                q_head_start = q_head_0
                q_head_end = q_head_0
        else:
            q_head_start = None
            q_head_end = None

        # 1. Move to start position (-20 or -10 deg)
        if log_callback: log_callback(f"[INFO] Moving to start sweep position...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=q_start_arm, head=q_head_start, minimum_time=2.5, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=q_start_arm, head=q_head_start, minimum_time=2.5, apply_offsets=False)
        time.sleep(1.0)

        # 2. Continuous sweep from start to end position (30s duration)
        if log_callback: log_callback(f"[INFO] Commencing Continuous Sweep on Marker Axis {axis_mode} (duration=30s)...")
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_arm if arm_side == "right" else None,
            left_arm=q_end_arm if arm_side == "left" else None,
            head=q_head_end, minimum_time=30.0
        )
        
        captured_poses = []
        captured_angles = []
        
        move_thread.start()
        
        # High speed data collection
        while move_thread.is_alive():
            lpf_results = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if lpf_results:
                pose = np.array(lpf_results[0]).reshape(4, 4) if isinstance(lpf_results, list) else np.array(list(lpf_results.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                q_captured = q_full_captured[arm_idx[joint_i]]
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    captured_poses.append(pose)
                    captured_angles.append(np.degrees(q_captured - initial_joint_pos[joint_i]))
            time.sleep(0.01) # 100Hz polling

        move_thread.join()
        if log_callback: log_callback(f"    -> Swept {len(captured_poses)} dense raw coordinate frames.")

        # Return arm and head to original ready pose
        if log_callback: log_callback("\n[INFO] Sweep complete. Returning to initial ready pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.5, apply_offsets=False)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.5, apply_offsets=False)

        if len(captured_poses) < 10:
            if log_callback: log_callback("[ERROR] Too few valid marker poses. Calibration failed.")
            return None

        # Solve Circle Fitting
        n_nom = [1.0, 0.0, 0.0] if "6" in str(axis_mode).lower() else [0.0, 1.0, 0.0]
        res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom)
        return res

    def get_link_length(self, arm_side):
        if not self.robot or self.robot == "mock_robot":
            return 300.0
        try:
            dyn_model = self.robot.get_dynamics()
            names = self.robot.model().robot_joint_names
            state = dyn_model.make_state(
                [f"link_{arm_side}_arm_5", f"ee_{arm_side}"],
                names
            )
            state.set_q(self.robot.get_state().position)
            dyn_model.compute_forward_kinematics(state)
            T = dyn_model.compute_transformation(state, 0, 1)
            return np.linalg.norm(T[:3, 3]) * 1000.0 # m to mm
        except Exception as e:
            logging.warning(f"Failed to get link kinematics: {e}")
            return 300.0

    def compute_unified_bracket_calibration(self, marker_data_5, marker_data_6, arm_side):
        L_5_ee = self.get_link_length(arm_side)

        z_e_in_m = marker_data_6['axis']
        y_e_in_m = marker_data_5['axis']
        
        if arm_side == "left":
            ideal_rpy = [90.0, 0.0, 0.0]
        else:
            ideal_rpy = [90.0, 0.0, 180.0]
            
        T_ee_m_ideal = self.make_transform([0, 0, 0] + ideal_rpy)
        R_ee_m_ideal = T_ee_m_ideal[:3, :3]
        
        # Check directions
        if np.dot(z_e_in_m, R_ee_m_ideal[:, 2]) < 0: z_e_in_m = -z_e_in_m
        if np.dot(y_e_in_m, R_ee_m_ideal[:, 1]) < 0: y_e_in_m = -y_e_in_m
        
        # Orthogonalize
        y_e_in_m = y_e_in_m - np.dot(y_e_in_m, z_e_in_m) * z_e_in_m
        y_e_in_m /= np.linalg.norm(y_e_in_m)
        x_e_in_m = np.cross(y_e_in_m, z_e_in_m)
        
        R_ee_m_actual = np.column_stack((x_e_in_m, y_e_in_m, z_e_in_m))
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        radius_6 = marker_data_6['radius']
        radius_5 = marker_data_5['radius']
        
        # Offset translation
        x_e = 0.0
        y_e = radius_6 if arm_side == "left" else -radius_6
        z_e = -abs(radius_5 - L_5_ee)
        
        # Analysis Orthogonality / Quality metrics
        dot_val = np.dot(z_e_in_m, y_e_in_m)
        angle_between = np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0)))
        ortho_err = abs(90.0 - angle_between)
        
        rmse_6 = marker_data_6['rmse']
        rmse_5 = marker_data_5['rmse']
        
        return {
            'x_e': x_e, 'y_e': y_e, 'z_e': z_e,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee, 'radius_6': radius_6, 'radius_5': radius_5,
            'ortho_err': ortho_err, 'rmse_6': rmse_6, 'rmse_5': rmse_5
        }
