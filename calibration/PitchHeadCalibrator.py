import time
import logging
import numpy as np
import rby1_sdk as rby
import math
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
import yaml
import os

class PitchHeadCalibrator:
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot
        self.setting_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "setting.yaml")
        self.camera_config = self.load_camera_config()

    def load_camera_config(self):
        try:
            with open(self.setting_yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)
                return cfg.get("camera", {})
        except Exception as e:
            logging.error(f"Failed to load camera settings: {e}")
            return {}

    @staticmethod
    def make_transform(data):
        """
        Creates a 4x4 transformation matrix from [x, y, z, roll, pitch, yaw].
        Coordinates in meters, angles in degrees (ZYX Euler).
        """
        T = np.eye(4)
        T[:3, 3] = data[:3]
        yaw = data[5]
        pitch = data[4]
        roll = data[3]
        T[:3, :3] = R_scipy.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        return T

    @staticmethod
    def compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        model = robot.model()
        state = dyn_model.make_state([base_link, ee_link], model.robot_joint_names)
        state.set_q(q)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        return T

    @staticmethod
    def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0, priority=10):
        if not robot:
            return False
            
        cmd = rby.ComponentBasedCommandBuilder()
        
        has_body = False
        body_cmd = rby.BodyComponentBasedCommandBuilder()
        if torso is not None:
            body_cmd.set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(torso)
            )
            has_body = True
        if right_arm is not None:
            body_cmd.set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
            has_body = True
        if left_arm is not None:
            body_cmd.set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_arm)
            )
            has_body = True

        if has_body:
            cmd.set_body_command(body_cmd)

        if head is not None:
            cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )

        rv = robot.send_command(
            rby.RobotCommandBuilder().set_command(cmd),
            priority,
        ).get()

        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            logging.error(f"Failed to conduct movej. Finish code: {rv.finish_code}")
            return False

        return True

    @staticmethod
    def fit_circle_3d(points):
        """
        Fits a 3D circle to points. Returns (center_3d, normal, radius, rmse, pts_2d, uc, vc, ex, ey)
        """
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        pts_centered = points - centroid
        
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        ex = vh[0, :]
        ey = vh[1, :]
        pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)

        A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
        b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        uc, vc = res[0], res[1]
        radius = np.sqrt(max(0, res[2] + uc**2 + vc**2))
        
        # Refine using circle residuals
        def residuals(params):
            u, v, R = params
            return np.sqrt((pts_2d[:, 0] - u)**2 + (pts_2d[:, 1] - v)**2) - R
            
        opt = least_squares(residuals, [uc, vc, radius], loss='huber')
        uc_opt, vc_opt, R_opt = opt.x
        rmse = np.sqrt(np.mean(opt.fun**2))
        center_3d = centroid + uc_opt * ex + vc_opt * ey
        return center_3d, normal, R_opt, rmse, pts_2d, uc_opt, vc_opt, ex, ey

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system (marker_st) is not initialized.")
            return False

        if log_callback:
            log_callback("\n" + "="*40)
            log_callback("   STARTING MOVE TO CENTER & ALIGN")
            log_callback("="*40)

        # Transformation from Camera to Robot (link_torso_5)
        T_rob_to_cam = self.make_transform([0, 0, 0, -90, 0, -90])
        
        # Target pose of marker in camera frame
        T_target_cam = np.eye(4)
        T_target_cam[:3, 3] = [0, 0, target_dist / 1000.0]

        for attempt in range(5): 
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move to Center cancelled by user.")
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
                if log_callback: log_callback(f"  [SUCCESS] Reached target pose! (Norm: {err_norm:.2f})")
                break

            if log_callback: log_callback("  Calculating and moving to correct pose...")
            
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
                if log_callback: log_callback(f"  [ERROR] Failed to move Cartesian: {rv.finish_code}")
                return False
                
            time.sleep(0.5)

        if log_callback: log_callback("Move to Center & Align finished.\n")
        return True

    def perform_move_to_ready_pose(self, arm_side, mode, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving to {arm_side} Pitch/Head Ready Pose (Mode: {mode})...")

        torso = [0, 0, 0, 0, 0, 0]
        
        if mode == "wrist_pitch":
            # Joint 5 (wrist_pitch) is 0
            if arm_side == "right":
                right_arm = np.deg2rad([-55, -45, 25, -127, 90, 0, 0])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-55, 45, -25, -127, -90, 0, 0])
        elif mode == "elbow":
            # Joint 3 (elbow) is 0, Joint 5 (wrist_pitch) is 90
            if arm_side == "right":
                right_arm = np.deg2rad([-107, -17, 0, 0, 73, -90, -107])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-90, 45, -73, 0, -90, 90, 0])
        else: # head mode (standard park/ready pose)
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

    def perform_calibration_sweep_5_or_3(self, arm_side, mode, log_callback=None, status_callback=None, use_head_tracking=True):
        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} OFFSET CALIBRATION SWEEP")
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

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

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

        # Define sweep axes based on mode
        if mode == "wrist_pitch":
            # We calibrate Joint 5. Sweep candidates of Joint 5: -4, -2, 0, 2, 4 deg
            cand_joint = 5
            cand_offsets = [-4, -2, 0, 2, 4]
            # Sweep joint A (index 4) and joint B (index 6) to fit two circles
            sweep_joint_A = 4
            sweep_joint_B = 6
        else: # elbow mode
            # We calibrate Joint 3. Sweep candidates of Joint 3: -4, -2, 0, 2, 4 deg
            cand_joint = 3
            cand_offsets = [-4, -2, 0, 2, 4]
            # Sweep joint A (index 2) and joint B (index 4)
            sweep_joint_A = 2
            sweep_joint_B = 4

        sweep_angles = [-20, -10, 0, 10, 20]
        results = []

        for c_idx, offset_deg in enumerate(cand_offsets):
            if log_callback:
                log_callback(f"\n--- [Candidate Step {c_idx+1}/{len(cand_offsets)}] Candidate Joint Offset: {offset_deg} deg ---")
            
            # 1. Set Candidate joint angle
            q_cand = list(initial_joint_pos)
            q_cand[cand_joint] = initial_joint_pos[cand_joint] + np.radians(offset_deg)
            
            # First sweep (Joint A)
            pts_A = []
            for sa in sweep_angles:
                q_sweep = list(q_cand)
                q_sweep[sweep_joint_A] = q_cand[sweep_joint_A] + np.radians(sa)
                
                # Active head tracking computation
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
                    self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=1.0)
                else:
                    self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=1.0)
                time.sleep(0.5)
                
                res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
                if res:
                    pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                    pts_A.append(pose[:3, 3] * 1000.0) # mm
            
            # Return to candidate baseline pose before next sweep
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

            if arm_side == "left":
                self.movej(self.robot, left_arm=q_cand, head=head_q_step_cand, minimum_time=1.2)
            else:
                self.movej(self.robot, right_arm=q_cand, head=head_q_step_cand, minimum_time=1.2)
            time.sleep(0.5)

            # Second sweep (Joint B)
            pts_B = []
            for sb in sweep_angles:
                q_sweep = list(q_cand)
                q_sweep[sweep_joint_B] = q_cand[sweep_joint_B] + np.radians(sb)
                
                # Active head tracking computation
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
                    self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=1.0)
                else:
                    self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=1.0)
                time.sleep(0.5)
                
                res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
                if res:
                    pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                    pts_B.append(pose[:3, 3] * 1000.0) # mm

            # Return to candidate pose
            if arm_side == "left":
                self.movej(self.robot, left_arm=q_cand, head=head_q_step_cand, minimum_time=1.2)
            else:
                self.movej(self.robot, right_arm=q_cand, head=head_q_step_cand, minimum_time=1.2)
            time.sleep(0.5)

            if len(pts_A) >= 3 and len(pts_B) >= 3:
                # Fit 3D circles
                c3d_A, _, r_A, rmse_A, _, _, _, _, _ = self.fit_circle_3d(pts_A)
                c3d_B, _, r_B, rmse_B, _, _, _, _, _ = self.fit_circle_3d(pts_B)
                
                dist_error = np.linalg.norm(c3d_A - c3d_B)
                if log_callback:
                    log_callback(f"  * fitted Circle A (rmse={rmse_A:.3f}), B (rmse={rmse_B:.3f})")
                    log_callback(f"  * Center A: {np.round(c3d_A, 2)}, Center B: {np.round(c3d_B, 2)}")
                    log_callback(f"  * Center-to-Center Distance Error: {dist_error:.3f} mm")
                results.append((offset_deg, dist_error, pts_A, pts_B))
            else:
                if log_callback: log_callback("  [ERROR] Marker tracking failed during sweep.")
                return None

        # Return arm and head to original pose
        if log_callback: log_callback("\n[INFO] Sweep finished. Returning arm and head to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)

        # Parabolic fitting of errors to find minimum
        offsets = [r[0] for r in results]
        errors = [r[1] for r in results]
        
        # Fit E = a * x^2 + b * x + c
        p_coeff = np.polyfit(offsets, errors, 2)
        a, b, c = p_coeff
        
        optimal_offset = -b / (2 * a) if a > 0 else 0.0
        # Restrict offset estimation to safe boundaries
        optimal_offset = np.clip(optimal_offset, -10.0, 10.0)

        if log_callback:
            log_callback("\n" + "="*40)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()})")
            log_callback("="*40)
            log_callback(f"  * Fitted Curve coefficients: a={a:.5f}, b={b:.5f}, c={c:.5f}")
            log_callback(f"  * Estimated Optimal Offset : {optimal_offset:.3f} deg")
            log_callback("="*40)

        # Prepare optimal fitting datasets for visualization
        # Find index closest to optimal or the 0 deg run
        best_idx = np.argmin(np.abs(np.array(offsets) - optimal_offset))
        _, _, pts_A_best, pts_B_best = results[best_idx]
        
        c3d_A, normal_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A, ex_A, ey_A = self.fit_circle_3d(pts_A_best)
        c3d_B, normal_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B, ex_B, ey_B = self.fit_circle_3d(pts_B_best)

        return {
            'mode': mode,
            'optimal_offset': optimal_offset,
            'offsets': offsets,
            'errors': errors,
            'pts_2d_A': pts_2d_A,
            'uc_A': uc_A,
            'vc_A': vc_A,
            'r_A': r_A,
            'pts_2d_B': pts_2d_B,
            'uc_B': uc_B,
            'vc_B': vc_B,
            'r_B': r_B,
            'rmse_A': rmse_A,
            'rmse_B': rmse_B
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

        # 1. Compute stationary T_t5_to_marker
        dyn_model = self.robot.get_dynamics()
        q_current = self.robot.get_state().position
        
        # Load calibrated Tf_to_marker from setting.yaml
        tf_key = f"Tf_to_marker_{arm_side}"
        tf_vec = self.camera_config.get(tf_key, [0.0, 0.0775, -0.0667, 90.0, 0.0, 0.0])
        T_ee_to_marker = self.make_transform(tf_vec)
        
        ee_link = f"ee_{arm_side}"
        T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_current, ee_link, base_link="link_torso_5")
        T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker
        
        # Load mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # 2. Build Cross Sweep angles (-10 to 10 deg, 1 deg step = 21 steps each)
        sweep_deg = list(range(-10, 11)) # -10, -9, ..., 9, 10
        
        captured_data = []

        # Part A: Yaw sweep (Pitch kept at 0)
        if log_callback: log_callback("\n[Part A] Sweeping Head Yaw (-10 to 10 deg)...")
        for yaw in sweep_deg:
            q_head = np.deg2rad([yaw, 0.0])
            self.movej(self.robot, head=q_head, minimum_time=1.0)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_meas = pose[:3, 3] # meters
                captured_data.append((yaw, 0.0, p_meas))

        # Return head to zero
        self.movej(self.robot, head=[0, 0], minimum_time=1.5)
        time.sleep(0.5)

        # Part B: Pitch sweep (Yaw kept at 0)
        if log_callback: log_callback("\n[Part B] Sweeping Head Pitch (-10 to 10 deg)...")
        for pitch in sweep_deg:
            if pitch == 0: continue # Already collected in Part A
            q_head = np.deg2rad([0.0, pitch])
            self.movej(self.robot, head=q_head, minimum_time=1.0)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_meas = pose[:3, 3] # meters
                captured_data.append((0.0, pitch, p_meas))

        # Return head to zero
        self.movej(self.robot, head=[0, 0], minimum_time=1.5)

        if len(captured_data) < 10:
            if log_callback: log_callback("[ERROR] Too few data points captured. Calibration aborted.")
            return None

        # 3. Non-linear optimization to estimate Yaw/Pitch offsets
        model = self.robot.model()
        head_names = model.robot_joint_names
        state = dyn_model.make_state(["link_torso_5", "link_head_2"], head_names)

        def loss_func(params):
            dyaw, dpitch = params
            residuals = []
            
            for y_cmd, p_cmd, p_meas in captured_data:
                # Apply candidate offsets
                y_act = np.radians(y_cmd + dyaw)
                p_act = np.radians(p_cmd + dpitch)
                
                # Update q full
                q_temp = list(q_current)
                # Head joints are head_0 and head_1
                head_indices = model.head_idx
                q_temp[head_indices[0]] = y_act
                q_temp[head_indices[1]] = p_act
                
                # FK
                state.set_q(q_temp)
                dyn_model.compute_forward_kinematics(state)
                T_t5_to_mount = dyn_model.compute_transformation(state, 0, 1)
                
                T_t5_to_cam = T_t5_to_mount @ T_mount_to_cam
                T_cam_to_t5 = np.linalg.inv(T_t5_to_cam)
                T_cam_to_marker_pred = T_cam_to_t5 @ T_t5_to_marker
                p_pred = T_cam_to_marker_pred[:3, 3]
                
                residuals.extend(p_meas - p_pred)
                
            return np.array(residuals)

        # Initial guess: zero offsets
        opt_res = least_squares(loss_func, [0.0, 0.0], loss='huber')
        opt_yaw, opt_pitch = opt_res.x
        rmse = np.sqrt(np.mean(opt_res.fun**2)) * 1000.0 # mm

        if log_callback:
            log_callback("\n" + "="*40)
            log_callback("   HEAD CALIBRATION SUMMARY RESULTS")
            log_callback("="*40)
            log_callback(f"  * Estimated Head Yaw Offset  : {opt_yaw:.3f} deg")
            log_callback(f"  * Estimated Head Pitch Offset: {opt_pitch:.3f} deg")
            log_callback(f"  * Alignment Residual (RMSE)  : {rmse:.3f} mm")
            log_callback("="*40)

        # Construct projection points for visualization
        meas_pts_yaw = []
        pred_pts_yaw = []
        meas_pts_pitch = []
        pred_pts_pitch = []

        for y_cmd, p_cmd, p_meas in captured_data:
            y_act = np.radians(y_cmd + opt_yaw)
            p_act = np.radians(p_cmd + opt_pitch)
            
            q_temp = list(q_current)
            q_temp[model.head_idx[0]] = y_act
            q_temp[model.head_idx[1]] = p_act
            
            state.set_q(q_temp)
            dyn_model.compute_forward_kinematics(state)
            T_t5_to_mount = dyn_model.compute_transformation(state, 0, 1)
            
            T_t5_to_cam = T_t5_to_mount @ T_mount_to_cam
            T_cam_to_t5 = np.linalg.inv(T_t5_to_cam)
            p_pred = (T_cam_to_t5 @ T_t5_to_marker)[:3, 3]
            
            if p_cmd == 0.0: # Yaw sweep point
                meas_pts_yaw.append([y_cmd, p_meas[0], p_meas[1], p_meas[2]])
                pred_pts_yaw.append([y_cmd, p_pred[0], p_pred[1], p_pred[2]])
            else: # Pitch sweep point
                meas_pts_pitch.append([p_cmd, p_meas[0], p_meas[1], p_meas[2]])
                pred_pts_pitch.append([p_cmd, p_pred[0], p_pred[1], p_pred[2]])

        return {
            'mode': 'head',
            'opt_yaw': opt_yaw,
            'opt_pitch': opt_pitch,
            'rmse': rmse,
            'meas_pts_yaw': np.array(meas_pts_yaw),
            'pred_pts_yaw': np.array(pred_pts_yaw),
            'meas_pts_pitch': np.array(meas_pts_pitch),
            'pred_pts_pitch': np.array(pred_pts_pitch)
        }
