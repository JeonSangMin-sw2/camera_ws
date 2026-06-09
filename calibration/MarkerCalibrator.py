import time
import logging
import os
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from CalibratorBase import BaseCalibrator

class MarkerCalibrator(BaseCalibrator):

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm)...")
        
        # Get rotation only from mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        R_rob_to_cam = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
        p_target_cam = np.array([0.0, 0.0, target_dist / 1000.0])

        for attempt in range(5):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move canceled by user.")
                self.robot.cancel_control()
                return False
                
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
            
            pos_err_mm = np.linalg.norm(cam_pos - p_target_cam) * 1000.0
            rot_err_mat = cam_rot.T
            rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
            err_norm = np.linalg.norm([pos_err_mm, rot_err_deg])

            if log_callback:
                log_callback(f"  Current: X={cam_pos[0]*1000:.1f}, Y={cam_pos[1]*1000:.1f}, Z={cam_pos[2]*1000:.1f} mm")
                log_callback(f"  Error Norm: {err_norm:.2f} (Pos:{pos_err_mm:.1f}mm, Ang:{rot_err_deg:.1f}deg)")

            if err_norm <= 0.5:
                if log_callback: log_callback(f"  [SUCCESS] Reached center aligned pose! (Norm: {err_norm:.2f})")
                break

            if log_callback: log_callback("  Calculating joint command and moving...")
            
            dp_cam = p_target_cam - cam_pos
            dR_cam = cam_rot.T  # relative rotation error to identity
            
            # Rotate errors to robot frame (using only rotation R_rob_to_cam)
            dp_rob = R_rob_to_cam @ dp_cam
            dR_rob = R_rob_to_cam @ dR_cam @ R_rob_to_cam.T
            
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            p_ee = T_rob_to_ee[:3, 3]
            R_ee = T_rob_to_ee[:3, :3]
            
            T_rob_to_ee_new = np.eye(4)
            T_rob_to_ee_new[:3, :3] = dR_rob @ R_ee
            T_rob_to_ee_new[:3, 3] = p_ee + dp_rob
            
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
            success_other = self.movej(self.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0)
        else:
            success_other = self.movej(self.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0)
            
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
            left_arm = np.deg2rad([-90, 45, -73, -107, -80, 90, 0])
            
        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=[0, 0], minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True):
        if getattr(self, 'stop_requested', False):
            return None
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
        if getattr(self, 'stop_requested', False):
            return None

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
        
        # Retrieve joint limits for the global joint index of the active joint
        try:
            state_lim = dyn_model.make_state([f"ee_{arm_side}"], model.robot_joint_names)
            q_lower_all = np.array(dyn_model.get_limit_q_lower(state_lim))
            q_upper_all = np.array(dyn_model.get_limit_q_upper(state_lim))
            global_joint_idx = arm_idx[joint_i]
            q_min = q_lower_all[global_joint_idx]
            q_max = q_upper_all[global_joint_idx]
            if log_callback:
                log_callback(f"[INFO] Active joint {global_joint_idx} limits: min={np.degrees(q_min):.2f}°, max={np.degrees(q_max):.2f}°")
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to retrieve joint limits: {e}")
            q_min = -np.inf
            q_max = np.inf

        # Compute start and end arm poses with joint limit safety clamping (0.5 degree margin)
        safety_margin = np.radians(0.5)
        q_start_val = initial_joint_pos[joint_i] + np.radians(start_deg)
        q_end_val = initial_joint_pos[joint_i] + np.radians(end_deg)

        if q_start_val < q_min + safety_margin:
            q_start_val_new = q_min + safety_margin
            if log_callback:
                log_callback(f"[WARN] Start angle ({np.degrees(q_start_val):.2f}°) exceeds/overlaps min limit ({np.degrees(q_min):.2f}°). Clamping to {np.degrees(q_start_val_new):.2f}° with 0.5° safety margin.")
            q_start_val = q_start_val_new
            start_deg = np.degrees(q_start_val - initial_joint_pos[joint_i])

        if q_end_val > q_max - safety_margin:
            q_end_val_new = q_max - safety_margin
            if log_callback:
                log_callback(f"[WARN] End angle ({np.degrees(q_end_val):.2f}°) exceeds/overlaps max limit ({np.degrees(q_max):.2f}°). Clamping to {np.degrees(q_end_val_new):.2f}° with 0.5° safety margin.")
            q_end_val = q_end_val_new
            end_deg = np.degrees(q_end_val - initial_joint_pos[joint_i])
        
        q_start_arm = list(initial_joint_pos)
        q_start_arm[joint_i] = q_start_val
        q_end_arm = list(initial_joint_pos)
        q_end_arm[joint_i] = q_end_val
        
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
            ok = self.movej(self.robot, left_arm=q_start_arm, head=q_head_start, minimum_time=2.5, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start_arm, head=q_head_start, minimum_time=2.5, apply_offsets=False)
            
        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to move to start sweep position or stop was requested.")
            return None
            
        time.sleep(1.0)

        # 2. Continuous sweep from start to end position (30s duration)
        if log_callback: log_callback(f"[INFO] Commencing Continuous Sweep on Marker Axis {axis_mode} (duration=30s)...")
        
        if getattr(self, 'stop_requested', False):
            return None
            
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_arm if arm_side == "right" else None,
            left_arm=q_end_arm if arm_side == "left" else None,
            head=q_head_end, minimum_time=30.0
        )
        
        captured_poses = []
        captured_angles = []
        captured_q_full = []
        
        move_thread.start()
        
        # High speed data collection
        while move_thread.is_alive():
            if getattr(self, 'stop_requested', False):
                if self.robot and self.robot != "mock_robot":
                    self.robot.cancel_control()
                move_thread.join()
                return None
            lpf_results = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
            if lpf_results:
                pose = np.array(lpf_results[0]).reshape(4, 4) if isinstance(lpf_results, list) else np.array(list(lpf_results.values())[0]).reshape(4, 4)
                q_full_captured = np.array(self.robot.get_state().position)
                q_captured = q_full_captured[arm_idx[joint_i]]
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    captured_poses.append(pose)
                    captured_angles.append(np.degrees(q_captured - initial_joint_pos[joint_i]))
                    captured_q_full.append(q_full_captured)
            time.sleep(0.01) # 100Hz polling

        move_thread.join()
        if not move_thread.success:
            if log_callback: log_callback("[ERROR] Marker sweep motion failed or was cancelled.")
            return None
            
        if getattr(self, 'stop_requested', False):
            return None
            
        if log_callback: log_callback(f"    -> Swept {len(captured_poses)} dense raw coordinate frames.")

        # Return arm and head to original ready pose
        if log_callback: log_callback("\n[INFO] Sweep complete. Returning to initial ready pose...")
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.5, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.5, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to return to initial ready pose or stop was requested.")
            return None

        if len(captured_poses) < 10:
            if log_callback: log_callback("[ERROR] Too few valid marker poses. Calibration failed.")
            return None

        # Solve Circle Fitting
        n_nom = [1.0, 0.0, 0.0] if "6" in str(axis_mode).lower() else [0.0, 1.0, 0.0]
        res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom)
        
        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        # Force camera translation components to zero as requested and use fixed rotation
        mount_to_cam_rot_only = [0.0, 0.0, 0.0] + list(mount_to_cam[3:])
        T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
        
        pts_ee = []
        for q_full, pose_cam_to_marker in zip(captured_q_full, captured_poses):
            try:
                # Use strictly fixed transformation with zero translation and no live FK
                T_t5_to_marker = T_t5_to_cam_fixed @ pose_cam_to_marker
                T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                p_ee = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker @ np.array([0, 0, 0, 1])
                pts_ee.append(p_ee[:3] * 1000.0) # in mm
            except Exception as e:
                pass
        
        if len(pts_ee) > 0:
            res['pts_ee'] = np.array(pts_ee)
        else:
            res['pts_ee'] = np.zeros((0, 3))
            
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

        # Define fixed camera-to-robot rotation relationship (ZYX Euler: [-90.0, 0.0, -90.0])
        R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
        R_cam_to_rob = R_rob_to_cam.T

        # Transform fitted camera axis vectors directly to the robot torso (base) frame
        z_e_in_rob = R_cam_to_rob @ marker_data_6['axis']
        y_e_in_rob = R_cam_to_rob @ marker_data_5['axis']
        
        if arm_side == "left":
            ideal_rpy = [90.0, 0.0, 0.0]
        else:
            ideal_rpy = [90.0, 0.0, 180.0]
            
        T_ee_m_ideal = self.make_transform([0, 0, 0] + ideal_rpy)
        R_ee_m_ideal = T_ee_m_ideal[:3, :3]
        
        # Check directions in robot frame
        if np.dot(z_e_in_rob, R_ee_m_ideal[:, 2]) < 0: z_e_in_rob = -z_e_in_rob
        if np.dot(y_e_in_rob, R_ee_m_ideal[:, 1]) < 0: y_e_in_rob = -y_e_in_rob
        
        # Orthogonalize in robot frame
        y_e_in_rob = y_e_in_rob - np.dot(y_e_in_rob, z_e_in_rob) * z_e_in_rob
        y_e_in_rob /= np.linalg.norm(y_e_in_rob)
        x_e_in_rob = np.cross(y_e_in_rob, z_e_in_rob)
        
        R_ee_m_actual = np.column_stack((x_e_in_rob, y_e_in_rob, z_e_in_rob))
        
        # Enforce zero Z-axis twist constraint: Yaw is always 0.0 (left) or 180.0 (right)
        yaw_fixed = 0.0 if arm_side == "left" else 180.0
        R_z = R_scipy.from_euler('z', yaw_fixed, degrees=True).as_matrix()
        # Strip the target Yaw component to solve pure Pitch and Roll on R_yx
        R_yx = R_z.T @ R_ee_m_actual
        euler_deg_fixed = R_scipy.from_matrix(R_yx).as_euler('ZYX', degrees=True)
        _, pitch_e, roll_e = euler_deg_fixed
        yaw_e = yaw_fixed
        
        radius_6 = marker_data_6['radius']
        radius_5 = marker_data_5['radius']
        
        # Calculate offsets directly from circle radii and physical link length
        # strictly bypassing back-projected coordinates average as requested by the user
        x_e = 0.0
        y_e = radius_6 if arm_side == "left" else -radius_6
        z_e = -abs(radius_5 - L_5_ee)
            
        # Analysis Orthogonality / Quality metrics
        dot_val = np.dot(z_e_in_rob, y_e_in_rob)
        angle_between = np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0)))
        ortho_err = abs(90.0 - angle_between)
        
        # Geodesic rotation error to check difference between ideal and actual matrices
        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
        
        # Check alignment of individual axis tilts (axis 5 tilt and axis 6 tilt should be close if they match)
        tilt_diff = abs(marker_data_5.get('tilt', 0.0) - marker_data_6.get('tilt', 0.0))
        
        rmse_6 = marker_data_6['rmse']
        rmse_5 = marker_data_5['rmse']
        
        return {
            'x_e': x_e, 'y_e': y_e, 'z_e': z_e,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee, 'radius_6': radius_6, 'radius_5': radius_5,
            'ortho_err': ortho_err, 'rmse_6': rmse_6, 'rmse_5': rmse_5,
            'rot_err_deg': rot_err_deg, 'tilt_diff': tilt_diff,
            'warn_large_angle': rot_err_deg > 15.0
        }
