import time
import logging
import os
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator

class MarkerCalibrator(BaseCalibrator):

    @staticmethod
    def rodrigues_rotation(vector, axis, theta_rad):
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return vector * cos_t + np.cross(axis, vector) * sin_t + axis * np.dot(axis, vector) * (1 - cos_t)

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0, max_attempts=5):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm, max_attempts: {max_attempts})...")
        
        # Get rotation only from mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        R_rob_to_cam = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
        p_target_cam = np.array([0.0, 0.0, target_dist / 1000.0])

        for attempt in range(max_attempts):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move canceled by user.")
                self.robot.cancel_control()
                return False
                
            if log_callback: log_callback(f"[Attempt {attempt + 1}/{max_attempts}] Capturing marker pose...")
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

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True, save_debug=False):
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
            
        is_camera_mock = (self.marker_st is None or type(self.marker_st).__name__ == "SimulatedMarkerTransform")

        if not is_camera_mock:
            # Pre-check marker visibility
            initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not initial_check:
                if log_callback: log_callback("[ERROR] Marker is not visible in ready pose.")
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

        # Sweep configuration
        axis_str = str(axis_mode).lower()
        if "6" in axis_str:
            start_deg = -20.0
            end_deg = 20.0
            joint_i = 6
        elif "4" in axis_str:
            start_deg = -10.0
            end_deg = 10.0
            joint_i = 4
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

        # 2. Continuous sweep from start to end position (15s duration)
        if log_callback: log_callback(f"[INFO] Commencing Continuous Sweep on Marker Axis {axis_mode} (duration=15s)...")
        
        if getattr(self, 'stop_requested', False):
            return None
            
        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end_arm if arm_side == "right" else None,
            left_arm=q_end_arm if arm_side == "left" else None,
            head=q_head_end, minimum_time=15.0
        )
        
        captured_poses = []
        captured_angles = []
        captured_q_full = []
        
        if self.robot and self.robot != "mock_robot":
            initial_full_pose = np.array(self.robot.get_state().position)
        else:
            initial_full_pose = np.zeros(20)

        t_start = time.time()
        move_thread.start()
        
        # High speed data collection
        while move_thread.is_alive():
            if getattr(self, 'stop_requested', False):
                if self.robot and self.robot != "mock_robot":
                    self.robot.cancel_control()
                move_thread.join()
                return None
                
            if is_camera_mock:
                pose = self.get_simulated_marker_pose(arm_side, sweep_joint=joint_i)
                lpf_results = [pose.tolist()]
            else:
                lpf_results = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
                
            if lpf_results:
                pose = np.array(lpf_results[0]).reshape(4, 4) if isinstance(lpf_results, list) else np.array(list(lpf_results.values())[0]).reshape(4, 4)
                
                q_full_captured = np.array(self.robot.get_state().position)
                global_joint_idx = arm_idx[joint_i]
                q_captured = q_full_captured[global_joint_idx]
                
                if np.linalg.norm(pose[:3, 3]) > 0.01:
                    captured_poses.append(pose)
                    captured_angles.append(np.degrees(q_captured - initial_joint_pos[joint_i]))
                    captured_q_full.append(q_full_captured)
                    
            if is_camera_mock:
                time.sleep(0.3)
            else:
                time.sleep(0.01)

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
        axis_str = str(axis_mode).lower()
        if "6" in axis_str:
            n_nom = [1.0, 0.0, 0.0]
        elif "4" in axis_str:
            n_nom = [0.0, 0.0, 1.0]
        else:
            n_nom = [0.0, 1.0, 0.0]
        res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom)
        
        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        # Force camera translation components to zero as requested and use fixed rotation
        mount_to_cam_rot_only = [0.0, 0.0, 0.0] + list(mount_to_cam[3:])
        T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
        
        pts_ee = []
        for q_full, pose_cam_to_marker in zip(captured_q_full, captured_poses):
            try:
                if self.robot and self.robot != "mock_robot":
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_t5_to_cam_fixed
                else:
                    T_t5_to_cam = T_t5_to_cam_fixed
                
                T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                p_ee = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker @ np.array([0, 0, 0, 1])
                pts_ee.append(p_ee[:3] * 1000.0) # in mm
            except Exception as e:
                pass
        
        if len(pts_ee) > 0:
            res['pts_ee'] = np.array(pts_ee)
        else:
            res['pts_ee'] = np.zeros((0, 3))
            
        res['captured_poses'] = captured_poses
        res['captured_q_full'] = captured_q_full
        if save_debug:
            dataset = list(zip(captured_q_full, captured_poses))
            self.save_debug_points(
                arm_side, axis_mode, dataset, initial_joint_pos, ee_name, dyn_model, T_t5_to_cam_fixed, "marker", log_callback
            )
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

    def compute_unified_bracket_calibration_v1_3(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None):
        L_5_ee = self.get_link_length(arm_side)

        # 1. Nominal marker orientation in EE frame
        version_suffix = "_v13" if self.is_v13() else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            
        if tf_vec is not None and len(tf_vec) >= 6:
            nominal_rpy = [tf_vec[3], tf_vec[4], tf_vec[5]]
        else:
            nominal_rpy = [90.0, 0.0, -90.0]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

        # Helper to extract rotation axis
        def extract_axis_from_rotations(poses, ideal_axis):
            if len(poses) < 2: return ideal_axis
            mid_idx = len(poses) // 2
            R_ref = poses[mid_idx][:3, :3]
            axes = []
            for i, T in enumerate(poses):
                if i == mid_idx: continue
                R_rel = R_ref.T @ T[:3, :3] 
                rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle > np.radians(1.0):
                    axis = rotvec / angle
                    if np.dot(axis, ideal_axis) < 0: axis = -axis
                    axes.append(axis)
            if len(axes) > 0:
                avg_axis = np.mean(axes, axis=0)
                return avg_axis / np.linalg.norm(avg_axis)
            return ideal_axis

        # Ideal axes in marker frame (for sign resolution)
        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        poses_6 = marker_data_6.get('captured_poses', []) if marker_data_6 else []
        n6_marker_actual = extract_axis_from_rotations(poses_6, x_ee_m_ideal)

        poses_5 = marker_data_5.get('captured_poses', []) if marker_data_5 else []
        n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)

        if marker_data_4 is not None:
            poses_4 = marker_data_4.get('captured_poses', [])
            n4_marker_actual = extract_axis_from_rotations(poses_4, z_ee_m_ideal)
        else:
            n4_marker_actual = None

        radius_6 = marker_data_6.get('radius', 0.0) if marker_data_6 else 0.0
        radius_5 = marker_data_5.get('radius', 0.0) if marker_data_5 else 0.0
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0

        x_nom = tf_vec[0] * 1000.0 if tf_vec is not None else 95.0
        y_nom = tf_vec[1] * 1000.0 if tf_vec is not None else 0.0
        z_nom = tf_vec[2] * 1000.0 if tf_vec is not None else -5.0

        # Stage 1: Solve for Joint 6 offset and marker roll misalignment using 6-axis sweep data
        R_list_6 = []
        if self.robot and self.robot != "mock_robot":
            try:
                mount_to_cam_rot_only = [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
                T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
                R_t5_to_cam = T_t5_to_cam_fixed[:3, :3]
                dyn_model = self.robot.get_dynamics()
                ee_name = f"ee_{arm_side}"
                q_full_6 = marker_data_6.get('captured_q_full', [])
                for q_full, T_cam_to_marker in zip(q_full_6, poses_6):
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                    R_ee_to_t5 = T_t5_to_ee[:3, :3].T
                    R_cam_to_marker = T_cam_to_marker[:3, :3]
                    R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
                    R_list_6.append(R_ee_to_marker)
            except Exception as e:
                logging.warning(f"Kinematic calculation failed in stage 1: {e}")

        if len(R_list_6) > 0:
            M = np.mean(R_list_6, axis=0)
            U, S, Vt = np.linalg.svd(M)
            R_ee_m_measured = U @ Vt
            if np.linalg.det(R_ee_m_measured) < 0:
                U[:, 2] *= -1
                R_ee_m_measured = U @ Vt
        else:
            # Fallback to Gram-Schmidt if no robot kinematics available
            x_col = n6_marker_actual
            y_col = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
            y_col /= np.linalg.norm(y_col)
            z_col = np.cross(x_col, y_col)
            R_ee_m_measured = np.column_stack((x_col, y_col, z_col)).T

        # Decompose R_ee_m_measured into ZYX Euler angles
        # Calculate rotation difference in the EE frame: R_diff = R_ee_m_measured @ R_ee_m_ideal.T
        R_diff = R_ee_m_measured @ R_ee_m_ideal.T
        yaw_diff, pitch_diff, roll_diff = R_scipy.from_matrix(R_diff).as_euler('ZYX', degrees=True)
        
        # Calculate J6 offset based on marker roll misalignment in the EE frame
        opt_delta_6_deg = roll_diff
        # Normalize to [-180, 180] range
        opt_delta_6_deg = (opt_delta_6_deg + 180.0) % 360.0 - 180.0
        d6_init = np.radians(opt_delta_6_deg)

        print(f"DEBUG STAGE 1: roll_diff={roll_diff:.3f}, opt_delta_6_deg={opt_delta_6_deg:.3f}")

        # Stage 2: QP optimization for offsets, position, and orientation errors (in meters & radians)
        # NOTE: L_5_ee is included as a free optimization variable (index 8) because:
        # - The mock robot returns a nominal 300 mm which may differ from the actual robot
        # - The physical z_e can be ~-130 mm, far outside z_nom ± 60 mm bounds
        # - Treating L_5_ee as free allows the optimizer to jointly solve for true geometry
        x_nom_m = x_nom / 1000.0
        y_nom_m = y_nom / 1000.0
        # Estimate a physically meaningful z_e initial value from the J6 sweep radius:
        # r6 = sqrt(y_e^2 + z_e^2). With y_e ≈ 0, z_e ≈ -r6 (negative Z convention for v1.3)
        radius_6_m = radius_6 / 1000.0
        radius_5_m = radius_5 / 1000.0
        radius_4_m = radius_4 / 1000.0
        z_init_m = -radius_6_m if radius_6_m > abs(z_nom / 1000.0) + 0.010 else (z_nom / 1000.0)
        L_5_ee_m = L_5_ee / 1000.0
        L_nom_m  = L_5_ee_m  # nominal link length from robot model

        # State vector: [yaw_off, pitch_off, roll_off, d5, d6, x_e, y_e, z_e, L_5_ee] in meters/radians
        x_state = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom_m, y_nom_m, z_init_m, L_nom_m], dtype=float)
        x_target = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom_m, y_nom_m, z_init_m, L_nom_m], dtype=float)
        # Regularization weights:
        # - y_e: strongly pulled to y_nom (physical bracket should have small y offset ≤1 mm)
        # - L_5_ee: strongly pulled to robot model value (unlikely to deviate >40 mm)
        # - x_e, z_e: moderate (allow larger position freedom)
        w_reg = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-2, 1e-3, 2e-2])

        # Bounds: z_e uses absolute physical range (v1.3 bracket extends up to ~200 mm in -Z)
        # L_5_ee bounded tightly around robot-model value (±80 mm)
        x_min = np.array([
            -np.radians(30.0), -np.radians(30.0), 0.0,
            -np.radians(15.0), d6_init - np.radians(15.0),
            x_nom_m - 0.030, y_nom_m - 0.003, -0.250,
            L_nom_m - 0.080
        ])
        x_max = np.array([
            np.radians(30.0), np.radians(30.0), 0.0,
            np.radians(15.0), d6_init + np.radians(15.0),
            x_nom_m + 0.030, y_nom_m + 0.003, 0.010,
            L_nom_m + 0.080
        ])

        def eval_residuals(x):
            y_off, p_off, r_off, d5_val, d6_val, xe, ye, ze, L_m = x
            R_off = R_scipy.from_euler('ZYX', [y_off, p_off, r_off]).as_matrix()
            R_em = R_off @ R_ee_m_ideal
            
            n6_p = R_em.T @ np.array([1.0, 0.0, 0.0])
            n5_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ np.array([0.0, 1.0, 0.0])
            
            r6_p = np.sqrt(ye**2 + ze**2)
            Z_p = ye * np.sin(d6_val) + ze * np.cos(d6_val) + L_m
            r5_p = np.sqrt(xe**2 + Z_p**2)
            
            res = []
            res.extend(n6_marker_actual - n6_p)
            res.extend(n5_marker_actual - n5_p)
            
            if n4_marker_actual is not None:
                n4_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ R_scipy.from_euler('Y', -d5_val).as_matrix() @ np.array([0.0, 0.0, 1.0])
                res.extend(n4_marker_actual - n4_p)
                
            res.append(radius_6_m - r6_p)
            res.append(radius_5_m - r5_p)
            if marker_data_4 is not None:
                Y_p = ye * np.cos(d6_val) - ze * np.sin(d6_val)
                r4_p = np.sqrt((xe * np.cos(d5_val) + Z_p * np.sin(d5_val))**2 + Y_p**2)
                res.append(radius_4_m - r4_p)
                
            for idx in range(len(x)):
                res.append(w_reg[idx] * (x[idx] - x_target[idx]))
            return np.array(res, dtype=float)

        max_iter = 150
        eps_converge = 1e-9
        qp_reg = 1e-8

        import qpsolvers
        for iteration in range(max_iter):
            f_vals = eval_residuals(x_state)
            
            # Numeric Jacobian (centered differences)
            eps_jac = 1e-7
            J = np.zeros((len(f_vals), len(x_state)))
            for j in range(len(x_state)):
                x_plus = x_state.copy()
                x_plus[j] += eps_jac
                f_plus = eval_residuals(x_plus)
                x_minus = x_state.copy()
                x_minus[j] -= eps_jac
                f_minus = eval_residuals(x_minus)
                J[:, j] = (f_plus - f_minus) / (2.0 * eps_jac)
                
            H = J.T @ J
            g = J.T @ f_vals
            
            P = H + qp_reg * np.eye(len(x_state))
            q = g
            
            dx_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            lb = np.maximum(-dx_max, x_min - x_state)
            ub = np.minimum(dx_max, x_max - x_state)
            
            dx = qpsolvers.solve_qp(P, q, lb=lb, ub=ub, solver='osqp')
            if dx is None:
                dx = -0.1 * g / (np.linalg.norm(g) + 1e-8)
                dx = np.clip(dx, lb, ub)
                
            x_state += dx
            if np.linalg.norm(dx) < eps_converge:
                break

        yaw_off_opt, pitch_off_opt, roll_off_opt, d5_opt, d6_opt, xe_opt, ye_opt, ze_opt, L_5_ee_solved = x_state
        xe_opt = xe_opt * 1000.0
        ye_opt = ye_opt * 1000.0
        ze_opt = ze_opt * 1000.0
        L_5_ee = L_5_ee_solved * 1000.0  # update with optimized value

        opt_delta_5 = float(np.degrees(d5_opt))
        opt_delta_6 = float(np.degrees(d6_opt))

        R_off_opt = R_scipy.from_euler('ZYX', [yaw_off_opt, pitch_off_opt, roll_off_opt]).as_matrix()
        R_ee_m_actual = R_off_opt @ R_ee_m_ideal
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        if arm_side == "right" and yaw_e < 0:
            yaw_e += 360.0

        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))

        dot_val = np.dot(n6_marker_actual, n5_marker_actual)
        ortho_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0))))

        return {
            'converged': True,
            'x_e': xe_opt, 'y_e': ye_opt, 'z_e': ze_opt,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee,  # optimized link length
            'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
            'ortho_err': ortho_err,
            'rmse_6': marker_data_6.get('rmse', 0.0) if marker_data_6 else 0.0,
            'rmse_5': marker_data_5.get('rmse', 0.0) if marker_data_5 else 0.0,
            'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 is not None else 0.0,
            'rot_err_deg': rot_err_deg, 'tilt_diff': 0.0,
            'warn_large_angle': rot_err_deg > 15.0,
            'opt_delta_5': opt_delta_5,
            'opt_delta_6': opt_delta_6,
            'min_radius': radius_4
        }

    def compute_unified_bracket_calibration_v1_3_with_divergence_check(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None):
        L_5_ee = self.get_link_length(arm_side)

        # 1. Nominal marker orientation in EE frame
        version_suffix = "_v13" if self.is_v13() else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            
        if tf_vec is not None and len(tf_vec) >= 6:
            nominal_rpy = [tf_vec[3], tf_vec[4], tf_vec[5]]
        else:
            nominal_rpy = [90.0, 0.0, -90.0]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

        # Helper to extract rotation axis
        def extract_axis_from_rotations(poses, ideal_axis):
            if len(poses) < 2: return ideal_axis
            mid_idx = len(poses) // 2
            R_ref = poses[mid_idx][:3, :3]
            axes = []
            for i, T in enumerate(poses):
                if i == mid_idx: continue
                R_rel = R_ref.T @ T[:3, :3] 
                rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle > np.radians(1.0):
                    axis = rotvec / angle
                    if np.dot(axis, ideal_axis) < 0: axis = -axis
                    axes.append(axis)
            if len(axes) > 0:
                avg_axis = np.mean(axes, axis=0)
                return avg_axis / np.linalg.norm(avg_axis)
            return ideal_axis

        # Ideal axes in marker frame (for sign resolution)
        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        poses_6 = marker_data_6.get('captured_poses', []) if marker_data_6 else []
        n6_marker_actual = extract_axis_from_rotations(poses_6, x_ee_m_ideal)

        poses_5 = marker_data_5.get('captured_poses', []) if marker_data_5 else []
        n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)

        if marker_data_4 is not None:
            poses_4 = marker_data_4.get('captured_poses', [])
            n4_marker_actual = extract_axis_from_rotations(poses_4, z_ee_m_ideal)
        else:
            n4_marker_actual = None

        radius_6 = marker_data_6.get('radius', 0.0) if marker_data_6 else 0.0
        radius_5 = marker_data_5.get('radius', 0.0) if marker_data_5 else 0.0
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0

        x_nom = tf_vec[0] * 1000.0 if tf_vec is not None else 95.0
        y_nom = tf_vec[1] * 1000.0 if tf_vec is not None else 0.0
        z_nom = tf_vec[2] * 1000.0 if tf_vec is not None else -5.0

        # Stage 1: Solve for Joint 6 offset and marker roll misalignment using 6-axis sweep data
        R_list_6 = []
        if self.robot and self.robot != "mock_robot":
            try:
                mount_to_cam_rot_only = [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
                T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
                R_t5_to_cam = T_t5_to_cam_fixed[:3, :3]
                dyn_model = self.robot.get_dynamics()
                ee_name = f"ee_{arm_side}"
                q_full_6 = marker_data_6.get('captured_q_full', [])
                for q_full, T_cam_to_marker in zip(q_full_6, poses_6):
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                    R_ee_to_t5 = T_t5_to_ee[:3, :3].T
                    R_cam_to_marker = T_cam_to_marker[:3, :3]
                    R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
                    R_list_6.append(R_ee_to_marker)
            except Exception as e:
                logging.warning(f"Kinematic calculation failed in stage 1: {e}")

        if len(R_list_6) > 0:
            M = np.mean(R_list_6, axis=0)
            U, S, Vt = np.linalg.svd(M)
            R_ee_m_measured = U @ Vt
            if np.linalg.det(R_ee_m_measured) < 0:
                U[:, 2] *= -1
                R_ee_m_measured = U @ Vt
        else:
            # Fallback to Gram-Schmidt if no robot kinematics available
            x_col = n6_marker_actual
            y_col = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
            y_col /= np.linalg.norm(y_col)
            z_col = np.cross(x_col, y_col)
            R_ee_m_measured = np.column_stack((x_col, y_col, z_col)).T

        # Decompose R_ee_m_measured into ZYX Euler angles
        # Calculate rotation difference in the EE frame: R_diff = R_ee_m_measured @ R_ee_m_ideal.T
        R_diff = R_ee_m_measured @ R_ee_m_ideal.T
        yaw_diff, pitch_diff, roll_diff = R_scipy.from_matrix(R_diff).as_euler('ZYX', degrees=True)
        
        # Calculate J6 offset based on marker roll misalignment in the EE frame
        opt_delta_6_deg = roll_diff
        # Normalize to [-180, 180] range
        opt_delta_6_deg = (opt_delta_6_deg + 180.0) % 360.0 - 180.0
        d6_init = np.radians(opt_delta_6_deg)

        # Stage 2: QP optimization for offsets, position, and orientation errors (in meters & radians)
        x_nom_m = x_nom / 1000.0
        y_nom_m = y_nom / 1000.0
        radius_6_m = radius_6 / 1000.0
        radius_5_m = radius_5 / 1000.0
        radius_4_m = radius_4 / 1000.0
        z_init_m = -radius_6_m if radius_6_m > abs(z_nom / 1000.0) + 0.010 else (z_nom / 1000.0)
        L_5_ee_m = L_5_ee / 1000.0
        L_nom_m  = L_5_ee_m  # nominal link length from robot model

        # State vector: [yaw_off, pitch_off, roll_off, d5, d6, x_e, y_e, z_e, L_5_ee] in meters/radians
        x_state = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom_m, y_nom_m, z_init_m, L_nom_m], dtype=float)
        x_target = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom_m, y_nom_m, z_init_m, L_nom_m], dtype=float)
        w_reg = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-2, 1e-3, 2e-2])

        x_min = np.array([
            -np.radians(30.0), -np.radians(30.0), 0.0,
            -np.radians(15.0), d6_init - np.radians(15.0),
            x_nom_m - 0.030, y_nom_m - 0.003, -0.250,
            L_nom_m - 0.080
        ])
        x_max = np.array([
            np.radians(30.0), np.radians(30.0), 0.0,
            np.radians(15.0), d6_init + np.radians(15.0),
            x_nom_m + 0.030, y_nom_m + 0.003, 0.010,
            L_nom_m + 0.080
        ])

        def eval_residuals(x):
            y_off, p_off, r_off, d5_val, d6_val, xe, ye, ze, L_m = x
            R_off = R_scipy.from_euler('ZYX', [y_off, p_off, r_off]).as_matrix()
            R_em = R_off @ R_ee_m_ideal
            
            n6_p = R_em.T @ np.array([1.0, 0.0, 0.0])
            n5_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ np.array([0.0, 1.0, 0.0])
            
            r6_p = np.sqrt(ye**2 + ze**2)
            Z_p = ye * np.sin(d6_val) + ze * np.cos(d6_val) + L_m
            r5_p = np.sqrt(xe**2 + Z_p**2)
            
            res = []
            res.extend(n6_marker_actual - n6_p)
            res.extend(n5_marker_actual - n5_p)
            
            if n4_marker_actual is not None:
                n4_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ R_scipy.from_euler('Y', -d5_val).as_matrix() @ np.array([0.0, 0.0, 1.0])
                res.extend(n4_marker_actual - n4_p)
                
            res.append(radius_6_m - r6_p)
            res.append(radius_5_m - r5_p)
            if marker_data_4 is not None:
                Y_p = ye * np.cos(d6_val) - ze * np.sin(d6_val)
                r4_p = np.sqrt((xe * np.cos(d5_val) + Z_p * np.sin(d5_val))**2 + Y_p**2)
                res.append(radius_4_m - r4_p)
                
            for idx in range(len(x)):
                res.append(w_reg[idx] * (x[idx] - x_target[idx]))
            return np.array(res, dtype=float)

        max_iter = 150
        eps_converge = 1e-9
        qp_reg = 1e-8

        import qpsolvers
        best_err = float('inf')
        best_x_state = x_state.copy()
        consecutive_increases = 0

        for iteration in range(max_iter):
            f_vals = eval_residuals(x_state)
            curr_err = np.linalg.norm(f_vals)

            if np.isnan(curr_err) or np.isinf(curr_err):
                print(f"[WARNING] Bracket calibration optimization error is numerical invalid ({curr_err}) at iteration {iteration}! Reverting to best parameters and halting.")
                x_state = best_x_state.copy()
                break

            if curr_err < best_err:
                best_err = curr_err
                best_x_state = x_state.copy()
                consecutive_increases = 0
            else:
                consecutive_increases += 1

            if curr_err > best_err * 2.0:
                print(f"[WARNING] Bracket calibration optimization error exploded ({curr_err:.3e} > 2.0 * best_error {best_err:.3e}) at iteration {iteration}! Reverting to best parameters and halting.")
                x_state = best_x_state.copy()
                break

            if consecutive_increases >= 10:
                print(f"[WARNING] Bracket calibration optimization error has been increasing consecutively for {consecutive_increases} iterations (current error: {curr_err:.3e}, best error: {best_err:.3e}) at iteration {iteration}! Reverting to best parameters and halting.")
                x_state = best_x_state.copy()
                break
            
            # Numeric Jacobian (centered differences)
            eps_jac = 1e-7
            J = np.zeros((len(f_vals), len(x_state)))
            for j in range(len(x_state)):
                x_plus = x_state.copy()
                x_plus[j] += eps_jac
                f_plus = eval_residuals(x_plus)
                x_minus = x_state.copy()
                x_minus[j] -= eps_jac
                f_minus = eval_residuals(x_minus)
                J[:, j] = (f_plus - f_minus) / (2.0 * eps_jac)
                
            H = J.T @ J
            g = J.T @ f_vals
            
            P = H + qp_reg * np.eye(len(x_state))
            q = g
            
            dx_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            lb = np.maximum(-dx_max, x_min - x_state)
            ub = np.minimum(dx_max, x_max - x_state)
            
            dx = qpsolvers.solve_qp(P, q, lb=lb, ub=ub, solver='osqp')
            if dx is None:
                dx = -0.1 * g / (np.linalg.norm(g) + 1e-8)
                dx = np.clip(dx, lb, ub)
                
            x_state += dx
            if np.linalg.norm(dx) < eps_converge:
                break

        yaw_off_opt, pitch_off_opt, roll_off_opt, d5_opt, d6_opt, xe_opt, ye_opt, ze_opt, L_5_ee_solved = x_state
        xe_opt = xe_opt * 1000.0
        ye_opt = ye_opt * 1000.0
        ze_opt = ze_opt * 1000.0
        L_5_ee = L_5_ee_solved * 1000.0  # update with optimized value

        opt_delta_5 = float(np.degrees(d5_opt))
        opt_delta_6 = float(np.degrees(d6_opt))

        R_off_opt = R_scipy.from_euler('ZYX', [yaw_off_opt, pitch_off_opt, roll_off_opt]).as_matrix()
        R_ee_m_actual = R_off_opt @ R_ee_m_ideal
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        if arm_side == "right" and yaw_e < 0:
            yaw_e += 360.0

        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))

        dot_val = np.dot(n6_marker_actual, n5_marker_actual)
        ortho_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0))))

        return {
            'converged': True,
            'x_e': xe_opt, 'y_e': ye_opt, 'z_e': ze_opt,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee,  # optimized link length
            'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
            'ortho_err': ortho_err,
            'rmse_6': marker_data_6.get('rmse', 0.0) if marker_data_6 else 0.0,
            'rmse_5': marker_data_5.get('rmse', 0.0) if marker_data_5 else 0.0,
            'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 is not None else 0.0,
            'rot_err_deg': rot_err_deg, 'tilt_diff': 0.0,
            'warn_large_angle': rot_err_deg > 15.0,
            'opt_delta_5': opt_delta_5,
            'opt_delta_6': opt_delta_6,
            'min_radius': radius_4
        }

    def compute_unified_bracket_calibration(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None):
        if self.is_v13():
            return self.compute_unified_bracket_calibration_v1_3(marker_data_5, marker_data_6, arm_side, tolerance=tolerance, marker_data_4=marker_data_4)

        L_5_ee = self.get_link_length(arm_side)

        # 1. 이상적인 마커 오일러 각도 (ZYX 기준)
        version_suffix = "_v13" if self.is_v13() else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            
        if tf_vec is not None and len(tf_vec) >= 6:
            nominal_rpy = [tf_vec[3], tf_vec[4], tf_vec[5]]
        else:
            if arm_side == "left":
                nominal_rpy = [90.0, 0.0, 0.0]
            else:
                nominal_rpy = [90.0, 0.0, 180.0]
            
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
        
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])

        def extract_axis_from_rotations(poses, ideal_axis):
            if len(poses) < 2: return ideal_axis
            mid_idx = len(poses) // 2
            R_ref = poses[mid_idx][:3, :3]
            axes = []
            for i, T in enumerate(poses):
                if i == mid_idx: continue
                R_rel = R_ref.T @ T[:3, :3] 
                rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle > np.radians(1.0):
                    axis = rotvec / angle
                    if np.dot(axis, ideal_axis) < 0: axis = -axis
                    axes.append(axis)
            if len(axes) > 0:
                avg_axis = np.mean(axes, axis=0)
                return avg_axis / np.linalg.norm(avg_axis)
            return ideal_axis

        # 2. 정밀 회전축 벡터 산출 (신뢰도 평가 및 Fallback 용)
        poses_6 = marker_data_6.get('captured_poses', [])
        n6_marker_actual = extract_axis_from_rotations(poses_6, z_ee_m_ideal)
        
        poses_5 = marker_data_5.get('captured_poses', [])
        n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)

        # Try direct kinematic averaging first, as it is mathematically far more accurate and does not require Joint 4 sweep!
        kinematic_success = False
        if self.robot and self.robot != "mock_robot":
            try:
                mount_to_cam_rot_only = [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
                T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
                R_t5_to_cam = T_t5_to_cam_fixed[:3, :3]
                
                dyn_model = self.robot.get_dynamics()
                ee_name = f"ee_{arm_side}"
                
                R_list = []
                # Process Stage 6 (Roll)
                poses_6 = marker_data_6.get('captured_poses', [])
                q_full_6 = marker_data_6.get('captured_q_full', [])
                for q_full, T_cam_to_marker in zip(q_full_6, poses_6):
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                    R_ee_to_t5 = T_t5_to_ee[:3, :3].T
                    R_cam_to_marker = T_cam_to_marker[:3, :3]
                    R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
                    R_list.append(R_ee_to_marker)
                    
                # Process Stage 5 (Pitch)
                poses_5 = marker_data_5.get('captured_poses', [])
                q_full_5 = marker_data_5.get('captured_q_full', [])
                for q_full, T_cam_to_marker in zip(q_full_5, poses_5):
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                    R_ee_to_t5 = T_t5_to_ee[:3, :3].T
                    R_cam_to_marker = T_cam_to_marker[:3, :3]
                    R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
                    R_list.append(R_ee_to_marker)
                    
                if marker_data_4 is not None:
                    # Process Stage 4 (Yaw) if available
                    poses_4 = marker_data_4.get('captured_poses', [])
                    q_full_4 = marker_data_4.get('captured_q_full', [])
                    for q_full, T_cam_to_marker in zip(q_full_4, poses_4):
                        T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_full, ee_name, "link_torso_5")
                        R_ee_to_t5 = T_t5_to_ee[:3, :3].T
                        R_cam_to_marker = T_cam_to_marker[:3, :3]
                        R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
                        R_list.append(R_ee_to_marker)

                if len(R_list) > 0:
                    M = np.mean(R_list, axis=0)
                    U, S, Vt = np.linalg.svd(M)
                    R_ee_m_actual = U @ Vt
                    if np.linalg.det(R_ee_m_actual) < 0:
                        U[:, 2] *= -1
                        R_ee_m_actual = U @ Vt
                    kinematic_success = True
                else:
                    raise ValueError("No pose data available for kinematic averaging")
            except Exception as e:
                logging.warning(f"Kinematic averaging failed ({e}). Falling back to axis fitting.")

        if not kinematic_success:

            # Joint 6 angle correction for Joint 5 sweep
            theta_6 = marker_data_5.get('theta_6', None)
            if theta_6 is None:
                q_full_5 = marker_data_5.get('captured_q_full', [])
                if len(q_full_5) > 0 and self.robot and self.robot != "mock_robot":
                    try:
                        model = self.robot.model()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        q_idx = arm_idx[6]
                        theta_6 = np.mean([q[q_idx] for q in q_full_5])
                    except Exception as e:
                        logging.warning(f"Failed to extract joint 6 angle for correction: {e}")
                        theta_6 = 0.0
                else:
                    theta_6 = 0.0

            if marker_data_4 is not None:
                # --- 3-Axis SVD Alignment (Using Joint 4, 5, and 6) ---
                poses_4 = marker_data_4.get('captured_poses', [])
                n4_marker_actual = extract_axis_from_rotations(poses_4, x_ee_m_ideal)

                # Joint 6 angle correction for Joint 4 sweep
                theta_6_4 = marker_data_4.get('theta_6', None)
                if theta_6_4 is None:
                    q_full_4 = marker_data_4.get('captured_q_full', [])
                    if len(q_full_4) > 0 and self.robot and self.robot != "mock_robot":
                        try:
                            model = self.robot.model()
                            arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                            q_idx = arm_idx[6]
                            theta_6_4 = np.mean([q[q_idx] for q in q_full_4])
                        except Exception as e:
                            logging.warning(f"Failed to extract joint 6 angle for joint 4 correction: {e}")
                            theta_6_4 = 0.0
                    else:
                        theta_6_4 = 0.0

                z_col = n6_marker_actual
                
                # Form orthogonal projections of n5 and n4 onto plane perpendicular to z_col
                y_col_rot = n5_marker_actual - np.dot(n5_marker_actual, z_col) * z_col
                y_col_rot /= np.linalg.norm(y_col_rot)
                
                x_col_rot = n4_marker_actual - np.dot(n4_marker_actual, z_col) * z_col
                x_col_rot /= np.linalg.norm(x_col_rot)

                # Apply Joint 6 angle rotations back
                if abs(theta_6) > 1e-5:
                    y_col = self.rodrigues_rotation(y_col_rot, z_col, theta_6)
                else:
                    y_col = y_col_rot

                if abs(theta_6_4) > 1e-5:
                    x_col = self.rodrigues_rotation(x_col_rot, z_col, theta_6_4)
                else:
                    x_col = x_col_rot

                # Use SVD to clean up orthogonality errors and build R_m_ee
                M = np.column_stack((x_col, y_col, z_col))
                U, S, Vt = np.linalg.svd(M)
                R_m_ee_actual = U @ Vt
                if np.linalg.det(R_m_ee_actual) < 0:
                    U[:, 2] *= -1
                    R_m_ee_actual = U @ Vt
                
                R_ee_m_actual = R_m_ee_actual.T
            else:
                # --- 2-Axis Gram-Schmidt Alignment (Joint 5 and 6) ---
                z_col = n6_marker_actual
                y_col_rotated = n5_marker_actual - np.dot(n5_marker_actual, z_col) * z_col
                y_col_rotated /= np.linalg.norm(y_col_rotated)
                
                if abs(theta_6) > 1e-5:
                    y_col = self.rodrigues_rotation(y_col_rotated, z_col, theta_6)
                else:
                    y_col = y_col_rotated
                    
                x_col = np.cross(y_col, z_col)
                
                R_m_ee_actual = np.column_stack((x_col, y_col, z_col))
                R_ee_m_actual = R_m_ee_actual.T

        # 4. 오일러 각도 추출
        # 기준 행렬이 +90도를 기반으로 구축되었으므로, ZYX 분해 시 자연스럽게 +90도 근처의 값이 도출됩니다.
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        if arm_side == "right" and yaw_e < 0:
            yaw_e += 360.0

        # 5. 평행이동 오프셋 계산 (Least-Squares Solver allowing small attachment errors)
        radius_6 = marker_data_6.get('radius', 0.0)
        radius_5 = marker_data_5.get('radius', 0.0)
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0
        
        # Load nominal values
        version_suffix = "_v13" if self.is_v13() else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            
        x_nom = tf_vec[0] * 1000.0 if tf_vec is not None else 0.0
        y_nom = tf_vec[1] * 1000.0 if tf_vec is not None else (77.5 if arm_side == "left" else -77.5)
        z_nom = tf_vec[2] * 1000.0 if tf_vec is not None else -66.77
        
        # In v1.2, we assume J5/J6 joint offsets are already zero/corrected
        opt_delta_5_rad = 0.0
        opt_delta_6_rad = 0.0
        
        from scipy.optimize import least_squares
        def residuals_trans(params):
            xe, ye, ze = params
            r6_pred = np.sqrt(ye**2 + ze**2)
            Z_prime = ye * np.sin(opt_delta_6_rad) + ze * np.cos(opt_delta_6_rad) + L_5_ee
            r5_pred = np.sqrt(xe**2 + Z_prime**2)
            res = [
                r6_pred - radius_6,
                r5_pred - radius_5
            ]
            if marker_data_4 is not None:
                Y_prime = ye * np.cos(opt_delta_6_rad) - ze * np.sin(opt_delta_6_rad)
                r4_pred = np.sqrt((xe * np.cos(opt_delta_5_rad) + Z_prime * np.sin(opt_delta_5_rad))**2 + Y_prime**2)
                res.append(r4_pred - radius_4)
                
            reg_weight = 1e-7
            res.append(reg_weight * (xe - x_nom))
            res.append(reg_weight * (ye - y_nom))
            res.append(reg_weight * (ze - z_nom))
            return res
            
        initial_guess = [x_nom, y_nom, z_nom]
        opt_res = least_squares(residuals_trans, initial_guess, loss='huber')
        x_e, y_e, z_e = opt_res.x

        # 6. 알고리즘 신뢰도 평가 점수
        dot_val = np.dot(n6_marker_actual, n5_marker_actual)
        ortho_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0))))
        
        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
        
        return {
            'converged': True,
            'x_e': x_e, 'y_e': y_e, 'z_e': z_e,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee, 'radius_6': radius_6, 'radius_5': radius_5,
            'radius_4': marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0,
            'ortho_err': ortho_err,
            'rmse_6': marker_data_6.get('rmse', 0.0),
            'rmse_5': marker_data_5.get('rmse', 0.0),
            'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 is not None else 0.0,
            'rot_err_deg': rot_err_deg, 'tilt_diff': 0.0,
            'warn_large_angle': rot_err_deg > 15.0
        }

    def generate_marker_plot(self, res_5, res_6, res_4, unified_res, arm_side, is_v13, save_path):
        """
        Generates unified marker calibration plots and saves the image to disk.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def plot_single_axis(ax, res, axis_num, color):
            if res is None or 'pts_2d' not in res:
                return
            ax.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c=color, label='Captured Points')
            circle = plt.Circle((res['uc_opt'], res['vc_opt']), res['radius'], color='r', fill=False, label='Fitted Circle')
            ax.add_patch(circle)
            ax.plot(res['uc_opt'], res['vc_opt'], 'rx', label='Center')
            
            x_min, x_max = res['pts_2d'][:, 0].min(), res['pts_2d'][:, 0].max()
            y_min, y_max = res['pts_2d'][:, 1].min(), res['pts_2d'][:, 1].max()
            span = max(x_max - x_min, y_max - y_min)
            margin = max(1.0, span * 0.5)
            cx = (x_max + x_min) / 2
            cy = (y_max + y_min) / 2
            ax.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
            ax.set_ylim(cy - span/2 - margin, cy + span/2 + margin)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f"Axis {axis_num} Sweep (Radius: {res['radius']:.2f}mm, RMSE: {res['rmse']:.3f})")
            ax.legend()

        # Plot results
        if is_v13 and res_4 is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            plot_single_axis(ax1, res_6, 6, 'blue')
            plot_single_axis(ax2, res_5, 5, 'green')
            plot_single_axis(ax3, res_4, 4, 'purple')
            fig.suptitle(f"Unified Marker Sweep Results ({arm_side.upper()} Arm)\n"
                         f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                         f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°\n"
                         f"Opt d5: {unified_res.get('opt_delta_5', 0.0):.3f}° | Opt d6: {unified_res.get('opt_delta_6', 0.0):.3f}° | Min Radius: {unified_res.get('min_radius', 0.0):.2f} mm", fontsize=12, fontweight='bold')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plot_single_axis(ax1, res_6, 6, 'blue')
            plot_single_axis(ax2, res_5, 5, 'green')
            fig.suptitle(f"Unified Marker Sweep Results ({arm_side.upper()} Arm)\n"
                         f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                         f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            return True
        except Exception as e:
            logging.warning(f"[generate_marker_plot] Failed to save plot: {e}")
            return False
        finally:
            plt.close()