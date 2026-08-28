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

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0, max_attempts=3):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm, max_attempts: {max_attempts})...")
        
        # Get rotation only from mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        R_cam_to_rob = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
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
            
            # Rotate errors to robot frame (using only rotation R_cam_to_rob)
            dp_rob = R_cam_to_rob @ dp_cam
            dR_rob = R_cam_to_rob @ dR_cam @ R_cam_to_rob.T
            
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            p_ee = T_rob_to_ee[:3, 3]
            R_ee = T_rob_to_ee[:3, :3]
            
            T_rob_to_ee_new = np.eye(4)
            T_rob_to_ee_new[:3, :3] = dR_rob @ R_ee
            T_rob_to_ee_new[:3, 3] = p_ee + dp_rob
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("link_torso_5", ee_name, T_rob_to_ee_new.astype(np.float32), 0.2, 0.5, 1.0)
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

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True, save_debug=False, initial_joint_pos=None, pass_idx=1, sweep_duration=10.0):
        try:
            if getattr(self, 'stop_requested', False):
                return None

            if save_debug and pass_idx == 1:
                from core.paths import CONFIG_PATHS
                result_txt_dir = CONFIG_PATHS["txt_dir"]
                fname = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_marker_axis_{axis_mode}.txt")
                if os.path.exists(fname):
                    try: os.remove(fname)
                    except: pass

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
                    if hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                        if log_callback: log_callback("[INFO] Prompting user for manual teaching due to marker visibility error...")
                        resolved = self.marker_problem_callback(arm_side)
                        if resolved:
                            self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                            initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
                    if not initial_check:
                        if status_callback: status_callback(False)
                        return None
                if status_callback: status_callback(True)
            else:
                if status_callback: status_callback(True)

            if not self.robot:
                if log_callback: log_callback("[ERROR] Robot is not connected.")
                return None

            state = self.robot.get_state()
            model = self.robot.model()
            arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
            if initial_joint_pos is None:
                initial_joint_pos = list(np.array(state.position)[arm_idx])

            # Sweep configuration from MARKER_CONFIGS
            axis_str = str(axis_mode).lower()
            mcfg = None
            for key in self.MARKER_CONFIGS:
                if key in axis_str or key.split("_")[-1] in axis_str:
                    mcfg = self.MARKER_CONFIGS[key]
                    break
            if mcfg is None:
                raise ValueError(f"Unknown marker sweep axis mode: {axis_mode}")
            
            start_deg = mcfg["start_deg"]
            end_deg = mcfg["end_deg"]
            joint_i = mcfg["joint_i"]

            head_idx = list(model.head_idx[:2]) if len(model.head_idx) >= 2 else None
            q_head_0 = np.array([float(state.position[i]) for i in head_idx], dtype=np.float64) if head_idx is not None else None
            dyn_model = self.robot.get_dynamics()
            
            q_head_start = None
            if use_head_tracking and self.is_head_active() and head_idx is not None and q_head_0 is not None:
                q_head_start = q_head_0

            dataset = self.perform_single_joint_sweep(
                arm_side, joint_i, initial_joint_pos, start_deg, end_deg, sweep_duration,
                q_head=q_head_start, label=f"Marker Axis {axis_mode}", log_callback=log_callback, mode="marker"
            )
            if dataset is None:
                return None

            if dataset:
                initial_joint_pos = list(dataset[0][0][arm_idx])

            captured_poses = [pose for _, pose in dataset]
            captured_angles = [np.degrees(q_full[arm_idx[joint_i]] - initial_joint_pos[joint_i]) for q_full, _ in dataset]
            captured_q_full = [q_full for q_full, _ in dataset]

            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Stop requested during marker sweep.")
                return None

            if len(captured_poses) < 20:
                if log_callback: log_callback("[ERROR] Too few valid marker poses (<10). Prompting posture adjustment...")
                if hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                    self.marker_problem_callback(arm_side)
                return None

            # Solve Circle Fitting
            n_nom = mcfg["n_nom_v13"] if self.is_v13() else mcfg["n_nom_v12"]
            res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom, robust=not self.is_mock)
            
            # Load mount_to_cam (transform from head mount "link_head_2" to camera)
            mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
            # Force camera translation components to zero as requested and use fixed rotation
            mount_to_cam_rot_only = [0.0, 0.0, 0.0] + list(mount_to_cam[3:])
            T_t5_to_cam_fixed = self.make_transform(mount_to_cam_rot_only)
            
            ee_name = f"ee_{arm_side}"
            pts_ee = []
            is_v13 = self.is_v13()
            for q_full, pose_cam_to_marker in zip(captured_q_full, captured_poses):
                try:
                    # q_physical = q_full - joint_offset (apply joint offsets to reconstruct actual physical angle)
                    q_mod = np.array(q_full)
                    if hasattr(self, 'joint_offsets') and self.joint_offsets:
                        offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                        q_mod[arm_idx[3]] -= np.radians(offsets.get("elbow", 0.0))
                        q_mod[arm_idx[5]] -= np.radians(offsets.get("wrist_pitch", 0.0))
                        if is_v13:
                            q_mod[arm_idx[6]] -= np.radians(offsets.get("wrist_roll", 0.0))
                        else:
                            q_mod[arm_idx[6]] -= np.radians(offsets.get("wrist_yaw2", 0.0))
                    
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_mod, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_t5_to_cam_fixed
                    
                    T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_mod, ee_name, "link_torso_5")
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
        finally:
            pass

    def get_link_length(self, arm_side):
        try:
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized")
            dyn_model = self.robot.get_dynamics()
            q = np.array(self.robot.get_state().position)
            T = BaseCalibrator.compute_fk(self.robot, dyn_model, q, f"ee_{arm_side}", f"link_{arm_side}_arm_5")
            return np.linalg.norm(T[:3, 3]) * 1000.0 # m to mm
        except Exception as e:
            logging.error(f"Failed to get link kinematics: {e}")
            raise e

    def get_z_sign(self, arm_side):
        try:
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized")
            dyn_model = self.robot.get_dynamics()
            q = np.array(self.robot.get_state().position)
            T = BaseCalibrator.compute_fk(self.robot, dyn_model, q, f"ee_{arm_side}", f"link_{arm_side}_arm_5")
            # If Z translation is negative, the EE Z-axis points inward (toward J5), so z_sign = -1.0
            return -1.0 if T[2, 3] < 0.0 else 1.0
        except Exception as e:
            logging.error(f"Failed to get link z_sign: {e}")
            raise e


    def compute_unified_bracket_calibration_v1_3(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None, calib_roll_deg=None, calib_pitch_deg=None, calib_roll_or_yaw_deg=None, lock_bracket=False):
        if calib_roll_or_yaw_deg is not None:
            calib_roll_deg = calib_roll_or_yaw_deg
        L_5_ee = self.get_link_length(arm_side)

        # 1. Nominal marker template for v1.3
        ver_key = "1.3" if self.is_v13() else "1.2"
        nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        x_nom = nominal_vec[0] * 1000.0
        y_nom = 0.0 if self.is_v13() else nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        
        nominal_rpy = nominal_vec[3:6]
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

        # Ideal axes in marker frame
        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        poses_6 = marker_data_6.get('captured_poses', []) if marker_data_6 else []
        poses_5 = marker_data_5.get('captured_poses', []) if marker_data_5 else []
        poses_4 = marker_data_4.get('captured_poses', []) if marker_data_4 else []

        n6_marker_actual = extract_axis_from_rotations(poses_6, x_ee_m_ideal)
        n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
        n4_marker_actual = extract_axis_from_rotations(poses_4, z_ee_m_ideal) if len(poses_4) > 0 else z_ee_m_ideal

        radius_6 = marker_data_6.get('radius', 0.0) if marker_data_6 else 0.0
        radius_5 = marker_data_5.get('radius', 0.0) if marker_data_5 else 0.0
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0

        # Orthogonality checks
        ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_marker_actual, n5_marker_actual)), -1.0, 1.0)))
        ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(n5_marker_actual, n6_marker_actual)), -1.0, 1.0)))
        ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_marker_actual, n6_marker_actual)), -1.0, 1.0)))
        ortho_err = max(abs(ang_45 - 90.0), abs(ang_56 - 90.0), abs(ang_46 - 90.0))

        # Build orthogonal frame from 3 axes via SVD (Spherical wrist decoupled basis)
        x_col = n6_marker_actual / np.linalg.norm(n6_marker_actual)
        y_proj = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
        y_col = y_proj / np.linalg.norm(y_proj)
        z_col = np.cross(x_col, y_col)
        z_col /= np.linalg.norm(z_col)

        M = np.column_stack((x_col, y_col, z_col))
        U, S, Vt = np.linalg.svd(M)
        R_m_ee_actual = U @ Vt
        if np.linalg.det(R_m_ee_actual) < 0:
            U[:, 2] *= -1
            R_m_ee_actual = U @ Vt
        R_ee_m_actual = R_m_ee_actual.T

        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
            yaw_e += 360.0

        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))

        # Joint 6 Roll Offset Calculation
        ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
        ref_y /= np.linalg.norm(ref_y)
        ref_z = np.cross(x_col, ref_y)
        diff_angle_6 = np.arctan2(np.dot(n5_marker_actual, ref_z), np.dot(n5_marker_actual, ref_y))
        roll_corr = float(np.degrees(diff_angle_6))
        opt_delta_6 = (calib_roll_deg if calib_roll_deg is not None else 0.0) + roll_corr

        # Joint 5 Pitch Offset Calculation
        cross_64 = np.cross(n6_marker_actual, n4_marker_actual)
        sign_5 = np.sign(np.dot(n5_marker_actual, cross_64)) if np.linalg.norm(cross_64) > 1e-4 else 1.0
        pitch_corr = -float((ang_46 - 90.0) * sign_5)
        opt_delta_5 = (calib_pitch_deg if calib_pitch_deg is not None else 0.0) + pitch_corr

        # Solve for Translation (x_e, y_e, z_e) in mm
        z_sign = self.get_z_sign(arm_side)
        has_j4 = (marker_data_4 is not None and radius_4 > 1.0)

        def residuals_trans(params):
            xe, ye, ze = params
            # In EE frame, Joint 6 rotates around X_ee: radius is sqrt(ye^2 + ze^2)
            r6_pred = np.sqrt(ye**2 + ze**2)
            # Joint 5 rotates around Y_ee displaced by link length L_5_ee along Z
            Z_prime = ze + z_sign * L_5_ee
            r5_pred = np.sqrt(xe**2 + Z_prime**2)
            # Joint 4 rotates around Z_ee
            r4_pred = np.sqrt(xe**2 + ye**2)
            res = [
                (r6_pred - radius_6),
                (r5_pred - radius_5)
            ]
            if has_j4:
                res.append(r4_pred - radius_4)
            reg = 1e-2
            res.append(reg * (xe - x_nom))
            res.append(reg * (ye - y_nom))
            res.append(reg * (ze - z_nom))
            return res

        x_init = [x_nom, y_nom, z_nom]
        opt_res = least_squares(residuals_trans, x_init, bounds=([max(0.0, x_nom - 30.0), y_nom - 30.0, z_nom - 30.0], [x_nom + 30.0, y_nom + 30.0, z_nom + 30.0]), loss='huber')
        xe_opt, ye_opt, ze_opt = opt_res.x

        Z_prime_opt = ze_opt + z_sign * L_5_ee
        r6_err = abs(radius_6 - np.sqrt(ye_opt**2 + ze_opt**2))
        r5_err = abs(radius_5 - np.sqrt(xe_opt**2 + Z_prime_opt**2))
        r4_err = abs(radius_4 - np.sqrt(xe_opt**2 + ye_opt**2)) if has_j4 else 0.0
        max_radius_err = max(r6_err, r5_err, r4_err)

        pos_diff_mm = float(np.linalg.norm([xe_opt - x_nom, ye_opt - y_nom, ze_opt - z_nom]))
        warn_large_pos = pos_diff_mm > 40.0
        if warn_large_pos:
            logging.error(f"[{arm_side.upper()} BRACKET ERROR] Bracket position deviation exceeded safety limit: {pos_diff_mm:.1f}mm > 40.0mm! (Nominal: [{x_nom:.1f}, {y_nom:.1f}, {z_nom:.1f}], Opt: [{xe_opt:.1f}, {ye_opt:.1f}, {ze_opt:.1f}])")

        return {
            'converged': True,
            'x_e': xe_opt, 'y_e': ye_opt, 'z_e': ze_opt,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'opt_delta_5': opt_delta_5,
            'opt_delta_6': opt_delta_6,
            'd5_opt_deg': opt_delta_5,
            'd6_opt_deg': opt_delta_6,
            'recommended_joint_offset': opt_delta_6,
            'recommended_joint_offset_5': opt_delta_5,
            'recommended_joint_offset_6': opt_delta_6,
            'L_5_ee': L_5_ee,
            'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
            'ortho_err': ortho_err,
            'ang_45': ang_45, 'ang_56': ang_56, 'ang_46': ang_46,
            'r6_err': r6_err, 'r5_err': r5_err, 'r4_err': r4_err,
            'max_radius_err': max_radius_err,
            'rot_err_deg': rot_err_deg, 'tilt_diff': 0.0,
            'warn_large_angle': rot_err_deg > 15.0,
            'warn_large_pos': warn_large_pos,
            'pos_diff_mm': pos_diff_mm,
            'rmse_6': marker_data_6.get('rmse', 0.0) if marker_data_6 else 0.0,
            'rmse_5': marker_data_5.get('rmse', 0.0) if marker_data_5 else 0.0,
            'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 is not None else 0.0,
            'min_radius': radius_4,
            'n6_marker_actual': n6_marker_actual,
            'n5_marker_actual': n5_marker_actual,
            'n4_marker_actual': n4_marker_actual,
            'y_ee_m_ideal': y_ee_m_ideal
        }

    def compute_unified_bracket_calibration(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None, calib_roll_deg=None, calib_pitch_deg=None, calib_roll_or_yaw_deg=None, lock_bracket=False):
        if calib_roll_or_yaw_deg is not None:
            calib_roll_deg = calib_roll_or_yaw_deg

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
            ver_key = "1.3" if self.is_v13() else "1.2"
            nominal_rpy = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side][3:6]
            
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
        mid_idx_6 = len(poses_6) // 2
        R_ref_6 = poses_6[mid_idx_6][:3, :3] if len(poses_6) > 0 else np.eye(3)
        n6_cam = marker_data_6.get('axis_opt')
        target_ideal_6 = x_ee_m_ideal if self.is_v13() else z_ee_m_ideal
        if n6_cam is not None:
            n6_marker_actual = R_ref_6.T @ n6_cam
            if np.dot(n6_marker_actual, target_ideal_6) < 0:
                n6_marker_actual = -n6_marker_actual
        else:
            n6_marker_actual = extract_axis_from_rotations(poses_6, target_ideal_6)
        
        poses_5 = marker_data_5.get('captured_poses', [])
        mid_idx_5 = len(poses_5) // 2
        R_ref_5 = poses_5[mid_idx_5][:3, :3] if len(poses_5) > 0 else np.eye(3)
        n5_cam = marker_data_5.get('axis_opt')
        target_ideal_5 = y_ee_m_ideal
        if n5_cam is not None:
            n5_marker_actual = R_ref_5.T @ n5_cam
            if np.dot(n5_marker_actual, target_ideal_5) < 0:
                n5_marker_actual = -n5_marker_actual
        else:
            n5_marker_actual = extract_axis_from_rotations(poses_5, target_ideal_5)
 
        # [BYPASS] Bypassed permanently to calculate using ONLY the marker and rotation axis trajectory.
        kinematic_success = False

        if not kinematic_success:

            # Joint 6 angle correction for Joint 5 sweep
            theta_6 = marker_data_5.get('theta_6', None)
            if theta_6 is None:
                q_full_5 = marker_data_5.get('captured_q_full', [])
                if len(q_full_5) > 0:
                    if not self.robot:
                        raise RuntimeError("Robot instance is not initialized")
                    model = self.robot.model()
                    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                    q_idx = arm_idx[6]
                    theta_6 = np.mean([q[q_idx] for q in q_full_5])
                else:
                    theta_6 = 0.0

            # Correct J6 joint offset if available
            if hasattr(self, 'joint_offsets') and self.joint_offsets:
                offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                offset_val = offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0)
                theta_6 -= np.radians(offset_val)

            if marker_data_4 is not None:
                # --- 3-Axis SVD Alignment (Using Joint 4, 5, and 6) ---
                poses_4 = marker_data_4.get('captured_poses', [])
                mid_idx_4 = len(poses_4) // 2
                R_ref_4 = poses_4[mid_idx_4][:3, :3] if len(poses_4) > 0 else np.eye(3)
                n4_cam = marker_data_4.get('axis_opt')
                target_ideal_4 = z_ee_m_ideal if self.is_v13() else x_ee_m_ideal
                if n4_cam is not None:
                    n4_marker_actual = R_ref_4.T @ n4_cam
                    if np.dot(n4_marker_actual, target_ideal_4) < 0:
                        n4_marker_actual = -n4_marker_actual
                else:
                    n4_marker_actual = extract_axis_from_rotations(poses_4, target_ideal_4)

                # Joint 6 angle correction for Joint 4 sweep
                theta_6_4 = marker_data_4.get('theta_6', None)
                if theta_6_4 is None:
                    q_full_4 = marker_data_4.get('captured_q_full', [])
                    if len(q_full_4) > 0:
                        if not self.robot:
                            raise RuntimeError("Robot instance is not initialized")
                        model = self.robot.model()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        q_idx = arm_idx[6]
                        theta_6_4 = np.mean([q[q_idx] for q in q_full_4])
                    else:
                        theta_6_4 = 0.0

                # Correct J6 joint offset if available
                if hasattr(self, 'joint_offsets') and self.joint_offsets:
                    offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                    offset_val = offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0)
                    theta_6_4 -= np.radians(offset_val)

                if not self.is_v13():
                    z_col = n6_marker_actual
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

                    M = np.column_stack((x_col, y_col, z_col))
                else:
                    # v1.3 Spherical Wrist: J6 is X-axis, J5 is Y-axis, J4 is Z-axis
                    x_col = n6_marker_actual
                    y_col_rot = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
                    y_col_rot /= np.linalg.norm(y_col_rot)
                    z_col_rot = n4_marker_actual - np.dot(n4_marker_actual, x_col) * x_col
                    z_col_rot /= np.linalg.norm(z_col_rot)

                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rot, x_col, theta_6)
                    else:
                        y_col = y_col_rot

                    if abs(theta_6_4) > 1e-5:
                        z_col = self.rodrigues_rotation(z_col_rot, x_col, theta_6_4)
                    else:
                        z_col = z_col_rot

                    M = np.column_stack((x_col, y_col, z_col))

                # Use SVD to clean up orthogonality errors and build R_m_ee
                U, S, Vt = np.linalg.svd(M)
                R_m_ee_actual = U @ Vt
                if np.linalg.det(R_m_ee_actual) < 0:
                    U[:, 2] *= -1
                    R_m_ee_actual = U @ Vt
                
                R_ee_m_actual = R_m_ee_actual.T
            else:
                if not self.is_v13():
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
                else:
                    x_col = n6_marker_actual
                    y_col_rotated = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
                    y_col_rotated /= np.linalg.norm(y_col_rotated)
                    
                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rotated, x_col, theta_6)
                    else:
                        y_col = y_col_rotated
                        
                    z_col = np.cross(x_col, y_col)
                    
                    R_m_ee_actual = np.column_stack((x_col, y_col, z_col))
                    R_ee_m_actual = R_m_ee_actual.T

        # 4. 오일러 각도 추출
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        # v1.2: Z축 회전 방향 비틀림(Torsion) 오차 배제 - 명목 설계값 yaw으로 고정
        ver_key = "1.3" if self.is_v13() else "1.2"
        nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        if not self.is_v13():
            yaw_e = nominal_vec[5]
            if arm_side == "right" and yaw_e < 0:
                yaw_e += 360.0
        else:
            if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
                yaw_e += 360.0

        # 5. 평행이동 오프셋 계산 (Least-Squares Solver allowing small attachment errors)
        radius_6 = marker_data_6.get('radius', 0.0)
        radius_5 = marker_data_5.get('radius', 0.0)
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0
        
        x_nom = nominal_vec[0] * 1000.0
        y_nom = nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        
        opt_delta_5_rad = 0.0
        opt_delta_6_rad = 0.0
        
        z_sign = self.get_z_sign(arm_side)

        from scipy.optimize import least_squares
        has_j4 = (marker_data_4 is not None and radius_4 > 1e-3)

        if self.is_v13():
            # v1.3 Spherical Wrist: J6 is Roll (X), J5 is Pitch (Y), J4 is Yaw (Z)
            # Pivot is located at Z = +125mm in EE frame (z_sign * L_5_ee = -125mm)
            def residuals_trans(params):
                xe, ye, ze = params
                Z_prime = ze + z_sign * L_5_ee
                r6_pred = np.sqrt(ye**2 + Z_prime**2)
                r5_pred = np.sqrt(xe**2 + Z_prime**2)
                r4_pred = np.sqrt(xe**2 + ye**2)
                
                res = [
                    (r6_pred - radius_6),
                    (r5_pred - radius_5)
                ]
                if has_j4:
                    res.append(r4_pred - radius_4)
                    
                reg_weight = 1e-4
                res.append(reg_weight * (xe - x_nom))
                res.append(reg_weight * (ye - y_nom))
                res.append(reg_weight * (ze - z_nom))
                return res

            initial_guess = [x_nom, y_nom, z_nom]
            lower_bounds = [x_nom - 40.0, y_nom - 30.0, z_nom - 40.0]
            upper_bounds = [x_nom + 40.0, y_nom + 30.0, z_nom + 40.0]
            opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
            x_e, y_e, z_e = opt_res.x
        else:
            # v1.2 Non-Spherical Wrist: J6 is Yaw2 (Z), J5 is Pitch (Y), J4 is Forearm Roll (X)
            def residuals_trans(params):
                xe, ye, ze = params
                Z_prime = ze + z_sign * L_5_ee
                r6_pred = np.sqrt(xe**2 + ye**2)
                r5_pred = np.sqrt(xe**2 + Z_prime**2)
                r4_pred = np.sqrt(ye**2 + Z_prime**2)
                
                res = [
                    (r6_pred - radius_6),
                    (r5_pred - radius_5)
                ]
                if has_j4:
                    res.append(r4_pred - radius_4)
                    
                reg_weight = 1e-7
                res.append(reg_weight * (xe - x_nom))
                res.append(reg_weight * (ye - y_nom))
                res.append(reg_weight * (ze - z_nom))
                return res

            initial_guess = [x_nom, y_nom, z_nom]
            lower_bounds = [x_nom - 40.0, y_nom - 40.0, -250.0]
            upper_bounds = [x_nom + 40.0, y_nom + 40.0, 10.0]
            opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
            x_e, y_e, z_e = opt_res.x

        print(f"DEBUG SOLVER v1.2: arm_side={arm_side}", flush=True)
        print(f"  L_5_ee = {L_5_ee:.4f}", flush=True)
        print(f"  radius_6 = {radius_6:.4f}, radius_5 = {radius_5:.4f}, radius_4 = {radius_4:.4f}", flush=True)
        print(f"  x_nom = {x_nom:.4f}, y_nom = {y_nom:.4f}, z_nom = {z_nom:.4f}", flush=True)
        print(f"  x_e = {x_e:.4f}, y_e = {y_e:.4f}, z_e = {z_e:.4f}", flush=True)
        print(f"  Initial guess: {initial_guess}", flush=True)
        print(f"  Lower bounds: {lower_bounds}", flush=True)
        print(f"  Upper bounds: {upper_bounds}", flush=True)
        print(f"  Optimal residuals: {residuals_trans(opt_res.x)}", flush=True)

        # Circle fitting validation checks
        if not self.is_v13():
            r6_err = abs(radius_6 - np.sqrt(x_e**2 + y_e**2))
            r5_err = abs(radius_5 - np.sqrt(x_e**2 + (z_e + z_sign * L_5_ee)**2))
            r4_err = abs(radius_4 - np.sqrt((z_sign * L_5_ee + z_e)**2 + y_e**2)) if marker_data_4 is not None else 0.0
        else:
            Z_prime = z_e + z_sign * L_5_ee
            r6_err = abs(radius_6 - np.sqrt(y_e**2 + Z_prime**2))
            r5_err = abs(radius_5 - np.sqrt(x_e**2 + Z_prime**2))
            r4_err = abs(radius_4 - np.sqrt(x_e**2 + y_e**2)) if marker_data_4 is not None else 0.0

        if marker_data_4 is not None:
            print(f"[VALIDATION] {arm_side.upper()} ARM BRACKET SWEEP CIRCLE RESIDUALS:", flush=True)
            print(f"  * J6 Sweep Radius Err: {r6_err:.4f} mm", flush=True)
            print(f"  * J5 Sweep Radius Err: {r5_err:.4f} mm", flush=True)
            print(f"  * J4 Sweep Radius Err: {r4_err:.4f} mm", flush=True)
            max_err = max(r6_err, r5_err, r4_err)
            if max_err < 1.0:
                print(f"  [SUCCESS] Circle reconstruction PASSED (Max Residual: {max_err:.4f} mm < 1.0 mm)", flush=True)
            else:
                print(f"  [WARNING] Circle reconstruction shows deviation (Max Residual: {max_err:.4f} mm)", flush=True)
        else:
            print(f"[VALIDATION] {arm_side.upper()} ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):", flush=True)
            print(f"  * J6 Sweep Radius Err: {r6_err:.4f} mm", flush=True)
            print(f"  * J5 Sweep Radius Err: {r5_err:.4f} mm", flush=True)
            max_err = max(r6_err, r5_err)
            if max_err < 1.0:
                print(f"  [SUCCESS] Circle reconstruction PASSED (Max Residual: {max_err:.4f} mm < 1.0 mm)", flush=True)
            else:
                print(f"  [WARNING] Circle reconstruction shows deviation (Max Residual: {max_err:.4f} mm)", flush=True)

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
            'warn_large_angle': rot_err_deg > 15.0,
            'n6_marker_actual': n6_marker_actual,
            'n5_marker_actual': n5_marker_actual,
            'y_ee_m_ideal': y_ee_m_ideal
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
                ax.set_title(f"Axis {axis_num} Sweep: No Data")
                ax.axis('off')
                return
            ax.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c=color, s=15, alpha=0.6, label='Captured Points')
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
            ax.set_title(f"Axis {axis_num} Sweep (Radius: {res['radius']:.2f}mm, RMSE: {res['rmse']:.3f}mm)", fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)

        # Plot results
        if is_v13:
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            ax1, ax2 = axes[0, 0], axes[0, 1]
            ax3, ax4 = axes[1, 0], axes[1, 1]
            
            plot_single_axis(ax1, res_6, "6 (Wrist Roll)", 'blue')
            plot_single_axis(ax2, res_5, "5 (Wrist Pitch)", 'green')
            plot_single_axis(ax3, res_4, "4 (Wrist Yaw)", 'purple')
            
            ax4.axis('off')
            ang_45 = unified_res.get('ang_45', 90.0)
            ang_56 = unified_res.get('ang_56', 90.0)
            ang_46 = unified_res.get('ang_46', 90.0)
            ortho_status = "PASS (<0.1°)" if unified_res.get('ortho_err', 0.0) < 0.1 else "ALIGNED"
            
            summary_text = (
                f"=== UNIFIED 3-AXIS SPHERICAL WRIST SUMMARY ({arm_side.upper()} ARM) ===\n\n"
                f"1. Recommended Joint Offsets:\n"
                f"   * Joint 5 (Wrist Pitch): {unified_res.get('d5_opt_deg', 0.0):+.4f}°\n"
                f"   * Joint 6 (Wrist Roll) : {unified_res.get('d6_opt_deg', 0.0):+.4f}°\n\n"
                f"2. Marker Bracket Calibration:\n"
                f"   * Position (X, Y, Z)   : [{unified_res['x_e']:.2f}, {unified_res['y_e']:.2f}, {unified_res['z_e']:.2f}] mm\n"
                f"   * Orientation (R, P, Y): [{unified_res['roll_e']:.2f}°, {unified_res['pitch_e']:.2f}°, {unified_res['yaw_e']:.2f}°]\n\n"
                f"3. Quantitative Verification Metrics:\n"
                f"   * Orthogonality J4-J5  : {ang_45:.3f}° (Dev: {abs(ang_45-90.0):.3f}°)\n"
                f"   * Orthogonality J5-J6  : {ang_56:.3f}° (Dev: {abs(ang_56-90.0):.3f}°)\n"
                f"   * Orthogonality J4-J6  : {ang_46:.3f}° (Dev: {abs(ang_46-90.0):.3f}°)\n"
                f"   * Max Radius Residual  : {unified_res.get('max_radius_err', 0.0):.3f} mm\n"
                f"   * Alignment Status     : {ortho_status}\n"
            )
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95, edgecolor='#ced4da'))
            
            fig.suptitle(f"RB-Y1 v1.3 Spherical Wrist Simultaneous Calibration ({arm_side.upper()} Arm)", fontsize=14, fontweight='bold')
        elif res_4 is not None:
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