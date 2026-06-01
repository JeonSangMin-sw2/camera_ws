import time
import logging
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy


class MarkerCalibrator:
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot

    @staticmethod
    def initialize_robot(address, model, power=".*", servo=".*"):
        robot = rby.create_robot(address, model)
        if not robot.connect():
            logging.error(f"Failed to connect robot {address}")
            return None
        if not robot.is_power_on(power):
            logging.info(f"Turning power ({power}) on...")
            if not robot.power_on(power):
                logging.error(f"Failed to turn power ({power}) on")
                return None
        else:
            logging.info(f"Power ({power}) is already ON.")

        if not robot.is_servo_on(servo):
            logging.info(f"Turning servo ({servo}) on...")
            if not robot.servo_on(servo):
                logging.error(f"Failed to servo ({servo}) on")
                return None
        else:
            logging.info(f"Servo ({servo}) is already ON.")

        cm_state = robot.get_control_manager_state()
        if cm_state.state in [
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ]:
            logging.warning(f"Control manager is in fault state: {cm_state.state}. Resetting...")
            if not robot.reset_fault_control_manager():
                logging.error(f"Failed to reset control manager")
                return None
        
        if cm_state.state != rby.ControlManagerState.State.Enabled:
            logging.info("Enabling control manager...")
            if not robot.enable_control_manager():
                logging.error(f"Failed to enable control manager")
                return None
        else:
            logging.info("Control manager is already enabled.")
        return robot

    @staticmethod
    def terminate_robot(robot):
        if robot:
            try:
                robot.disconnect()
                return True
            except Exception as e:
                logging.error(f"Failed to disconnect robot: {e}")
        return False

    @staticmethod
    def compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        model = robot.model()
        state = dyn_model.make_state([base_link, ee_link], model.robot_joint_names)
        state.set_q(q)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        return T

    @staticmethod
    def make_transform(data):
        """
        Creates a 4x4 transformation matrix from [x, y, z, roll, pitch, yaw].
        Coordinates in meters, angles in degrees (ZYX Euler).
        """
        T = np.eye(4)
        T[:3, 3] = data[:3]
        # data[3:] is [roll, pitch, yaw], but scipy 'ZYX' expects [yaw, pitch, roll]
        yaw = data[5]
        pitch = data[4]
        roll = data[3]
        T[:3, :3] = R_scipy.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        return T




    @staticmethod
    def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0):
        if not robot:
            return False
            
        body_cmd = rby.BodyComponentBasedCommandBuilder()
        if torso is not None:
            body_cmd.set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(torso)
            )
        if right_arm is not None:
            body_cmd.set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
        if left_arm is not None:
            body_cmd.set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_arm)
            )

        cmd = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
        if head is not None:
            cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )

        rv = robot.send_command(
            rby.RobotCommandBuilder().set_command(cmd),
            1,
        ).get()

        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            logging.error("Failed to conduct movej.")
            return False

        return True

    @staticmethod
    def rodrigues_rotation(vector, axis, theta_rad):
        axis = axis / np.linalg.norm(axis)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return vector * cos_t + np.cross(axis, vector) * sin_t + axis * np.dot(axis, vector) * (1 - cos_t)

    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False):
        """
        Fits a circle directly in 3D space using encoder angles and RANSAC/Huber loss,
        and computes the full 6-DOF misalignment (Tilt and Twist/Yaw).
        
        Args:
            relative_poses (list or np.ndarray): N x 4 x 4 matrices (translation in mm)
            captured_angles (list or np.ndarray): N encoder angles in degrees
            axis_prior (list or np.ndarray): 3D nominal axis direction
            return_plot_data (bool): Whether to return 2D projection data for UI plotting
        """
        points = np.array([T[:3, 3] for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # --- Robust Iterative Outlier Rejection ---
        inlier_mask = np.ones(len(points), dtype=bool)
        
        # 1. Initial rotation axis & center estimation
        if axis_prior is not None:
            best_normal = np.array(axis_prior, dtype=float)
            best_normal /= np.linalg.norm(best_normal)
        else:
            centroid = np.mean(points, axis=0)
            pts_centered = points - centroid
            _, _, vh = np.linalg.svd(pts_centered)
            best_normal = vh[2, :]
            best_normal /= np.linalg.norm(best_normal)
            
        centroid = np.mean(points, axis=0)
        if best_normal[0] < 0.9:
            ex = np.cross(best_normal, [1, 0, 0])
        else:
            ex = np.cross(best_normal, [0, 1, 0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(best_normal, ex)

        # Geometric Sagitta Formula for highly robust initialization on small arcs (like Axis 5 sweep)
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
            # Fallback to algebraic circle fit but clamp radius
            pts_centered = points - centroid
            pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
            A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
            b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            R_init = np.clip(np.sqrt(max(0.001, res[2] + uc**2 + vc**2)), 50.0, 450.0)
            c_init = centroid + uc * ex + vc * ey
        
        best_opt = None
        best_rmse = float('inf')
        best_sign = 1
        
        # Determine the encoder rotation sign (+1 or -1)
        for sign in [1, -1]:
            angles_rad = angles_rad_base * sign
            
            # Formulate parameters: [cx, cy, cz, nx, ny, nz, rx, ry, rz, R]
            r_dir_init = points[0] - c_init
            r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
            if np.linalg.norm(r_dir_init) > 1e-6:
                r_dir_init /= np.linalg.norm(r_dir_init)
            
            init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
            
            # Add physically realistic bounds to prevent divergence on noise-sensitive sweeps
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
                for pt, theta in zip(points[inlier_mask], angles_rad[inlier_mask]):
                    pred_pt = c + R * MarkerCalibrator.rodrigues_rotation(r_init, axis, theta)
                    residuals.extend(pt - pred_pt)
                    
                return np.array(residuals)
                
            opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            rmse = np.sqrt(np.mean(opt_res.fun**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_opt = opt_res
                best_sign = sign

        # Extract optimal parameters
        c_init = best_opt.x[0:3]
        best_normal = best_opt.x[3:6]
        best_normal /= np.linalg.norm(best_normal)
        r_final_dir = best_opt.x[6:9]
        r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
        if np.linalg.norm(r_final_dir) > 1e-6:
            r_final_dir /= np.linalg.norm(r_final_dir)
        R_init = best_opt.x[9]
        
        # Iterative Outlier Rejection (2 iterations)
        for iteration in range(2):
            angles_rad = angles_rad_base * best_sign
            pts_in = points[inlier_mask]
            rad_in = angles_rad[inlier_mask]
            
            if len(pts_in) < 5:
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
            
            # Recompute errors for all points
            all_errors = []
            for pt, theta in zip(points, angles_rad):
                pred_pt = c_init + R_init * MarkerCalibrator.rodrigues_rotation(r_final_dir, best_normal, theta)
                all_errors.append(np.linalg.norm(pt - pred_pt))
            all_errors = np.array(all_errors)
            
            inlier_errors = all_errors[inlier_mask]
            std_err = np.std(inlier_errors) if len(inlier_errors) > 0 else 1.0
            threshold = max(0.5, min(3.0, 2.0 * std_err))
            
            new_mask = all_errors < threshold
            if np.all(new_mask == inlier_mask) or np.sum(new_mask) < 5:
                break
            inlier_mask = new_mask
            
        # Re-orthogonalize best_normal with axis_prior to ensure sign consistency
        if axis_prior is not None:
            if np.dot(best_normal, axis_prior) < 0:
                best_normal = -best_normal
                
        # Re-project to 2D for UI plotting compatibility
        if best_normal[0] < 0.9:
            ex_final = np.cross(best_normal, [1, 0, 0])
        else:
            ex_final = np.cross(best_normal, [0, 1, 0])
        ex_final /= np.linalg.norm(ex_final)
        ey_final = np.cross(best_normal, ex_final)
        
        pts_centered_final = points - centroid
        pts_2d_final = np.dot(pts_centered_final, np.vstack((ex_final, ey_final)).T)
        uc_opt_final = np.dot(c_init - centroid, ex_final)
        vc_opt_final = np.dot(c_init - centroid, ey_final)

        # Compute tilt and twist/yaw for each frame and robustly estimate medians
        tilt_list = []
        yaw_list = []
        for T_i in relative_poses:
            R_i = T_i[:3, :3]
            axis_m_i = R_i.T @ best_normal
            tilt_i = np.degrees(np.arcsin(np.clip(axis_m_i[2], -1.0, 1.0)))
            tilt_list.append(tilt_i)
            
            vec_c_to_mi = T_i[:3, 3] - c_init
            radial_vec = vec_c_to_mi - np.dot(vec_c_to_mi, best_normal) * best_normal
            ideal_tangent = np.cross(best_normal, radial_vec)
            norm_ideal = np.linalg.norm(ideal_tangent)
            if norm_ideal > 1e-6:
                ideal_tangent /= norm_ideal
                
            marker_x = R_i[:, 0]
            marker_x_plane = marker_x - np.dot(marker_x, best_normal) * best_normal
            norm_mx = np.linalg.norm(marker_x_plane)
            if norm_mx > 1e-6:
                marker_x_plane /= norm_mx
                
            twist_cos = np.dot(marker_x_plane, ideal_tangent)
            twist_angle = np.degrees(np.arccos(np.clip(twist_cos, -1.0, 1.0)))
            if np.dot(np.cross(ideal_tangent, marker_x_plane), best_normal) < 0:
                twist_angle = -twist_angle
                
            if twist_angle > 90:
                twist_angle -= 180
            elif twist_angle < -90:
                twist_angle += 180
            yaw_list.append(twist_angle)
            
        all_errors = []
        for pt, theta in zip(points, angles_rad):
            pred_pt = c_init + R_init * MarkerCalibrator.rodrigues_rotation(r_final_dir, best_normal, theta)
            all_errors.append(np.linalg.norm(pt - pred_pt))
        all_errors = np.array(all_errors)
        final_rmse = np.sqrt(np.mean(all_errors[inlier_mask]**2))
            
        if return_plot_data:
            return c_init, best_normal, abs(R_init), final_rmse, pts_2d_final, uc_opt_final, vc_opt_final, np.median(tilt_list), np.median(yaw_list), tilt_list
            
        return c_init, best_normal, abs(R_init), final_rmse, np.median(tilt_list), np.median(yaw_list), tilt_list

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
        # RPY: [90, 0, -90] in ZYX order
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
            
            # Check convergence using combined norm (pos in mm, rot in deg)
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
            
            # Use only rotation of T_rob_to_cam to transform camera-frame errors to robot torso frame.
            # This completely avoids translation lever-arm errors.
            R_rob_to_cam = T_rob_to_cam[:3, :3]
            
            dp_cam = T_target_cam[:3, 3] - T_cam_to_marker[:3, 3]
            dR_cam = T_target_cam[:3, :3] @ T_cam_to_marker[:3, :3].T
            
            dp_rob = R_rob_to_cam @ dp_cam
            dR_rob = R_rob_to_cam @ dR_cam @ R_rob_to_cam.T
            
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            
            T_rob_to_ee_new = np.eye(4)
            T_rob_to_ee_new[:3, :3] = dR_rob @ T_rob_to_ee[:3, :3]
            T_rob_to_ee_new[:3, 3] = T_rob_to_ee[:3, 3] + dp_rob
            
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


    def perform_move_to_ready_pose(self, arm_side, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"Moving to {arm_side} Marker Ready Pose...")

        # Predefined ready pose for model 'm' (approximate center for calibration)
        # Note: These values should be adjusted based on actual hardware design
        torso = [0, 0, 0, 0, 0, 0] # link_torso_1 ~ 6
        
        if arm_side == "right":
            # Move left arm to zero first
            if log_callback: log_callback("  - Homing left arm...")
            self.movej(self.robot, torso=torso, left_arm=[0,0,0,0,0,0,0], minimum_time=4.0)
            
            # Right arm ready pose (deg): [0:90, 1:0, 2:90, 3:-90, 4:90, 5:0, 6:0]
            right_arm = np.deg2rad([-90, -45, 73, -107, 90, 90, 0])
            left_arm = [0, 0, 0, 0, 0, 0, 0]
        else:
            # Move right arm to zero first
            if log_callback: log_callback("  - Homing right arm...")
            self.movej(self.robot, torso=torso, right_arm=[0,0,0,0,0,0,0], minimum_time=4.0)
            
            right_arm = [0, 0, 0, 0, 0, 0, 0]
            # Left arm ready pose (deg): [0:90, 1:0, 2:-90, 3:90, 4:-90, 5:0, 6:0]
            left_arm = np.deg2rad([-90, 45, -73, -107, -90, 90, 0])

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, minimum_time=4.0)
        
        if success and log_callback:
            log_callback("Ready Pose Reached.")
        return success

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True):
        if log_callback:
            log_callback("\n" + "="*40)
            log_callback(f"   STARTING {axis_mode}-AXIS CALIBRATION SWEEP")
            log_callback("="*40)

        # Pre-check Marker Presence
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system (marker_st) is not initialized. Cannot perform sweep.")
            return None

        if log_callback: log_callback("  - Checking if marker is visible before starting...")
        initial_check = self.marker_st.get_marker_transform(sampling_time=1.0, side=arm_side)
        if not initial_check:
            if log_callback: log_callback("\n[ERROR] marker not detected.")
            if status_callback: status_callback(False)
            return None
        
        if status_callback: status_callback(True)
        captured_poses = []
        captured_angles = []

        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected. Cannot perform automated sweep.")
            return None

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])
        
        # Configure Sweep based on Axis Mode
        if axis_mode == 6:
            max_points = 11
            start_deg = -20
            step_deg = 4
            joint_i = 6
        else:
            max_points = 11
            # Increased sweep range to ±10 degrees to improve fitting stability
            start_deg = -10
            step_deg = 2
            joint_i = 5

        # Prepare Active Head/Camera Tracking
        head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
        q_head_0 = state.position[head_idx].copy() if (head_idx is not None and use_head_tracking) else None
        dyn_model = self.robot.get_dynamics()
        
        try:
            T_neck = self.compute_fk(self.robot, dyn_model, state.position, "link_head_2", "link_torso_5")
            p_neck = T_neck[:3, 3] if use_head_tracking else None
        except Exception:
            p_neck = None
            
        ee_name = f"ee_{arm_side}"
        try:
            T_ee_0 = self.compute_fk(self.robot, dyn_model, state.position, ee_name, "link_torso_5")
            p_marker_0 = T_ee_0[:3, 3] if use_head_tracking else None
        except Exception:
            p_marker_0 = None

        if log_callback: log_callback(f"[INFO] Initial Joint Pose: {np.round(initial_joint_pos, 2)}")
        
        for i in range(max_points):
            if log_callback: log_callback(f"\n[STEP {i + 1}/{max_points}]")
            
            target_offset_deg = start_deg + (i * step_deg)
            target_joint_pos = list(initial_joint_pos)
            target_joint_pos[joint_i] = initial_joint_pos[joint_i] + np.radians(target_offset_deg)
            
            if target_offset_deg == 0:
                if log_callback: log_callback(f"  - Moving to Center Pose (0 deg)...")
            else:
                if log_callback: log_callback(f"  - Moving axis {axis_mode} to {target_offset_deg:.1f} deg offset...")
            
            # Compute active head tracking target position
            head_q_step = None
            if use_head_tracking and q_head_0 is not None and p_neck is not None and p_marker_0 is not None:
                q_full_temp = np.array(state.position)
                if arm_side == "left":
                    q_full_temp[model.left_arm_idx] = target_joint_pos
                else:
                    q_full_temp[model.right_arm_idx] = target_joint_pos
                
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
                    
                    # Clip head angles to safe ranges (Yaw: ±25 deg, Pitch: ±20 deg relative to zero)
                    yaw_target = np.clip(yaw_target, -25.0 * np.pi / 180.0, 25.0 * np.pi / 180.0)
                    pitch_target = np.clip(pitch_target, -20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0)
                    
                    head_q_step = np.array([yaw_target, pitch_target])
                except Exception:
                    pass

            if arm_side == "left":
                move_status = self.movej(self.robot, left_arm=target_joint_pos, head=head_q_step, minimum_time=1.5)
            else:
                move_status = self.movej(self.robot, right_arm=target_joint_pos, head=head_q_step, minimum_time=1.5)

            if move_status:
                time.sleep(1.0) # Settling time
            else:
                if log_callback: log_callback(f"  [ERROR] Arm movement failed.")
                break

            if log_callback: log_callback(f"  - Capturing {arm_side} marker with LPF (2.0s)...")
            lpf_results = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            
            captured_pose = None
            if lpf_results and len(lpf_results) > 0:
                if status_callback: status_callback(True)
                if isinstance(lpf_results, list):
                    captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                elif isinstance(lpf_results, dict):
                    first_key = list(lpf_results.keys())[0]
                    captured_pose = np.array(lpf_results[first_key]).reshape(4, 4)
            else:
                if status_callback: status_callback(False)

            if captured_pose is not None:
                captured_pose[:3, 3] *= 1000.0 # m to mm
                captured_poses.append(captured_pose)
                captured_angles.append(target_offset_deg)
                if log_callback: log_callback(f"  - Pose Saved: Pos={np.round(captured_pose[:3, 3], 2)}")
            else:
                if log_callback: log_callback("  [ERROR] Marker lost during sweep. Aborting.")
                break 
        
        # Return to Initial Pose
        if log_callback: log_callback("\n[INFO] Sweep complete. Returning to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)

        # Calculation logic
        if len(captured_poses) >= max_points:
            T_cam_ref = captured_poses[0]
            T_ref_cam = np.linalg.inv(T_cam_ref)
            relative_poses = [T_ref_cam @ T for T in captured_poses]
            
            # Use nominal joint rotation axis as axis_prior to guarantee stability
            if axis_mode == 6:
                axis_prior = [0.0, -1.0, 0.0]
            else:
                axis_prior = [0.0, 0.0, 1.0]

            center, axis, radius, rmse, pts_2d, uc_opt, vc_opt, robust_tilt, robust_yaw, tilt_list = \
                self.fit_circle_3d_and_6dof_misalignment(
                    relative_poses, captured_angles, axis_prior=axis_prior, return_plot_data=True
                )
            
            result_dict = {
                'axis_mode': axis_mode,
                'radius': radius,
                'rmse': rmse,
                'axis': axis,
                'center': center,
                'pts_2d': pts_2d,
                'uc_opt': uc_opt,
                'vc_opt': vc_opt,
                'tilt': robust_tilt,
                'yaw': robust_yaw,
                'tilt_list': tilt_list
            }
            return result_dict
        
        return None
