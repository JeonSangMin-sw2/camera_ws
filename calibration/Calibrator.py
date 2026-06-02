import time
import logging
import os
import yaml
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as R_scipy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BaseCalibrator:
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot
        
        # Load camera setting config if available
        self.camera_config = {}
        self.load_camera_config()
        
        # Active joint home offsets to apply to commanded trajectories
        self.joint_offsets = {"wrist_pitch": 0.0, "elbow": 0.0}

    def load_camera_config(self):
        # Locate setting.yaml
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
        yaml_path = os.path.join(config_dir, "setting.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    self.camera_config = yaml.safe_load(f) or {}
                logging.info(f"Loaded config from setting.yaml")
            except Exception as e:
                logging.error(f"Failed to load setting.yaml: {e}")

    def save_debug_points(self, arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback=None):
        try:
            config_dir = os.path.abspath(os.path.dirname(__file__))
            arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
            
            # Save dataset A (4축 또는 sweep_joint_A)
            filename_A = os.path.join(config_dir, f"sweep_points_{arm_side}_joint_A_axis_{sweep_joint_A}.txt")
            with open(filename_A, "w") as f:
                f.write("# Joint_A_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm)\n")
                for q_full, pose in dataset_A:
                    sa_deg = np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A])
                    p_cam = pose[:3, 3]
                    
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    
                    f.write(f"{sa_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}\n")
            
            # Save dataset B (6축 또는 sweep_joint_B)
            filename_B = os.path.join(config_dir, f"sweep_points_{arm_side}_joint_B_axis_{sweep_joint_B}.txt")
            with open(filename_B, "w") as f:
                f.write("# Joint_B_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm)\n")
                for q_full, pose in dataset_B:
                    sb_deg = np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B])
                    p_cam = pose[:3, 3]
                    
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    
                    f.write(f"{sb_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}\n")
            
            if log_callback:
                log_callback(f"[DEBUG] Saved Axis {sweep_joint_A} debug points to {os.path.basename(filename_A)}")
                log_callback(f"[DEBUG] Saved Axis {sweep_joint_B} debug points to {os.path.basename(filename_B)}")
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save debug points: {e}")

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
        yaw = data[5]
        pitch = data[4]
        roll = data[3]
        T[:3, :3] = R_scipy.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        return T

    @staticmethod
    def rodrigues_rotation(vector, axis, theta_rad):
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return vector * cos_t + np.cross(axis, vector) * sin_t + axis * np.dot(axis, vector) * (1 - cos_t)

    def movej(self, robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0, apply_offsets=True, priority=10):
        if not robot:
            return False
            
        if apply_offsets and hasattr(self, 'joint_offsets'):
            # Offset mapping: Joint 3 (index 3) is elbow, Joint 5 (index 5) is wrist_pitch
            if right_arm is not None:
                right_arm = list(right_arm)
                right_arm[5] += np.radians(self.joint_offsets.get("wrist_pitch", 0.0))
                right_arm[3] += np.radians(self.joint_offsets.get("elbow", 0.0))
            if left_arm is not None:
                left_arm = list(left_arm)
                left_arm[5] += np.radians(self.joint_offsets.get("wrist_pitch", 0.0))
                left_arm[3] += np.radians(self.joint_offsets.get("elbow", 0.0))

        comp_cmd = rby.ComponentBasedCommandBuilder()
        
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
            comp_cmd.set_body_command(body_cmd)

        if head is not None:
            comp_cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )
        
        cmd = rby.RobotCommandBuilder().set_command(comp_cmd)
        
        try:
            rv = robot.send_command(cmd, priority).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                logging.error(f"Failed to conduct movej. Finish code: {rv.finish_code}")
                return False
            return True
        except Exception as e:
            logging.error(f"movej exception: {e}")
            return False

    @staticmethod
    def fit_circle_3d(points):
        """
        Fits a 3D circle to points with robust worst-inlier outlier rejection (up to 3 points).
        Returns (center_3d, R_circle, radius, rmse, pts_2d, uc, vc)
        where R_circle is the 3x3 rotation matrix in robot coordinate system: [ex, ey, normal]
        """
        points = np.array(points)
        inlier_mask = np.ones(len(points), dtype=bool)
        
        for out_iter in range(3):
            pts_in = points[inlier_mask]
            if len(pts_in) < 6:
                break
                
            centroid = np.mean(pts_in, axis=0)
            pts_centered = pts_in - centroid
            
            _, _, vh = np.linalg.svd(pts_centered)
            normal = vh[2, :]
            ex = vh[0, :]
            ey = vh[1, :]
            pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)

            A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
            b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
            
            def residuals(params):
                u, v, R = params
                return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
                
            opt = least_squares(residuals, [uc, vc, radius], loss='huber')
            uc_opt, vc_opt, R_opt = opt.x
            
            # Recompute errors for all original points
            all_errors = []
            for pt in points:
                pt_centered_all = pt - centroid
                u_all = np.dot(pt_centered_all, ex)
                v_all = np.dot(pt_centered_all, ey)
                dist_to_center = np.sqrt((u_all - uc_opt)**2 + (v_all - vc_opt)**2)
                err = abs(dist_to_center - R_opt)
                all_errors.append(err)
            all_errors = np.array(all_errors)
            
            # Find worst point among inliers
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.5:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final fit on clean inliers
        pts_in = points[inlier_mask]
        centroid = np.mean(pts_in, axis=0)
        pts_centered = pts_in - centroid
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        ex = vh[0, :]
        ey = vh[1, :]
        pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)
        A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
        b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        uc, vc = res[0], res[1]
        radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
        
        def residuals_final(params):
            u, v, R = params
            return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
            
        opt = least_squares(residuals_final, [uc, vc, radius], loss='huber')
        uc_opt, vc_opt, R_opt = opt.x
        rmse = np.sqrt(np.mean(opt.fun**2))
        center_3d = centroid + uc_opt * ex + vc_opt * ey
        
        # 2D representation for all points
        pts_centered_all = points - centroid
        pts_2d_all = np.dot(pts_centered_all, np.vstack((ex, ey)).T)
        
        R_circle = np.column_stack((ex, ey, normal))
        
        return center_3d, R_circle, R_opt, rmse, pts_2d_all, uc_opt, vc_opt



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
                left_arm = np.deg2rad([-90, 45, -73, 0, -90, 90, 0])
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
            
            # 정지 측정 중이므로 칼만 필터를 완전히 우회(use_filter=False)
            res = self.marker_st.get_marker_transform(sampling_time=0.5, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                
                # Capture full robot joint position
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
            
            # 정지 측정 중이므로 칼만 필터를 완전히 우회(use_filter=False)
            res = self.marker_st.get_marker_transform(sampling_time=0.5, side=arm_side, use_filter=False)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                
                # Capture full robot joint position
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

        # 3. OFFLINE NUMERICAL OPTIMIZATION (Brent's 1D search)
        # Load mount_to_cam (transform from head mount "link_head_2" to camera)
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        # Save captured sweep points to debug txt files before offline optimization
        self.save_debug_points(
            arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, 
            cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback
        )

        def evaluate_offset(offset_rad):
            pts_ee = []
            
            # Transform pts_A to ee frame
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                # Dynamically compute T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee.append(p_ee * 1000.0) # mm
                
            # Transform pts_B to ee frame
            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                # Dynamically compute T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee.append(p_ee * 1000.0) # mm
                
            pts_ee = np.array(pts_ee)
            mean_pt = np.mean(pts_ee, axis=0)
            deviations = np.linalg.norm(pts_ee - mean_pt, axis=1)
            # Minimize standard deviation of marker coordinates in ee frame (makes the two circles collapse together!)
            return np.sqrt(np.mean(deviations**2))

        iteration_count = 0
        def evaluate_offset_logged(offset_rad):
            nonlocal iteration_count
            iteration_count += 1
            error = evaluate_offset(offset_rad)
            if log_callback:
                log_callback(f"  [Iter {iteration_count:2d}] Joint Offset: {np.degrees(offset_rad):.6f}° | Marker Spread (SD): {error:.8f} mm")
            return error

        # Run brent scalar optimizer (bounds: ±10 deg = ±0.1745 rad)
        opt_res = minimize_scalar(
            evaluate_offset_logged, 
            bounds=(-np.radians(10.0), np.radians(10.0)), 
            method='bounded', 
            options={'xatol': 1e-8}
        )
        
        optimal_offset_rad = opt_res.x
        optimal_offset_deg = np.degrees(optimal_offset_rad)
        final_dist = opt_res.fun

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()})")
            log_callback("="*50)
            log_callback(f"  * Total Iterations          : {iteration_count}")
            log_callback(f"  * Estimated Optimal Offset  : {optimal_offset_deg:.6f} deg")
            log_callback(f"  * Final Marker Spread (SD)  : {final_dist:.8f} mm")
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

        # Generate candidates & errors array for the plot
        plot_offsets = np.linspace(-4.0, 4.0, 5)
        plot_errors = [evaluate_offset(np.radians(o)) for o in plot_offsets]

        # Simultaneous Marker Axis 6 parameter calculation
        marker_6_res = None
        if mode == "wrist_pitch":
            try:
                # Extract captured_poses projected into the static torso frame
                captured_poses_torso = []
                for q_full, pose_cam_to_marker in dataset_B:
                    # Dynamically compute T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    
                    # Project T_cam_to_marker back to torso frame T_t5_to_marker
                    T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                    captured_poses_torso.append(T_t5_to_marker)
                    
                # Solve Marker Bracket Axis 6 Calibration on torso-projected poses
                # Note: axis_prior = [1.0, 0.0, 0.0] for Axis 6 (roll)
                marker_6_res = self.fit_circle_3d_and_6dof_misalignment(
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
            'offsets': list(plot_offsets),
            'errors': plot_errors,
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
        
        # Capture marker pose and joint angles in real time
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
        
        # Capture marker pose and joint angles in real time
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

        # 3. OFFLINE NUMERICAL OPTIMIZATION (Brent's 1D search)
        if log_callback: log_callback("\n--- [3] Starting Offline Iterative Brent Optimization (Fitted Circle Center/Normal Orthogonality - Dense) ---")
        
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_mount_to_cam = self.make_transform(mount_to_cam)

        def evaluate_offset(offset_rad):
            pts_ee_A = []
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee_A.append(p_ee * 1000.0)
                
            pts_ee_B = []
            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee_B.append(p_ee * 1000.0)
                
            # Perform robust SVD 3D circle fitting on both projected datasets
            c_A, R_circle_A, r_A, rmse_A, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_A)
            c_B, R_circle_B, r_B, rmse_B, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_B)
            
            n_A = R_circle_A[:, 2] # Circle A Normal vector (sweep rotation axis A)
            n_B = R_circle_B[:, 2] # Circle B Normal vector (sweep rotation axis B)
            
            # Center offset error: spatial distance in mm between two fitted centers
            center_dist = np.linalg.norm(c_A - c_B)
            
            # Orthogonality error: dot product of orthogonal axes should be exactly zero
            dot_prod = np.dot(n_A, n_B)
            angle_error_deg = np.degrees(np.arcsin(np.clip(abs(dot_prod), -1.0, 1.0)))
            
            # Weighted joint calibration metric (1 degree axis alignment error ~ 2 mm center shift weight)
            return center_dist + angle_error_deg * 2.0

        iteration_count = 0
        def evaluate_offset_logged(offset_rad):
            nonlocal iteration_count
            iteration_count += 1
            error = evaluate_offset(offset_rad)
            if log_callback:
                log_callback(f"  [Iter {iteration_count:2d}] Joint Offset: {np.degrees(offset_rad):.6f}° | Circle Fit Error: {error:.8f}")
            return error

        # Run Brent scalar optimizer (extended bounds: ±20 deg = ±0.349 rad)
        opt_res = minimize_scalar(
            evaluate_offset_logged, 
            bounds=(-np.radians(20.0), np.radians(20.0)), 
            method='bounded', 
            options={'xatol': 1e-8}
        )
        
        optimal_offset_rad = opt_res.x
        optimal_offset_deg = np.degrees(optimal_offset_rad)
        final_dist = opt_res.fun

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()} - CONTINUOUS)")
            log_callback("="*50)
            log_callback(f"  * Total Iterations          : {iteration_count}")
            log_callback(f"  * Estimated Optimal Offset  : {optimal_offset_deg:.6f} deg")
            log_callback(f"  * Final Circle Fit Error    : {final_dist:.8f}")
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

        # Simultaneous Marker Axis 6 parameter calculation
        marker_6_res = None
        if mode == "wrist_pitch":
            try:
                captured_poses_torso = []
                sweep_angles_dummy = np.linspace(-20.0, 20.0, len(dataset_B))
                for q_full, pose_cam_to_marker in dataset_B:
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                    captured_poses_torso.append(T_t5_to_marker)
                    
                marker_6_res = self.fit_circle_3d_and_6dof_misalignment(
                    captured_poses_torso, 
                    sweep_angles_dummy, 
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
