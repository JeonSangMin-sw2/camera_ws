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
        if head is not None:
            body_cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )
        
        cmd = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
        )
        
        try:
            robot.send_command(cmd).get()
            return True
        except Exception as e:
            logging.error(f"movej failed: {e}")
            return False

    @staticmethod
    def fit_circle_3d(points):
        """
        Fits a 3D circle to points with robust worst-inlier outlier rejection (up to 3 points).
        Returns (center_3d, normal, radius, rmse, pts_2d, uc, vc, ex, ey)
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
        
        return center_3d, normal, R_opt, rmse, pts_2d_all, uc_opt, vc_opt, ex, ey


class MarkerCalibrator(BaseCalibrator):
    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False):
        points = np.array([T[:3, 3] for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # Initial Normal
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
        if axis_prior is not None:
            n_nominal = np.array(axis_prior, dtype=float)
            n_nominal /= np.linalg.norm(n_nominal)
        else:
            n_nominal = np.array([0.0, 0.0, 1.0])
            
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
            'inlier_mask': inlier_mask
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
        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {axis_mode.upper()} CALIBRATION SWEEP")
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
        if "axis 6" in axis_mode.lower():
            start_deg = -20.0
            step_deg = 5.0
            max_points = 9
            joint_i = 6
        else:
            start_deg = -10.0
            step_deg = 2.5
            max_points = 9
            joint_i = 5

        # Active head tracking setup
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

        if log_callback: log_callback(f"[INFO] Initial arm pose: {np.round(initial_joint_pos, 2)}")
        
        captured_poses = []
        captured_angles = []

        for i in range(max_points):
            if log_callback: log_callback(f"\n[STEP {i + 1}/{max_points}]")
            
            target_offset_deg = start_deg + (i * step_deg)
            target_joint_pos = list(initial_joint_pos)
            target_joint_pos[joint_i] = initial_joint_pos[joint_i] + np.radians(target_offset_deg)
            
            if target_offset_deg == 0:
                if log_callback: log_callback(f"  - Moving to Center Pose (0 deg)...")
            else:
                if log_callback: log_callback(f"  - Moving to {target_offset_deg:.1f} deg offset...")
            
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
                time.sleep(1.0)
            else:
                if log_callback: log_callback(f"  [ERROR] Arm movement failed.")
                break

            if log_callback: log_callback(f"  - Capturing marker transform...")
            lpf_results = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            
            captured_pose = None
            if lpf_results and len(lpf_results) > 0:
                if status_callback: status_callback(True)
                if isinstance(lpf_results, list):
                    captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                else:
                    captured_pose = np.array(list(lpf_results.values())[0]).reshape(4, 4)
            else:
                if status_callback: status_callback(False)

            if captured_pose is not None:
                captured_poses.append(captured_pose)
                captured_angles.append(target_offset_deg)
                if log_callback: log_callback(f"  - Pose Saved: Pos={np.round(captured_pose[:3, 3] * 1000.0, 2)} mm")
            else:
                if log_callback: log_callback("  [ERROR] Marker tracking failed at this pose.")

        # Return to ready pose
        if log_callback: log_callback("\n[INFO] Returning arm and head to ready position...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)

        if len(captured_poses) < 5:
            if log_callback: log_callback("[ERROR] Too few valid marker poses. Calibration failed.")
            return None

        # Solve Circle Fitting
        n_nom = [1.0, 0.0, 0.0] if "axis 6" in axis_mode.lower() else [0.0, 1.0, 0.0]
        res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom)
        return res


class PitchHeadCalibrator(BaseCalibrator):
    def perform_move_to_ready_pose(self, arm_side, mode, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving to {arm_side} Pitch/Head Ready Pose (Mode: {mode})...")
        torso = [0, 0, 0, 0, 0, 0]
        
        if mode == "wrist_pitch":
            if arm_side == "right":
                right_arm = np.deg2rad([-60, -45, 30, -107, 90, 0, 0])
                left_arm = [0, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [0, 0, 0, 0, 0, 0, 0]
                left_arm = np.deg2rad([-60, 45, -30, -107, -90, 0, 0])
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

    def perform_calibration_sweep_5_or_3(self, arm_side, mode, log_callback=None, status_callback=None, use_head_tracking=True):
        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} OFFSET CALIBRATION SWEEP (ITERATIVE BRENT OPTIMIZATION)")
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

        # 9 steps from -20 to 20 deg (5 deg steps)
        sweep_angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        
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

        # Arm cand baseline pose (joint 5 is 0)
        q_cand = list(initial_joint_pos)

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
                self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=1.2)
            else:
                self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=1.2)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_cam = pose[:3, 3] # meters
                
                # Capture full robot joint position
                q_full_captured = np.array(self.robot.get_state().position)
                dataset_A.append((q_full_captured, p_cam))
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
            self.movej(self.robot, left_arm=q_cand, head=head_q_step_cand, minimum_time=1.5)
        else:
            self.movej(self.robot, right_arm=q_cand, head=head_q_step_cand, minimum_time=1.5)
        time.sleep(0.5)

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
                self.movej(self.robot, left_arm=q_sweep, head=head_q_step, minimum_time=1.2)
            else:
                self.movej(self.robot, right_arm=q_sweep, head=head_q_step, minimum_time=1.2)
            time.sleep(0.5)
            
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if res:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                p_cam = pose[:3, 3]
                
                # Capture full robot joint position
                q_full_captured = np.array(self.robot.get_state().position)
                dataset_B.append((q_full_captured, p_cam))
                if log_callback: log_callback(f"  Captured Point {len(dataset_B)}/{len(sweep_angles)}: sb={sb:.1f}°")

        # Return arm and head to original ready pose
        if log_callback: log_callback("\n[INFO] Sweep finished. Returning arm and head to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, head=q_head_0, minimum_time=2.0)

        if len(dataset_A) < 6 or len(dataset_B) < 6:
            if log_callback: log_callback("[ERROR] Too few valid captured points. Calibration failed.")
            return None

        # 3. OFFLINE NUMERICAL OPTIMIZATION (Brent's 1D search)
        if log_callback: log_callback("\n--- [3] Starting Offline Iterative Brent Optimization ---")
        
        # Load mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_t5_to_cam = self.make_transform(mount_to_cam)

        def evaluate_offset(offset_rad):
            # Transform pts_A to ee frame
            pts_ee_A = []
            for q_full, p_cam in dataset_A:
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee_A.append(p_ee * 1000.0) # mm
                
            # Transform pts_B to ee frame
            pts_ee_B = []
            for q_full, p_cam in dataset_B:
                q_mod = np.array(q_full)
                q_mod[arm_idx[cand_joint]] += offset_rad
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                
                p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                R = T_t5_to_ee[:3, :3]
                p = T_t5_to_ee[:3, 3]
                p_ee = R.T @ (p_meas_t5 - p)
                pts_ee_B.append(p_ee * 1000.0) # mm
                
            c3d_A, _, _, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_A)
            c3d_B, _, _, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_B)
            return np.linalg.norm(c3d_A - c3d_B)

        iteration_count = 0
        def evaluate_offset_logged(offset_rad):
            nonlocal iteration_count
            iteration_count += 1
            error = evaluate_offset(offset_rad)
            if log_callback:
                log_callback(f"  [Iter {iteration_count:2d}] Joint Offset: {np.degrees(offset_rad):.6f}° | Circle Centers Dist: {error:.8f} mm")
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
            log_callback(f"  * Final Circle Centers Dist : {final_dist:.8f} mm")
            log_callback("="*50)

        # Prepare final visual projection datasets
        pts_ee_A_best = []
        for q_full, p_cam in dataset_A:
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_A_best.append(p_ee * 1000.0)
            
        pts_ee_B_best = []
        for q_full, p_cam in dataset_B:
            q_mod = np.array(q_full)
            q_mod[arm_idx[cand_joint]] += optimal_offset_rad
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
            p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
            p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
            pts_ee_B_best.append(p_ee * 1000.0)

        # High-precision 3D fit circles for plotting
        c3d_A, _, r_A, rmse_A, pts_2d_A, uc_A, vc_A, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_A_best)
        c3d_B, _, r_B, rmse_B, pts_2d_B, uc_B, vc_B, _, _ = BaseCalibrator.fit_circle_3d(pts_ee_B_best)

        # Generate candidates & errors array for the plot
        plot_offsets = np.linspace(-4.0, 4.0, 5)
        plot_errors = [evaluate_offset(np.radians(o)) for o in plot_offsets]

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
