import time
import logging
import os
import yaml
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
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
        
        # Check if control manager is already enabled and power/servo are already ON
        try:
            import time
            cm_state = robot.get_control_manager_state().state
            is_enabled = (cm_state == rby.ControlManagerState.State.Enabled)
        except Exception as e:
            logging.warning(f"Failed to check control manager state: {e}")
            is_enabled = False

        need_power = not robot.is_power_on(power)
        need_servo = not robot.is_servo_on(servo)

        if need_power or need_servo:
            # 1) Disable control manager first if it is Enabled to allow power/servo commands
            if is_enabled:
                logging.info("Control manager is Enabled. Disabling to allow power/servo commands...")
                try:
                    robot.disable_control_manager()
                    time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Failed to disable control manager: {e}")

            # 2) Perform power/servo commands
            if need_power:
                logging.info(f"Turning power ({power}) on...")
                if not robot.power_on(power):
                    logging.error(f"Failed to turn power ({power}) on. Continuing...")
            else:
                logging.info(f"Power ({power}) is already ON.")
            
            if need_servo:
                logging.info(f"Turning servo ({servo}) on...")
                if not robot.servo_on(servo):
                    logging.error(f"Failed to servo ({servo}) on. Continuing...")
            else:
                logging.info(f"Servo ({servo}) is already ON.")
        else:
            logging.info("Power and Servos are already ON and Control Manager is Enabled. Skipping redundant power/servo commands.")

        # 3) Reset fault and enable control manager
        try:
            cm_state_post = robot.get_control_manager_state()
            if cm_state_post.state in [
                rby.ControlManagerState.State.MajorFault,
                rby.ControlManagerState.State.MinorFault,
            ]:
                logging.warning(f"Control manager is in fault state: {cm_state_post.state}. Resetting...")
                if not robot.reset_fault_control_manager():
                    logging.error(f"Failed to reset control manager")
            
            cm_state_post = robot.get_control_manager_state()
            if cm_state_post.state != rby.ControlManagerState.State.Enabled:
                logging.info("Enabling control manager...")
                if not robot.enable_control_manager():
                    logging.error(f"Failed to enable control manager")
            else:
                logging.info("Control manager is already enabled.")
        except Exception as e:
            logging.error(f"Failed to configure control manager: {e}. Continuing...")
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
        Fits a 3D circle to points with robust worst-inlier outlier rejection (up to 15 points).
        Returns (center_3d, R_circle, radius, rmse, pts_2d, uc, vc)
        where R_circle is the 3x3 rotation matrix in robot coordinate system: [ex, ey, normal]
        """
        points = np.array(points)
        
        # Apply 3D Moving Median Filter (window size 5) to smooth out camera sensor jitter
        if len(points) >= 5:
            smoothed = np.copy(points)
            for i in range(2, len(points) - 2):
                smoothed[i] = np.median(points[i - 2 : i + 3], axis=0)
            points = smoothed
            
        inlier_mask = np.ones(len(points), dtype=bool)
        
        for out_iter in range(15):
            pts_in = points[inlier_mask]
            if len(pts_in) < 10:
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
            
            if worst_error > 0.1:
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
