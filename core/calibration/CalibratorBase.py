import sys
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
        self.robot_version = 1.2
        
        # Load camera setting config if available
        self.camera_config = {}
        self.load_camera_config()
        
        self.ready_poses = {}
        self.load_ready_poses()
        
        # Active joint home offsets to apply to commanded trajectories
        self.joint_offsets = {"wrist_pitch": 0.0, "elbow": 0.0}
        self.stop_requested = False

    def load_ready_poses(self):
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
        yaml_path = os.path.join(config_dir, "ready_poses.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    self.ready_poses = yaml.safe_load(f) or {}
                logging.info(f"Loaded ready poses from {yaml_path}")
            except Exception as e:
                logging.error(f"Failed to load ready_poses.yaml: {e}")
                sys.exit(f"[CRITICAL ERROR] Failed to parse ready_poses.yaml: {e}")
        else:
            logging.error(f"ready_poses.yaml not found at {yaml_path}!")
            sys.exit(f"[CRITICAL ERROR] ready_poses.yaml not found at {yaml_path}!")

    def get_robot_version(self):
        return getattr(self, "robot_version", 1.2)

    def get_ready_pose(self, version_key, type_key, mode_key, arm_side):
        # Fallback values
        fallbacks = {
            "v1.2": {
                "joint": {
                    "wrist_pitch": {
                        "right_arm": [-55.0, -45.0, 25.0, -127.0, 90.0, 0.0, 0.0],
                        "left_arm": [-55.0, 45.0, -25.0, -127.0, -90.0, 0.0, 0.0]
                    },
                    "elbow": {
                        "right_arm": [-107.0, -17.0, 0.0, 0.0, 73.0, -80.0, 73.0],
                        "left_arm": [-107.0, 17.0, 0.0, 0.0, -73.0, -80.0, -73.0]
                    }
                },
                "marker": {
                    "right_arm": [-90.0, -45.0, 73.0, -107.0, 90.0, 90.0, 0.0],
                    "left_arm": [-90.0, 45.0, -73.0, -107.0, -80.0, 90.0, 0.0]
                },
                "check_calib": {
                    "right_arm": [-90.0, -45.0, 73.0, -107.0, 90.0, 90.0, 0.0],
                    "left_arm": [-90.0, 45.0, -73.0, -107.0, -80.0, 90.0, 0.0]
                }
            },
            "v1.3": {
                "joint": {
                    "wrist_pitch": {
                        "right_arm": [-55.0, -45.0, 25.0, -127.0, 90.0, 0.0, 0.0],
                        "left_arm": [-55.0, 45.0, -25.0, -127.0, -90.0, 0.0, 0.0]
                    },
                    "elbow": {
                        "right_arm": [-107.0, -17.0, 0.0, 0.0, 73.0, -80.0, 73.0],
                        "left_arm": [-107.0, 17.0, 0.0, 0.0, -73.0, -80.0, -73.0]
                    }
                },
                "marker": {
                    "right_arm": [-90.0, -45.0, 73.0, -107.0, 0.0, 0.0, -80.0],
                    "left_arm": [-90.0, 45.0, -73.0, -107.0, 0.0, 0.0, 80.0]
                },
                "check_calib": {
                    "right_arm": [-90.0, -45.0, 73.0, -107.0, 0.0, 0.0, -80.0],
                    "left_arm": [-90.0, 45.0, -73.0, -107.0, 0.0, 0.0, 80.0]
                }
            }
        }
        
        try:
            val = self.ready_poses[version_key]
            if type_key == "joint":
                val = val["joint"][mode_key][f"{arm_side}_arm"]
            elif type_key == "check_calib":
                val = val["check_calib"][f"{arm_side}_arm"]
            else:
                val = val["marker"][f"{arm_side}_arm"]
            return np.deg2rad(val)
        except Exception:
            # Fallback
            val = fallbacks[version_key]
            if type_key == "joint":
                val = val["joint"][mode_key][f"{arm_side}_arm"]
            elif type_key == "check_calib":
                val = val["check_calib"][f"{arm_side}_arm"]
            else:
                val = val["marker"][f"{arm_side}_arm"]
            return np.deg2rad(val)

    def load_camera_config(self):
        # Locate setting.yaml
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
        yaml_path = os.path.join(config_dir, "setting.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    config_data = yaml.safe_load(f) or {}
                    self.camera_config = config_data.get("camera", {})
                logging.info(f"Loaded config from setting.yaml")
            except Exception as e:
                logging.error(f"Failed to load setting.yaml: {e}")
        else:
            logging.warning(f"setting.yaml not found at {yaml_path}")

    def save_debug_points(self, arm_side, mode, dataset_A, dataset_B, sweep_joint_A, sweep_joint_B, cand_joint, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback=None):
        try:
            config_dir = os.path.abspath(os.path.dirname(__file__))
            if not self.robot or self.robot == "mock_robot":
                arm_idx = [0]*20
                for i in range(20):
                    arm_idx[i] = i
            else:
                arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
            
            # Save dataset A (4축 또는 sweep_joint_A)
            filename_A = os.path.join(config_dir, f"sweep_points_{arm_side}_joint_A_axis_{sweep_joint_A}.txt")
            with open(filename_A, "w") as f:
                f.write("# Joint_A_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm), "
                        "T_cam2marker_flat(16), T_torso2marker_flat(16), T_ee2marker_flat(16)\n")
                for q_full, pose in dataset_A:
                    if not self.robot or self.robot == "mock_robot":
                        q_val = q_full[7 + sweep_joint_A] if arm_side == "left" else q_full[sweep_joint_A]
                    else:
                        q_val = q_full[arm_idx[sweep_joint_A]]
                    sa_deg = np.degrees(q_val - initial_joint_pos[sweep_joint_A])
                    p_cam = pose[:3, 3]
                    
                    T_cam_to_marker = pose
                    if not self.robot or self.robot == "mock_robot":
                        p_meas_t5 = p_cam
                        p_ee = p_cam
                        T_t5_to_marker = T_cam_to_marker
                        T_ee_to_marker = T_cam_to_marker
                    else:
                        T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                        T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                        p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                        p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                        T_t5_to_marker = T_t5_to_cam @ T_cam_to_marker
                        T_ee_to_marker = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker
                    
                    T_cam_flat_str = ", ".join(f"{v:.6f}" for v in T_cam_to_marker.flatten())
                    T_t5_flat_str = ", ".join(f"{v:.6f}" for v in T_t5_to_marker.flatten())
                    T_ee_flat_str = ", ".join(f"{v:.6f}" for v in T_ee_to_marker.flatten())
                    
                    f.write(f"{sa_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}, "
                            f"{T_cam_flat_str}, {T_t5_flat_str}, {T_ee_flat_str}\n")
            
            # Save dataset B (6축 또는 sweep_joint_B)
            filename_B = os.path.join(config_dir, f"sweep_points_{arm_side}_joint_B_axis_{sweep_joint_B}.txt")
            with open(filename_B, "w") as f:
                f.write("# Joint_B_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm), "
                        "T_cam2marker_flat(16), T_torso2marker_flat(16), T_ee2marker_flat(16)\n")
                for q_full, pose in dataset_B:
                    if not self.robot or self.robot == "mock_robot":
                        q_val = q_full[7 + sweep_joint_B] if arm_side == "left" else q_full[sweep_joint_B]
                    else:
                        q_val = q_full[arm_idx[sweep_joint_B]]
                    sb_deg = np.degrees(q_val - initial_joint_pos[sweep_joint_B])
                    p_cam = pose[:3, 3]
                    
                    T_cam_to_marker = pose
                    if not self.robot or self.robot == "mock_robot":
                        p_meas_t5 = p_cam
                        p_ee = p_cam
                        T_t5_to_marker = T_cam_to_marker
                        T_ee_to_marker = T_cam_to_marker
                    else:
                        T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                        T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                        p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                        p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                        T_t5_to_marker = T_t5_to_cam @ T_cam_to_marker
                        T_ee_to_marker = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker
                    
                    T_cam_flat_str = ", ".join(f"{v:.6f}" for v in T_cam_to_marker.flatten())
                    T_t5_flat_str = ", ".join(f"{v:.6f}" for v in T_t5_to_marker.flatten())
                    T_ee_flat_str = ", ".join(f"{v:.6f}" for v in T_ee_to_marker.flatten())
                    
                    f.write(f"{sb_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}, "
                            f"{T_cam_flat_str}, {T_t5_flat_str}, {T_ee_flat_str}\n")
            
            if log_callback:
                log_callback(f"[DEBUG] Saved Axis {sweep_joint_A} debug points to {os.path.basename(filename_A)}")
                log_callback(f"[DEBUG] Saved Axis {sweep_joint_B} debug points to {os.path.basename(filename_B)}")
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save debug points: {e}")

    def save_marker_debug_points(self, arm_side, axis_num, dataset, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, log_callback=None):
        try:
            config_dir = os.path.abspath(os.path.dirname(__file__))
            if not self.robot or self.robot == "mock_robot":
                arm_idx = [0]*20
                for i in range(20):
                    arm_idx[i] = i
            else:
                arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
            
            filename = os.path.join(config_dir, f"sweep_points_{arm_side}_marker_axis_{axis_num}.txt")
            with open(filename, "w") as f:
                f.write(f"# Joint_{axis_num}_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm), "
                        f"T_cam2marker_flat(16), T_torso2marker_flat(16), T_ee2marker_flat(16)\n")
                for q_full, pose in dataset:
                    if not self.robot or self.robot == "mock_robot":
                        q_val = q_full[7 + axis_num] if arm_side == "left" else q_full[axis_num]
                    else:
                        q_val = q_full[arm_idx[axis_num]]
                    s_deg = np.degrees(q_val - initial_joint_pos[axis_num])
                    p_cam = pose[:3, 3]
                    
                    T_cam_to_marker = pose
                    if not self.robot or self.robot == "mock_robot":
                        p_meas_t5 = p_cam
                        p_ee = p_cam
                        T_t5_to_marker = T_cam_to_marker
                        T_ee_to_marker = T_cam_to_marker
                    else:
                        T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                        T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                        p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                        p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                        T_t5_to_marker = T_t5_to_cam @ T_cam_to_marker
                        T_ee_to_marker = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker
                    
                    T_cam_flat_str = ", ".join(f"{v:.6f}" for v in T_cam_to_marker.flatten())
                    T_t5_flat_str = ", ".join(f"{v:.6f}" for v in T_t5_to_marker.flatten())
                    T_ee_flat_str = ", ".join(f"{v:.6f}" for v in T_ee_to_marker.flatten())
                    
                    f.write(f"{s_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}, "
                            f"{T_cam_flat_str}, {T_t5_flat_str}, {T_ee_flat_str}\n")
            if log_callback:
                log_callback(f"[DEBUG] Saved Axis {axis_num} marker sweep debug points to {os.path.basename(filename)}")
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save marker debug points: {e}")

    @staticmethod
    def initialize_robot(address, model, power=".*", servo=".*"):
        robot = rby.create_robot(address, model)
        if not robot.connect():
            logging.error(f"Failed to connect robot {address}")
            return None
        
        # Safety check: Verify actual connected robot model matches expected model
        try:
            robot_info = robot.get_robot_info()
            actual_model = robot_info.robot_model_name.lower()
            expected_model = model.lower()
            if actual_model != expected_model:
                logging.error(f"Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}")
                robot.disconnect()
                return None
        except Exception as e:
            logging.error(f"Failed to verify robot model: {e}")
            robot.disconnect()
            return None

        # Check if overall power is ON; if not, turn on overall power
        try:
            if not robot.is_power_on(".*"):
                logging.info("Power is not ON. Turning overall power on...")
                if not robot.power_on(".*"):
                    logging.error("Failed to turn power on.")
                    robot.disconnect()
                    return None
            else:
                logging.info("Power is already ON.")
        except Exception as e:
            logging.error(f"Failed to check or set power status: {e}")
            robot.disconnect()
            return None

        # Wait 1 second
        time.sleep(1.0)

        # Check control manager status
        try:
            cm_state = robot.get_control_manager_state().state
            is_cm_enabled = (cm_state == rby.ControlManagerState.State.Enabled)
        except Exception as e:
            logging.warning(f"Failed to check control manager state: {e}")
            is_cm_enabled = False

        # Check if both arms' servos are ON
        try:
            is_servo_ok = robot.is_servo_on(".*")
        except Exception as e:
            logging.warning(f"Failed to check servo status: {e}")
            is_servo_ok = False

        def enable_cm_helper(r):
            try:
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state in [
                    rby.ControlManagerState.State.MajorFault,
                    rby.ControlManagerState.State.MinorFault,
                ]:
                    logging.warning(f"Control manager is in fault state: {cm_state_post.state}. Resetting...")
                    if not r.reset_fault_control_manager():
                        logging.error("Failed to reset control manager")
                
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state == rby.ControlManagerState.State.Enabled:
                    logging.info("Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...")
                    try:
                        r.disable_control_manager()
                        time.sleep(0.5)
                    except Exception as ex:
                        logging.warning(f"Failed to disable control manager: {ex}")
                
                logging.info("Enabling control manager with unlimited_mode_enabled=True...")
                if not r.enable_control_manager(unlimited_mode_enabled=True):
                    logging.error("Failed to enable control manager with unlimited_mode_enabled=True")
            except Exception as ex:
                logging.error(f"Failed to configure control manager: {ex}")

        if is_servo_ok:
            # 맞으면, 컨트롤 매니저가 enable인지 확인하고 안돼있으면 enable
            if not is_cm_enabled:
                logging.info("Servos are ON but Control Manager is disabled. Enabling...")
                enable_cm_helper(robot)
            else:
                logging.info("Servos are ON and Control Manager is already enabled.")
        else:
            # 아니면, 일단 disable한다음에 양팔 서보 키고 enable
            logging.info("Servos are not ON. Disabling Control Manager first to turn on servos...")
            if is_cm_enabled:
                try:
                    robot.disable_control_manager()
                    time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Failed to disable control manager: {e}")
            
            logging.info("Turning servos on...")
            if not robot.servo_on(".*"):
                logging.error("Failed to turn servos on.")
            else:
                time.sleep(0.5)
            
            enable_cm_helper(robot)

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
        if getattr(self, 'stop_requested', False):
            return False
        if not robot:
            return False
            
        if apply_offsets and hasattr(self, 'joint_offsets') and self.joint_offsets is not None:
            # Offset mapping: Joint 3 (index 3) is elbow
            # For v1.3, Joint 6 (index 6) is wrist roll
            # For v1.2, Joint 5 (index 5) is wrist pitch
            is_v13 = abs(self.get_robot_version() - 1.3) < 0.05
            wrist_idx = 6 if is_v13 else 5
            if right_arm is not None:
                right_arm = list(right_arm)
                right_arm[wrist_idx] += np.radians(self.joint_offsets.get("wrist_pitch", 0.0))
                right_arm[3] += np.radians(self.joint_offsets.get("elbow", 0.0))
            if left_arm is not None:
                left_arm = list(left_arm)
                left_arm[wrist_idx] += np.radians(self.joint_offsets.get("wrist_pitch", 0.0))
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

    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False):
        points = np.array([T[:3, 3] * 1000.0 for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # Robust check for NaN or Inf in inputs
        if len(points) == 0:
            raise ValueError("fit_circle_3d_and_6dof_misalignment: Input points list is empty.")
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input points contain NaN or Inf values! points={points}")
        if np.any(np.isnan(angles_rad_base)) or np.any(np.isinf(angles_rad_base)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input captured_angles contain NaN or Inf values! angles={captured_angles}")
            
        # Initial Center and Normal estimation using unified circle fit
        c_fit, R_fit, radius_fit, rmse_fit, _, _, _ = BaseCalibrator.fit_circle_3d(points)
        
        # Check if initial circle fit yielded valid numbers
        if np.any(np.isnan(c_fit)) or np.any(np.isinf(c_fit)) or np.isnan(radius_fit):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Initial circle fit (fit_circle_3d) returned NaN or Inf values! c_fit={c_fit}, radius_fit={radius_fit}")
        
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
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [450.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
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
                
                cos_t = np.cos(angles_rad)[:, None]
                sin_t = np.sin(angles_rad)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (points - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 1 failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
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
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [450.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
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
                
                cos_t = np.cos(rad_in)[:, None]
                sin_t = np.sin(rad_in)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (pts_in - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals_in, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 2 (outlier loop) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
            c_init = opt_res.x[0:3]
            best_normal = opt_res.x[3:6]
            best_normal /= np.linalg.norm(best_normal)
            r_final_dir = opt_res.x[6:9]
            r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
            if np.linalg.norm(r_final_dir) > 1e-6:
                r_final_dir /= np.linalg.norm(r_final_dir)
            R_init = opt_res.x[9]
            
            cos_t = np.cos(angles_rad)[:, None]
            sin_t = np.sin(angles_rad)[:, None]
            cross_term = np.cross(best_normal, r_final_dir)
            dot_term = np.dot(best_normal, r_final_dir)
            
            pred_pts = c_init + R_init * (r_final_dir[None, :] * cos_t + 
                                         cross_term[None, :] * sin_t + 
                                         (best_normal * dot_term)[None, :] * (1.0 - cos_t))
            all_errors = np.linalg.norm(points - pred_pts, axis=1)
            
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
            
            cos_t = np.cos(rad_in)[:, None]
            sin_t = np.sin(rad_in)[:, None]
            cross_term = np.cross(axis, r_init)
            dot_term = np.dot(axis, r_init)
            
            pred_pts = c + R * (r_init[None, :] * cos_t + 
                               cross_term[None, :] * sin_t + 
                               (axis * dot_term)[None, :] * (1.0 - cos_t))
            return (pts_in - pred_pts).ravel()
            
        lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
        upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [450.0]])
        # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
        init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
        
        try:
            opt_res = least_squares(total_residuals_final, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
        except ValueError as e:
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 3 (final) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
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

    def perform_motion_test_sweep(self, arm_side, joint_i, start_deg, end_deg, log_callback=None):
        if getattr(self, 'stop_requested', False):
            return False
        if not self.robot or self.robot == "mock_robot":
            if log_callback: log_callback("[ERROR] Robot not connected or mock robot.")
            return False
            
        try:
            # 1. Get baseline joint position from current state
            state = self.robot.get_state()
            model = self.robot.model()
            arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
            initial_joint_pos = list(state.position[arm_idx])
            
            # 2. Prepare start and end poses
            q_start_arm = list(initial_joint_pos)
            q_start_arm[joint_i] += np.radians(start_deg)
            
            q_end_arm = list(initial_joint_pos)
            q_end_arm[joint_i] += np.radians(end_deg)
            
            # 3. Move to start position
            if log_callback: log_callback(f"[MOTION TEST] Moving {arm_side} arm joint {joint_i} to start angle ({start_deg} deg)...")
            if arm_side == "left":
                ok = self.movej(self.robot, left_arm=q_start_arm, minimum_time=3.0, apply_offsets=False)
            else:
                ok = self.movej(self.robot, right_arm=q_start_arm, minimum_time=3.0, apply_offsets=False)
            if not ok or getattr(self, 'stop_requested', False): return False
            time.sleep(0.5)
            
            # 4. Sweep to end position
            if log_callback: log_callback(f"[MOTION TEST] Sweeping {arm_side} arm joint {joint_i} to end angle ({end_deg} deg)...")
            if arm_side == "left":
                ok = self.movej(self.robot, left_arm=q_end_arm, minimum_time=5.0, apply_offsets=False)
            else:
                ok = self.movej(self.robot, right_arm=q_end_arm, minimum_time=5.0, apply_offsets=False)
            if not ok or getattr(self, 'stop_requested', False): return False
            time.sleep(0.5)
            
            # 5. Return to baseline ready pose
            if log_callback: log_callback(f"[MOTION TEST] Returning to ready pose...")
            if arm_side == "left":
                ok = self.movej(self.robot, left_arm=initial_joint_pos, minimum_time=3.0, apply_offsets=False)
            else:
                ok = self.movej(self.robot, right_arm=initial_joint_pos, minimum_time=3.0, apply_offsets=False)
            return ok
        except Exception as e:
            if log_callback: log_callback(f"[ERROR] Motion test sweep failed: {e}")
            return False

