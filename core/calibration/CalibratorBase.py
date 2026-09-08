import sys
import time
import logging
import os
import yaml
import numpy as np
import rby1_sdk as rby
import threading
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BaseCalibrator:
    # Shared by Full Auto and the individual joint-calibration UI.
    JOINT_SWEEP_SECONDS = {
        'wrist_yaw2': 20.0, 'wrist_roll_v13': 20.0,
        'wrist_pitch': 15.0, 'wrist_pitch_v13': 15.0, 'elbow': 20.0,
    }
    JOINT_CONFIGS = {
        "wrist_roll_v13":  {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_roll",  "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 15.0},
        "wrist_pitch_v13": {"cand_joint": 5, "sweep_joint_A": 6, "sweep_joint_B": 4, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 15.0, "sweep_range_B": 15.0},
        "wrist_yaw2":      {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_yaw2",  "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 15.0},
        "wrist_pitch":     {"cand_joint": 5, "sweep_joint_A": 4, "sweep_joint_B": 6, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 15.0, "sweep_range_B": 15.0},
        "elbow":           {"cand_joint": 3, "sweep_joint_A": 2, "sweep_joint_B": 4, "offset_key": "elbow",       "offset_range": (-5.0, 0.0),   "sweep_range_A": 15.0, "sweep_range_B": 15.0},
    }
    MARKER_CONFIGS = {
        "axis_4": {"joint_i": 4, "start_deg": -15.0, "end_deg": 15.0, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [0.0, 0.0, 1.0]},
        "axis_5": {"joint_i": 5, "start_deg": 0.0, "end_deg": -30.0, "n_nom_v12": [0.0, 1.0, 0.0], "n_nom_v13": [0.0, 1.0, 0.0]},
        "axis_6": {"joint_i": 6, "start_deg": -15.0, "end_deg": 15.0, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [1.0, 0.0, 0.0]},
    }
    NOMINAL_BRACKET_TEMPLATES = {
        "1.3": {
            "left":  [0.067, 0.0, 0.0, 90.0, 0.0, -90.0],
            "right": [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
        },
        "1.2": {
            "left":  [0.0, 0.054, -0.048, 90.0, 0.0, 0.0],
            "right": [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
        }
    }
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot
        self.robot_version = "1.2"
        
        # Load camera setting config if available
        self.camera_config = {}
        self.markers_config = {}
        self.load_camera_config()
        
        self.ready_poses = {}
        self.load_ready_poses()
        
        # Active joint home offsets to apply to commanded trajectories
        self.joint_offsets = {
            "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
            "left":  {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
        }
        self.app = None
        self.include_head_motion = True
        self.user_taught_ready_poses = {}
        self.stop_requested = False

    def clear_user_taught_ready_poses(self, arm_side=None, mode=None):
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            if arm_side and mode:
                norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].pop(norm_mode, None)
            elif arm_side:
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].clear()
            else:
                self.user_taught_ready_poses.clear()

    def load_ready_poses(self):
        from core.paths import CONFIG_PATHS
        yaml_path = CONFIG_PATHS["ready_poses_yaml"]
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    self.ready_poses = yaml.safe_load(f) or {}
                logging.info(f"Loaded ready poses from {yaml_path}")
            except Exception as e:
                logging.error(f"Failed to load ready_poses.yaml: {e}")
                sys.exit(f"[CRITICAL ERROR] Failed to parse ready_poses.yaml: {e}")
        else:
            logging.error(f"ready_poses.yaml not found at {yaml_path}!")
            sys.exit(f"[CRITICAL ERROR] ready_poses.yaml not found at {yaml_path}!")

    def get_robot_version(self) -> str:
        """Returns the robot version as a string: '1.0', '1.1', '1.2', or '1.3'."""
        return str(getattr(self, "robot_version", "1.2"))

    def is_v13(self) -> bool:
        """Returns True only for model-m v1.3 robots."""
        return self.get_robot_version() == "1.3"

    def is_head_active(self) -> bool:
        if getattr(self, 'include_head_motion', None) is False:
            return False
        if getattr(self, 'head_enabled', None) is False:
            return False
        if hasattr(self, 'app') and self.app is not None:
            if getattr(self.app, 'include_head_motion', None) is False:
                return False
            if hasattr(self.app, 'chk_servo_head') and not self.app.chk_servo_head.isChecked():
                return False
        model = getattr(self.robot, 'model', lambda: None)() if hasattr(self, 'robot') and self.robot else None
        if model is not None:
            return hasattr(model, 'head_idx') and len(getattr(model, 'head_idx', [])) >= 2
        return getattr(self, 'include_head_motion', True)

    def uses_head_camera(self):
        from core.marker_detection import uses_head_camera
        return uses_head_camera(self.camera_config, self.robot.model())

    def get_ready_pose(self, version_key, type_key, mode_key, arm_side):
        if not self.ready_poses:
            raise RuntimeError("Ready poses are not loaded or the configuration file is empty.")
        
        try:
            ver_clean = str(version_key).replace("v", "")
            val = self.ready_poses.get(version_key) or self.ready_poses.get(f"v{ver_clean}") or self.ready_poses.get(ver_clean)
            if val is None:
                raise KeyError(f"Version key {version_key} not in ready_poses (available: {list(self.ready_poses.keys())})")
            if type_key == "joint":
                lookup_key = mode_key
                if lookup_key == "wrist_pitch_v13":
                    lookup_key = "wrist_pitch"
                elif lookup_key == "wrist_roll_v13":
                    lookup_key = "wrist_roll"
                val = val["joint"][lookup_key][f"{arm_side}_arm"]
            elif type_key == "check_calib":
                val = val["check_calib"][f"{arm_side}_arm"]
            else:
                val = val["marker"][f"{arm_side}_arm"]
            val_arr = np.array(val, dtype=np.float64).copy()
            # Without head motion, lower Shoulder Pitch (Joint 0) for a fixed viewing direction.
            # and adjust Elbow (Joint 3) in negative direction to keep marker perpendicular to camera FOV without tilting backward.
            if not self.is_head_active() and len(val_arr) >= 7:
                j0_delta = 19.0
                elbow_delta = -4.0 if (type_key == "marker" or mode_key in ["wrist_pitch", "wrist_yaw2", "wrist_roll", "wrist_pitch_v13", "wrist_roll_v13"]) else 0.0
                val_arr[0] += j0_delta  # Joint 0 positive pitch lowers the arm down into fixed FOV (-55 -> -36 deg)
                val_arr[3] += elbow_delta  # Joint 3 negative offset flexes elbow to prevent marker from tilting backwards
                msg = f"[READY POSE] Head motion disabled (camera mount unchanged): Joint 0 lowered by +{j0_delta:.1f}° ({val_arr[0]:.1f}°), Elbow adjusted by {elbow_delta:+.1f}° ({val_arr[3]:.1f}°) for {arm_side}_arm ({type_key}/{mode_key})"
                print(msg)
                logging.info(msg)

            return np.deg2rad(val_arr)
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"[ERROR] Failed to get ready pose for version='{version_key}', type='{type_key}', mode='{mode_key}', arm='{arm_side}_arm'. "
                f"Please check your ready_poses.yaml file. Details: {e}"
            )


    def load_camera_config(self):
        # Locate setting.yaml
        from core.paths import CONFIG_PATHS
        yaml_path = CONFIG_PATHS["setting_yaml"]
        if not os.path.exists(yaml_path):
            logging.error(f"[CRITICAL ERROR] setting.yaml not found at {yaml_path}!")
            raise FileNotFoundError(f"[CRITICAL ERROR] setting.yaml not found at {yaml_path}!")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            self.camera_config = config_data.get("camera", {})
            self.markers_config = config_data.get("marker", {}) or {}
            
            # Legacy compatibility: if marker keys are under camera section, copy to markers_config
            legacy_keys = [
                "Tf_to_marker_left", "Tf_to_marker_right",
                "Tf_to_marker_left_v12", "Tf_to_marker_right_v12",
                "Tf_to_marker_left_v13", "Tf_to_marker_right_v13"
            ]
            for k in legacy_keys:
                if k not in self.markers_config and k in self.camera_config:
                    self.markers_config[k] = self.camera_config[k]

            # Strict validation for marker configurations in setting.yaml
            required_keys = ["Tf_to_marker_left_v13", "Tf_to_marker_right_v13", "Tf_to_marker_left_v12", "Tf_to_marker_right_v12"]
            missing_keys = [k for k in required_keys if k not in self.markers_config]
            if missing_keys:
                raise KeyError(f"[CRITICAL ERROR] Missing required marker configuration keys in setting.yaml: {missing_keys}")

            # For compatibility, merge Tf_to_marker keys from markers_config to camera_config
            for k, v in self.markers_config.items():
                if k.startswith("Tf_to_marker_"):
                    self.camera_config[k] = v
            
            # Dynamically synchronize NOMINAL_BRACKET_TEMPLATES with values from setting.yaml
            self.NOMINAL_BRACKET_TEMPLATES["1.3"]["left"] = list(self.markers_config["Tf_to_marker_left_v13"])
            self.NOMINAL_BRACKET_TEMPLATES["1.3"]["right"] = list(self.markers_config["Tf_to_marker_right_v13"])
            self.NOMINAL_BRACKET_TEMPLATES["1.2"]["left"] = list(self.markers_config["Tf_to_marker_left_v12"])
            self.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"] = list(self.markers_config["Tf_to_marker_right_v12"])
            
            logging.info(f"Loaded config from setting.yaml successfully.")
        except Exception as e:
            logging.error(f"[CRITICAL ERROR] Failed to load setting.yaml: {e}")
            raise RuntimeError(f"[CRITICAL ERROR] Failed to load setting.yaml: {e}")

    def save_observed_points(self, arm_side, axis_num, poses, label):
        """Camera poses only. No inferred physical pose or encoder columns."""
        from core.paths import CONFIG_PATHS
        directory = CONFIG_PATHS['txt_dir']
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'sweep_points_{arm_side}_{label}_axis_{axis_num}.txt')
        with open(path, 'a', encoding='utf-8') as stream:
            np.savetxt(stream, np.asarray(poses).reshape(-1, 16),
                       header='ordered T_camera_marker; translation metres; no encoder/FK')

    @staticmethod
    def initialize_robot(address, model, servo=None, include_head=True):
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
                logging.warning(f"Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}. Auto-reconnecting with actual model...")
                robot.disconnect()
                robot = rby.create_robot(address, robot_info.robot_model_name)
                if not robot.connect():
                    logging.error(f"Failed to connect robot {address} with actual model {robot_info.robot_model_name}")
                    return None
        except Exception as e:
            logging.error(f"Failed to verify robot model: {e}")
            robot.disconnect()
            return None

        # Check if connecting to localhost/simulator
        endpoint_host = str(address).rsplit(":", 1)[0].strip("[]").lower()
        is_local = endpoint_host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")

        # Check if power is ON; if not, turn on power
        try:
            power_pattern = ".*" if is_local else "48v"
            if not robot.is_power_on(power_pattern):
                logging.info(f"Power ({power_pattern}) is not ON. Turning power on...")
                if not robot.power_on(power_pattern):
                    logging.error(f"Failed to turn power ({power_pattern}) on.")
                    robot.disconnect()
                    return None
                time.sleep(1.0)
            else:
                logging.info(f"Power ({power_pattern}) is already ON.")
        except Exception as e:
            logging.error(f"Failed to check or set power status: {e}")
            robot.disconnect()
            return None

        # Wait 1 second
        time.sleep(1.0)

        # Check and reset control manager fault if necessary
        try:
            cm_state = robot.get_control_manager_state().state
            if cm_state in [
                rby.ControlManagerState.State.MajorFault,
                rby.ControlManagerState.State.MinorFault,
            ]:
                logging.warning("Control manager is in fault state. Resetting...")
                robot.reset_fault_control_manager()
                time.sleep(0.5)
            cm_state = robot.get_control_manager_state().state
            is_cm_enabled = (cm_state == rby.ControlManagerState.State.Enabled)
        except Exception as e:
            logging.warning(f"Failed to check control manager state: {e}")
            is_cm_enabled = False

        # Configure servo pattern based on include_head flag (independent of physical hardware)
        if servo is not None and servo != ".*":
            target_servo_pattern = servo
        else:
            target_servo_pattern = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
        if not include_head:
            # An explicit caller pattern must not override the no-head rule.
            target_servo_pattern = f"^(?!.*head)(?:{target_servo_pattern})$"

        # Check if servos are ON
        try:
            is_servo_ok = robot.is_servo_on(target_servo_pattern)
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
                else:
                    time.sleep(1.0)
            except Exception as ex:
                logging.error(f"Failed to configure control manager: {ex}")

        if is_servo_ok:
            logging.info("Servos are ON. Ensuring Control Manager is enabled with unlimited mode...")
            enable_cm_helper(robot)
        else:
            # Otherwise, disable control manager first, then turn on servos and enable
            logging.info("Servos are not ON. Disabling Control Manager first to turn on servos...")
            if is_cm_enabled:
                try:
                    robot.disable_control_manager()
                    time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Failed to disable control manager: {e}")
            
            logging.info(f"Turning servos on with pattern '{target_servo_pattern}'...")
            if not robot.servo_on(target_servo_pattern):
                logging.error(f"Failed to turn servos on with pattern '{target_servo_pattern}'.")
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
        num_joints = len(model.robot_joint_names)
        q_arr = np.zeros(num_joints)
        if len(q) >= num_joints:
            q_arr = np.array(q[:num_joints])
        else:
            q_arr[:len(q)] = q
        state.set_q(q_arr)
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



    def movej(self, robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0, apply_offsets=True, priority=10):
        if getattr(self, 'stop_requested', False):
            return False
        if not robot:
            return False
        if not self.is_head_active():
            head = None
            
        if head is not None:
            model = robot.model()
            has_head = hasattr(model, 'head_idx') and len(model.head_idx) > 0
            if not has_head:
                head = None

        if apply_offsets and hasattr(self, 'joint_offsets') and self.joint_offsets is not None:
            # Offset mapping: Joint 3 (index 3) is elbow
            # For v1.3:
            # - Joint 5 (index 5) is wrist pitch
            # - Joint 6 (index 6) is wrist roll
            # For v1.2:
            # - Joint 5 (index 5) is wrist pitch
            is_v13 = self.is_v13()
            
            # Support both flat and nested left/right dictionary structures
            if "left" in self.joint_offsets and "right" in self.joint_offsets:
                left_offsets = self.joint_offsets["left"]
                right_offsets = self.joint_offsets["right"]
            else:
                left_offsets = self.joint_offsets
                right_offsets = self.joint_offsets
                
            if right_arm is not None:
                right_arm = list(right_arm)
                r_j6_offset = right_offsets.get("wrist_roll", 0.0) if is_v13 else right_offsets.get("wrist_yaw2", 0.0)
                right_arm[6] += np.radians(r_j6_offset)
                right_arm[5] += np.radians(right_offsets.get("wrist_pitch", 0.0))
                right_arm[3] += np.radians(right_offsets.get("elbow", 0.0))
            if left_arm is not None:
                left_arm = list(left_arm)
                l_j6_offset = left_offsets.get("wrist_roll", 0.0) if is_v13 else left_offsets.get("wrist_yaw2", 0.0)
                left_arm[6] += np.radians(l_j6_offset)
                left_arm[5] += np.radians(left_offsets.get("wrist_pitch", 0.0))
                left_arm[3] += np.radians(left_offsets.get("elbow", 0.0))

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
        elif head is None:
            # A disabled head-only request must not send an empty robot command.
            return False

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
                print(f"[DEBUG MOVEJ ERROR] Failed to conduct movej. Finish code: {rv.finish_code}", flush=True)
                logging.error(f"Failed to conduct movej. Finish code: {rv.finish_code}")
                return False
            return True
        except Exception as e:
            print(f"[DEBUG MOVEJ EXCEPTION] movej exception: {e}", flush=True)
            logging.error(f"movej exception: {e}")
            return False

    @staticmethod
    def fit_observed_circle(poses, sweep_direction=1):
        """Fit ordered camera observations (metres), without angles or FK.

        The ordered cross products fix the plane-normal ambiguity; the sign
        of the commanded sweep maps that normal to the positive joint axis.
        """
        poses = np.asarray(poses, dtype=float)
        if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 10
                or not np.all(np.isfinite(poses)) or sweep_direction not in (-1, 1)):
            raise ValueError('At least ten finite ordered marker poses and a sweep direction are required')
        points = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        if (not np.allclose(poses[:, 3], [0., 0., 0., 1.], atol=1e-6)
                or np.max(np.abs(rotations.transpose(0,2,1) @ rotations - np.eye(3))) > 1e-3
                or np.max(np.abs(np.linalg.det(rotations)-1.)) > 1e-3):
            raise ValueError('Marker observations must be proper SE(3) transforms')
        origin = points.mean(axis=0)
        _, singular, vh = np.linalg.svd(points - origin, full_matrices=False)
        if singular[1] < 1e-5 or singular[1] < singular[0] * 1e-3:
            raise ValueError('Stationary or collinear marker trace; circle is unobservable')
        basis, normal = vh[:2].T, vh[2]
        xy = (points - origin) @ basis
        algebra, _, _, _ = np.linalg.lstsq(
            np.c_[2 * xy, np.ones(len(xy))], np.sum(xy**2, axis=1), rcond=None)
        initial_radius = np.sqrt(max(0., algebra[2] + np.dot(algebra[:2], algebra[:2])))
        fit = least_squares(lambda x: np.linalg.norm(xy - x[:2], axis=1) - x[2],
                            [*algebra[:2], initial_radius], loss='soft_l1', f_scale=.0001)
        center = origin + basis @ fit.x[:2]
        radius = float(fit.x[2])
        radial = points - center
        radial -= np.outer(radial @ normal, normal)
        lag = max(1, len(points) // 5)
        crosses = np.cross(radial[:-lag], radial[lag:]) @ normal
        if radius < .001 or abs(np.sum(crosses)) < 1e-8:
            raise ValueError('Insufficient rotation to determine circle direction')
        if np.mean(crosses * np.sign(np.sum(crosses)) > 0) < .9:
            raise ValueError('Ambiguous or reversing sweep direction')
        normal *= np.sign(np.sum(crosses)) * sweep_direction
        angles = np.unwrap(np.arctan2(xy[:, 1] - fit.x[1], xy[:, 0] - fit.x[0]))
        arc_deg = float(np.rad2deg(np.ptp(angles)))
        residual = np.hypot((points-center) @ normal,
                            np.linalg.norm(radial, axis=1)-radius)
        rms = float(np.sqrt(np.mean(residual**2)))
        if arc_deg < 5 or rms > .0005 or not fit.success:
            raise ValueError(f'Poor circle observation: arc={arc_deg:.2f} deg, RMS={rms*1000:.3f} mm')
        return dict(center_m=center, axis=normal, radius_m=radius,
                    residual_rms_m=rms, arc_deg=arc_deg, frames=len(poses),
                    c_opt=center*1000., axis_opt=normal, radius=radius*1000.,
                    rmse=rms*1000., pts_2d=xy*1000., uc_opt=fit.x[0]*1000.,
                    vc_opt=fit.x[1]*1000., measurement_accepted=True)

    @staticmethod
    def refine_adjacent_circles(datasets, circles, sweep_directions=(1, 1, 1)):
        """Fit observed A/B/C with the two fixed adjacent-axis constraints.

        All sweeps must share camera/upstream posture. Axis C is measured,
        not an encoder/FK prior. All three
        centers and radii remain independent: coincidence is tested AFTER
        fitting, never imposed to manufacture joint convergence.
        """
        if len(sweep_directions) != 3 or any(d not in (-1, 1) for d in sweep_directions):
            raise ValueError('Three commanded sweep directions are required')
        points = [np.asarray(poses)[:, :3, 3] for poses in datasets]
        nc = circles[2]['axis']
        projected = circles[0]['axis'] - (circles[0]['axis'] @ nc)*nc
        projected_b = circles[1]['axis'] - (circles[1]['axis'] @ nc)*nc
        if min(np.linalg.norm(projected), np.linalg.norm(projected_b)) < .5:
            raise ValueError('Direction-reference circle is not independent')
        ex = projected / np.linalg.norm(projected)
        ey = np.cross(nc, ex)
        frame = np.column_stack((ex, ey, nc))
        theta = np.arctan2(circles[1]['axis'] @ ey, circles[1]['axis'] @ ex)
        initial = np.r_[np.zeros(3), theta,
                        np.concatenate([c['center_m'] for c in circles]),
                        [c['radius_m'] for c in circles]]

        def axes(parameters):
            basis = R_scipy.from_rotvec(parameters[:3]).as_matrix() @ frame
            return (basis[:, 0], basis @ [np.cos(parameters[3]), np.sin(parameters[3]), 0.], basis[:, 2])

        def residual(parameters):
            terms = []
            for i, (pts, normal) in enumerate(zip(points, axes(parameters))):
                delta = pts - parameters[4+3*i:7+3*i]
                plane = delta @ normal
                radial = np.linalg.norm(delta - np.outer(plane, normal), axis=1) - parameters[13+i]
                terms.extend((plane, radial))
            return np.concatenate(terms)

        fit = least_squares(residual, initial, loss='soft_l1', f_scale=.0001,
                            x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100)
        scales = np.linalg.norm(fit.jac, axis=0)
        if not fit.success or np.any(scales < 1e-12):
            raise ValueError('Coupled observed-circle fit did not converge')
        singular = np.linalg.svd(fit.jac / scales, compute_uv=False)
        if singular[-1] < 1e-6:
            raise ValueError('Coupled observed-circle geometry is unobservable')
        refined = []
        for i, (pts, normal, previous) in enumerate(zip(points, axes(fit.x), circles)):
            center, radius = fit.x[4+3*i:7+3*i], float(fit.x[13+i])
            delta = pts - center
            plane = delta @ normal
            radial = delta - np.outer(plane, normal)
            lag = max(1, len(pts)//5)
            crosses = np.cross(radial[:-lag], radial[lag:]) @ normal
            direction = np.sign(np.sum(crosses))
            if radius < .001 or abs(np.sum(crosses)) < 1e-8 or np.mean(crosses*direction > 0) < .9:
                raise ValueError('Ambiguous refined observed-circle direction')
            normal = normal*direction*sweep_directions[i]
            u = radial[np.argmax(np.linalg.norm(radial, axis=1))]
            u = u / np.linalg.norm(u)
            xy = radial @ np.column_stack((u, np.cross(normal, u)))
            arc = float(np.rad2deg(np.ptp(np.unwrap(np.arctan2(xy[:, 1], xy[:, 0])))))
            rms = float(np.sqrt(np.mean(plane**2 + (np.linalg.norm(radial, axis=1)-radius)**2)))
            if arc < 5. or rms > .0005:
                raise ValueError('Observed circles violate adjacent-joint geometry')
            refined.append(dict(previous, center_m=center, axis=normal, radius_m=radius,
                residual_rms_m=rms, arc_deg=arc, c_opt=center*1000., axis_opt=normal,
                radius=radius*1000., rmse=rms*1000., pts_2d=xy*1000., uc_opt=0., vc_opt=0.))
        return refined

    def perform_move_to_ready_pose(self, arm_side, mode="marker", log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        self.current_calib_mode = mode
        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to {mode} Ready Pose...")
        torso = [0, 0, 0, 0, 0, 0]
        
        # 1. First move the inactive arm to zero pose to avoid collision
        if log_callback: log_callback("[INFO] Moving inactive arm to zero pose first...")
        if arm_side == "right":
            success_other = self.movej(self.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
        else:
            success_other = self.movej(self.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
            
        if not success_other:
            if log_callback: log_callback("[ERROR] Failed to move inactive arm to zero pose.")
            return False
            
        # 2. Move active arm and head/torso to ready pose
        if log_callback: log_callback("[INFO] Moving active arm, torso, and head to ready pose...")
        
        version_key = "v1.3" if self.is_v13() else "v1.2"
        
        if mode == "marker":
            type_key = "marker"
            ready_mode = None
        else:
            type_key = "joint"
            if mode == "elbow":
                ready_mode = "elbow"
            elif mode == "wrist_yaw2":
                ready_mode = "wrist_yaw2"
            elif mode in ("wrist_roll", "wrist_roll_v13"):
                ready_mode = "wrist_roll"
            else:
                ready_mode = "wrist_pitch"
            
        norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
        taught_pose = None
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            arm_dict = self.user_taught_ready_poses.get(arm_side)
            if isinstance(arm_dict, dict):
                taught_pose = arm_dict.get(norm_mode)

        if taught_pose is not None:
            if log_callback:
                log_callback(f"[INFO] Preserved user-taught ready pose detected for {arm_side} arm ({norm_mode}). Using taught posture.")
            if arm_side == "right":
                right_arm = taught_pose
                left_arm = None
            else:
                right_arm = None
                left_arm = taught_pose
        else:
            if arm_side == "right":
                right_arm = self.get_ready_pose(version_key, type_key, ready_mode, "right")
                left_arm = None
            else:
                right_arm = None
                left_arm = self.get_ready_pose(version_key, type_key, ready_mode, "left")

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=None, minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def perform_single_joint_sweep(self, arm_side, sweep_joint, q_center, start_deg, end_deg, sweep_duration, q_head=None, label="Joint Sweep", log_callback=None, **kwargs):
        if getattr(self, 'stop_requested', False):
            return None

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

        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot is not connected.")
            return None

        if self.marker_st is None:
            raise RuntimeError("Marker provider is not initialized")
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx

        # Retrieve joint limits with safety clamping
        dyn_model = self.robot.get_dynamics() if self.robot else None
        q_min = -np.inf
        q_max = np.inf
        if dyn_model and model:
            try:
                state_lim = dyn_model.make_state([f"ee_{arm_side}"], model.robot_joint_names)
                q_lower_all = np.array(dyn_model.get_limit_q_lower(state_lim))
                q_upper_all = np.array(dyn_model.get_limit_q_upper(state_lim))
                global_joint_idx = arm_idx[sweep_joint]
                q_min = q_lower_all[global_joint_idx]
                q_max = q_upper_all[global_joint_idx]
            except Exception as e:
                if log_callback: log_callback(f"[WARN] Failed to retrieve joint limits in base: {e}")

        safety_margin = np.radians(0.5)
        q_start_val = q_center[sweep_joint] + np.radians(start_deg)
        q_end_val = q_center[sweep_joint] + np.radians(end_deg)

        if q_start_val < q_min + safety_margin:
            q_start_val_new = q_min + safety_margin
            if log_callback:
                log_callback(f"[WARN] Start angle ({np.degrees(q_start_val):.2f}°) exceeds min limit ({np.degrees(q_min):.2f}°). Clamping to {np.degrees(q_start_val_new):.2f}°.")
            q_start_val = q_start_val_new

        if q_end_val > q_max - safety_margin:
            q_end_val_new = q_max - safety_margin
            if log_callback:
                log_callback(f"[WARN] End angle ({np.degrees(q_end_val):.2f}°) exceeds max limit ({np.degrees(q_max):.2f}°). Clamping to {np.degrees(q_end_val_new):.2f}°.")
            q_end_val = q_end_val_new

        q_start_val = float(np.clip(q_start_val, q_min+safety_margin, q_max-safety_margin))
        q_end_val = float(np.clip(q_end_val, q_min+safety_margin, q_max-safety_margin))
        if (q_end_val-q_start_val)*(end_deg-start_deg) <= 0:
            raise ValueError('Joint limits leave no valid sweep span')
        q_start = list(q_center)
        q_start[sweep_joint] = q_start_val
        q_end = list(q_center)
        q_end[sweep_joint] = q_end_val

        if log_callback:
            log_callback(f"[SWEEP COMMAND] {arm_side} J{sweep_joint}: "
                         f"{np.degrees(q_start_val):.4f}° -> {np.degrees(q_end_val):.4f}°; "
                         f"minimum_time={sweep_duration:.4f}s, start_move=1.2000s; "
                         f"center_deg={np.array2string(np.degrees(q_center), precision=4)}; "
                         "command already includes staged correction (apply_offsets=False)")

        # 1. Move to start position
        logging.info(f"[INFO] Moving {label} to start sweep position...")
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=q_start, head=q_head, minimum_time=1.2, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start, head=q_head, minimum_time=1.2, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback(f"[ERROR] Failed to move {label} to start pose or stop requested.")
            return None

        time.sleep(0.5)

        # 2. Continuous sweep from start to end position
        logging.info(f"[INFO] Commencing continuous sweep on {label} (duration={sweep_duration}s)...")
        if getattr(self, 'stop_requested', False):
            return None

        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end if arm_side == "right" else None,
            left_arm=q_end if arm_side == "left" else None,
            head=q_head, minimum_time=sweep_duration
        )

        dataset = []
        t_start = time.time()
        move_thread.start()

        # Sensor boundary is identical for real and synthetic observations.
        # Motion feedback never enters the Step1 measurement dataset.
        try:
            while move_thread.is_alive():
                if getattr(self, 'stop_requested', False):
                    return None
                res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)
                if res:
                    pose = np.asarray(res[0] if isinstance(res, list) else next(iter(res.values())), dtype=float).reshape(4, 4)
                    if not np.all(np.isfinite(pose)):
                        raise ValueError('Non-finite marker observation during sweep')
                    if len(dataset) == 0 or not np.array_equal(dataset[-1], pose):
                        dataset.append(pose)
                time.sleep(.01)
        finally:
            if move_thread.is_alive():
                self.robot.cancel_control()
            move_thread.join()

        move_thread.join()
        if not move_thread.success:
            if log_callback: log_callback(f"[ERROR] {label} sweep motion failed or was cancelled.")
            return None

        if len(dataset) < 10:
            if log_callback: log_callback(f"[ERROR] Too few valid captured points for {label} ({len(dataset)} points). Returning to ready pose and prompting posture adjustment...")
            try:
                sweep_mode = kwargs.get('mode', "marker" if "Marker" in label else "joint")
                self.perform_move_to_ready_pose(arm_side, mode=sweep_mode, log_callback=log_callback)
            except Exception as e:
                if log_callback: log_callback(f"[WARN] Failed to return to ready pose: {e}")
            if hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                resolved = self.marker_problem_callback(arm_side)
                if resolved:
                    sweep_mode = kwargs.get('mode', "marker" if "Marker" in label else "joint")
                    norm_mode = "wrist_pitch" if sweep_mode == "wrist_pitch_v13" else ("wrist_roll" if sweep_mode == "wrist_roll_v13" else sweep_mode)
                    if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
                        arm_dict = self.user_taught_ready_poses.get(arm_side)
                        if isinstance(arm_dict, dict) and arm_dict.get(norm_mode) is not None:
                            q_center = arm_dict.get(norm_mode)
                    return self.perform_single_joint_sweep(
                        arm_side, sweep_joint, q_center, start_deg, end_deg, sweep_duration,
                        q_head=q_head, label=label, log_callback=log_callback, **kwargs
                    )
            return None

        logging.info(f"    -> Swept {len(dataset)} dense raw coordinate frames during {label} motion.")
        return dataset
