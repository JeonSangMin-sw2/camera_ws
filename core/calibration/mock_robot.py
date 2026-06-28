import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# ==============================================================================
# 1. Pure Mock Classes (Pure mathematical mock when SDK/URDF is unavailable)
# ==============================================================================

class MockRobotState:
    def __init__(self, dof=20):
        self.position = np.zeros(dof)

class PureMockModel:
    def __init__(self):
        self.left_arm_idx = list(range(7))
        self.right_arm_idx = list(range(7))
        self.head_idx = [18, 19]
        self.torso_idx = list(range(10, 16))
        self.robot_joint_names = [f"joint_{i}" for i in range(20)]

class PureMockState:
    def __init__(self, q_size=20):
        self.q = np.zeros(q_size)
    def set_q(self, q):
        self.q = np.array(q)

class PureMockDynamics:
    def __init__(self, arm_side="right"):
        self.arm_side = arm_side
        self.L_5_ee = 300.0  # mm
        self.L_3_5 = 300.0  # mm

    def make_state(self, links, joint_names):
        return PureMockState()

    def compute_forward_kinematics(self, state):
        pass

    def compute_transformation(self, state, idx_from, idx_to):
        T = np.eye(4)
        T[2, 3] = self.L_3_5 / 1000.0
        return T

    def get_limit_q_lower(self, state):
        return np.full(20, -3.14)

    def get_limit_q_upper(self, state):
        return np.full(20, 3.14)

    def compute_diff_forward_kinematics(self, state):
        pass

    def compute_body_jacobian(self, state, idx_from, idx_to):
        return np.zeros((6, 20))


class PureMockRobot:
    def __init__(self, arm_side="right", model_name="m"):
        self._model = PureMockModel()
        self._dynamics = PureMockDynamics(arm_side)
        self.is_pure_mock = True
        self.model_name = model_name
        self._state = MockRobotState(20)
    
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self._dynamics
        
    def get_state(self):
        return self._state

    def send_command(self, command, priority=10):
        class CommandResult:
            def __init__(self):
                try:
                    import rby1_sdk as rby
                    self.finish_code = rby.RobotCommandFeedback.FinishCode.Ok
                except ImportError:
                    class FinishCode:
                        Ok = 0
                    self.finish_code = FinishCode.Ok
        
        class CommandFeedback:
            def get(self):
                return CommandResult()
                
        return CommandFeedback()

    def cancel_control(self):
        pass

    def get_robot_info(self):
        class RobotInfo:
            def __init__(self, version):
                self.robot_model_version = version
        version = "1.3" if self.model_name == "m" else "1.2"
        return RobotInfo(version)

def pure_mock_compute_fk_impl(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
    T = np.eye(4)
    model_name = getattr(robot, "model_name", "a")
    
    if model_name == "m":
        # v1.3 Kinematics: Y (q3) -> Z (q4) -> Y (q5) -> X (q6)
        if "ee" in ee_link:
            q3, q4, q5, q6 = q[3], q[4], q[5], q[6]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            R5 = R_scipy.from_euler('Y', q5).as_matrix()
            R6 = R_scipy.from_euler('X', q6).as_matrix()
            T[:3, :3] = R3 @ R4 @ R5 @ R6
            p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
            p_ee_rel_j5 = R3 @ R4 @ R5 @ np.array([0.0, 0.0, 0.3])
            T[:3, 3] = p_j5_rel_j3 + p_ee_rel_j5
        elif "arm_5" in ee_link:
            q3, q4, q5 = q[3], q[4], q[5]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            R5 = R_scipy.from_euler('Y', q5).as_matrix()
            T[:3, :3] = R3 @ R4 @ R5
            p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
            T[:3, 3] = p_j5_rel_j3
        elif "arm_4" in ee_link:
            q3, q4 = q[3], q[4]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            T[:3, :3] = R3 @ R4
            T[:3, 3] = R3 @ np.array([0.0, 0.0, 0.3])
        elif "arm_3" in ee_link:
            q3 = q[3]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            T[:3, :3] = R3
            T[:3, 3] = [0.0, 0.0, 0.0]
    else:
        # v1.2 Kinematics: Y (q3) -> Z (q4) -> Y (q5) -> Z (q6)
        if "ee" in ee_link:
            q3, q4, q5, q6 = q[3], q[4], q[5], q[6]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            R5 = R_scipy.from_euler('Y', q5).as_matrix()
            R6 = R_scipy.from_euler('Z', q6).as_matrix()
            T[:3, :3] = R3 @ R4 @ R5 @ R6
            p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
            p_ee_rel_j5 = R3 @ R4 @ R5 @ np.array([0.0, 0.0, 0.3])
            T[:3, 3] = p_j5_rel_j3 + p_ee_rel_j5
        elif "arm_5" in ee_link:
            q3, q4, q5 = q[3], q[4], q[5]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            R5 = R_scipy.from_euler('Y', q5).as_matrix()
            T[:3, :3] = R3 @ R4 @ R5
            p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
            T[:3, 3] = p_j5_rel_j3
        elif "arm_4" in ee_link:
            q3, q4 = q[3], q[4]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            R4 = R_scipy.from_euler('Z', q4).as_matrix()
            T[:3, :3] = R3 @ R4
            T[:3, 3] = R3 @ np.array([0.0, 0.0, 0.3])
        elif "arm_3" in ee_link:
            q3 = q[3]
            R3 = R_scipy.from_euler('Y', q3).as_matrix()
            T[:3, :3] = R3
            T[:3, 3] = [0.0, 0.0, 0.0]
    return T

# ==============================================================================
# 2. Offline SDK Dynamics Wrapper (When local URDF is loaded)
# ==============================================================================

class OfflineRobot:
    def __init__(self, dyn_robot, model_name="m"):
        self.dyn_robot = dyn_robot
        self._model = self._create_model_meta()
        self.is_pure_mock = False
        self.model_name = model_name
        self._state = MockRobotState(self.dyn_robot.get_dof())
        
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self.dyn_robot
        
    def get_state(self):
        return self._state
        
    def send_command(self, command, priority=10):
        class CommandResult:
            def __init__(self):
                try:
                    import rby1_sdk as rby
                    self.finish_code = rby.RobotCommandFeedback.FinishCode.Ok
                except ImportError:
                    class FinishCode:
                        Ok = 0
                    self.finish_code = FinishCode.Ok
        
        class CommandFeedback:
            def get(self):
                return CommandResult()
                
        return CommandFeedback()

    def cancel_control(self):
        pass

    def get_robot_info(self):
        class RobotInfo:
            def __init__(self, version):
                self.robot_model_version = version
        version = "1.3" if self.model_name == "m" else "1.2"
        return RobotInfo(version)
        
    def _create_model_meta(self):
        class ModelMeta:
            def __init__(self, dyn_robot):
                self.robot_joint_names = dyn_robot.get_joint_names()
                self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
                self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
                
                self.head_idx = []
                for name in ["head_0", "head_1", "head_yaw", "head_pitch"]:
                    if name in self.robot_joint_names:
                        self.head_idx.append(self.robot_joint_names.index(name))
                if len(self.head_idx) < 2:
                    self.head_idx = [i for i, n in enumerate(self.robot_joint_names) if "head" in n.lower()][:2]
                if len(self.head_idx) < 2:
                    self.head_idx = [18, 19]
                    
                self.torso_idx = []
                for i in range(6):
                    for name in [f"torso_{i}", f"torso_joint_{i}"]:
                        if name in self.robot_joint_names:
                            self.torso_idx.append(self.robot_joint_names.index(name))
                if len(self.torso_idx) < 6:
                    self.torso_idx = [i for i, n in enumerate(self.robot_joint_names) if "torso" in n.lower()][:6]
                if len(self.torso_idx) < 6:
                    self.torso_idx = list(range(6))
        return ModelMeta(self.dyn_robot)

# ==============================================================================
# 3. Main Factory Function
# ==============================================================================

def get_mock_robot(address="127.0.0.1", model_name="m", arm_side="right"):
    """
    Attempts simulator connection, falls back to offline URDF loading,
    and finally falls back to pure mathematical MockRobot.
    """
    # 1st Step: Simulator Connection
    try:
        import rby1_sdk
        print(f"[mock_robot] Attempting connection to simulated robot at {address} (model: {model_name})...")
        robot = rby1_sdk.create_robot(address, model_name)
        if robot.connect():
            print(f"[mock_robot] Successfully connected to simulated robot at {address}!")
            robot.is_pure_mock = False
            robot.model_name = model_name
            return robot
        else:
            print("[mock_robot] Simulator connection failed. Trying offline URDF loading...")
    except ImportError:
        print("[mock_robot] rby1_sdk is not installed. Trying offline URDF loading...")

    # 2nd Step: URDF File Loading
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
    if not os.path.exists(urdf_path):
        candidates = [
            "./model_v1.3.urdf",
            "../models/rby1m/urdf/model_v1.3.urdf",
            "./core/calibration/model_v1.3.urdf"
        ]
        for c in candidates:
            if os.path.exists(c):
                urdf_path = c
                break

    if os.path.exists(urdf_path):
        try:
            import rby1_sdk.dynamics as rd
            print(f"[mock_robot] Loading offline URDF from {urdf_path}...")
            robot_config = rd.load_robot_from_urdf(urdf_path, "base")
            dyn_robot = rd.Robot(robot_config)
            return OfflineRobot(dyn_robot, model_name)
        except Exception as e:
            print(f"[mock_robot] Failed to load offline URDF: {e}")
    else:
        print("[mock_robot] Offline URDF file not found.")

    # 3rd Step: Mathematical Fallback
    print("[mock_robot] Falling back to pure mathematical MockRobot.")
    return PureMockRobot(arm_side, model_name)
