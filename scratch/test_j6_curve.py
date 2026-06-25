import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares, minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.CalibratorBase import BaseCalibrator

class MockModel:
    def __init__(self):
        self.left_arm_idx = list(range(7))
        self.right_arm_idx = list(range(7))
        self.robot_joint_names = [f"joint_{i}" for i in range(20)]

class MockState:
    def __init__(self, q_size=20):
        self.q = np.zeros(q_size)
    def set_q(self, q):
        self.q = np.array(q)

class MockDynamics:
    def __init__(self, arm_side="right"):
        self.arm_side = arm_side
        self.L_5_ee = 300.0 # mm
        self.L_3_5 = 300.0 # mm

    def make_state(self, links, joint_names):
        return MockState()

    def compute_forward_kinematics(self, state):
        pass

    def compute_transformation(self, state, idx_from, idx_to):
        T = np.eye(4)
        T[2, 3] = self.L_3_5 / 1000.0 # 300 mm along Z axis
        return T

class MockRobot:
    def __init__(self, arm_side="right"):
        self._model = MockModel()
        self._dynamics = MockDynamics(arm_side)
    
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self._dynamics
        
    def get_state(self):
        class State:
            def __init__(self):
                self.position = np.zeros(20)
        return State()

def mock_compute_fk_impl(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
    T = np.eye(4)
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
    elif "arm_3" in ee_link:
        q3 = q[3]
        R3 = R_scipy.from_euler('Y', q3).as_matrix()
        T[:3, :3] = R3
        T[:3, 3] = [0.0, 0.0, 0.0]
    return T

# Override compute_fk
BaseCalibrator.compute_fk = mock_compute_fk_impl

def test_j6_optimization(injected_offset_deg):
    arm_side = "right"
    robot = MockRobot(arm_side)
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    calibrator.get_robot_version = lambda: "1.3"
    
    calibrator.camera_config = {
        "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, 180.0],
        "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
    }
    
    ee_name = f"ee_{arm_side}"
    T_ee_to_marker = calibrator.make_transform(calibrator.camera_config["Tf_to_marker_right_v13"])
    
    R_cam_to_torso = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    T_cam_to_torso = np.eye(4)
    T_cam_to_torso[:3, :3] = R_cam_to_torso
    T_torso_to_cam = np.linalg.inv(T_cam_to_torso)

    # Sweep A: Joint 6 sweep (from -20 to 20)
    dataset_A = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        q_captured = np.zeros(20)
        q_captured[6] = np.radians(angle_deg)
        q_actual = np.zeros(20)
        q_actual[6] = np.radians(angle_deg + injected_offset_deg)
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_A.append((q_captured, T_cam_to_marker))

    # Sweep B: Joint 5 sweep (from -20 to 20)
    dataset_B = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        q_captured = np.zeros(20)
        q_captured[5] = np.radians(angle_deg)
        q_actual = np.zeros(20)
        q_actual[5] = np.radians(angle_deg)
        q_actual[6] = np.radians(injected_offset_deg)
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_B.append((q_captured, T_cam_to_marker))

    # Calculate circles in torso frame
    poses_a_t5 = []
    for q_full, pose in dataset_A:
        T_t5_to_head = BaseCalibrator.compute_fk(robot, None, q_full, "link_head_2", "link_torso_5")
        T_t5_to_cam = T_t5_to_head @ calibrator.make_transform(calibrator.camera_config["mount_to_cam"])
        poses_a_t5.append(T_t5_to_cam @ pose)
    
    poses_b_t5 = []
    for q_full, pose in dataset_B:
        T_t5_to_head = BaseCalibrator.compute_fk(robot, None, q_full, "link_head_2", "link_torso_5")
        T_t5_to_cam = T_t5_to_head @ calibrator.make_transform(calibrator.camera_config["mount_to_cam"])
        poses_b_t5.append(T_t5_to_cam @ pose)

    angles_A = [np.degrees(q_full[6] - 0.0) for q_full, _ in dataset_A]
    angles_B = [np.degrees(q_full[5] - 0.0) for q_full, _ in dataset_B]

    res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_a_t5, angles_A, axis_prior=np.array([0.0, 0.0, 1.0]))
    res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_b_t5, angles_B, axis_prior=np.array([0.0, 1.0, 0.0]))

    c_A = res_A['c_opt']
    n_A = res_A['axis_opt']
    n_A_norm = n_A / np.linalg.norm(n_A)
    c_B = res_B['c_opt']

    def perp_dist_axis6(delta_deg):
        pts_pred = []
        for q_full, _ in dataset_B:
            q_mod = np.copy(q_full)
            q_mod[6] += np.radians(delta_deg)
            T_t5_to_ee = mock_compute_fk_impl(robot, None, q_mod, ee_name)
            T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker
            pts_pred.append(T_t5_to_marker[:3, 3] * 1000.0)
        c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_pred, robust=False)
        v = c_fit - c_B
        val = float(np.linalg.norm(v - np.dot(v, n_A_norm) * n_A_norm))
        return val

    print("\n--- Running least_squares (default) ---")
    res_opt = least_squares(lambda delta: [perp_dist_axis6(delta[0])], [0.0], bounds=([-15.0], [15.0]))
    print(f"Status: {res_opt.status}, Message: {res_opt.message}, x: {res_opt.x}")

    print("\n--- Running least_squares (diff_step=1e-3) ---")
    res_opt = least_squares(lambda delta: [perp_dist_axis6(delta[0])], [0.0], bounds=([-15.0], [15.0]), diff_step=1e-3)
    print(f"Status: {res_opt.status}, Message: {res_opt.message}, x: {res_opt.x}")

    print("\n--- Running minimize (Nelder-Mead) ---")
    res_min = minimize(lambda delta: perp_dist_axis6(delta[0]), [0.0], method='Nelder-Mead', bounds=[(-15.0, 15.0)])
    print(f"Status: {res_min.success}, Message: {res_min.message}, x: {res_min.x}")

if __name__ == "__main__":
    test_j6_optimization(3.4)
