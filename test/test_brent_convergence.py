import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize_scalar

# Add the calibration directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "calibration")))

from CalibratorBase import BaseCalibrator

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
        T[2, 3] = self.L_3_5 / 1000.0
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

BaseCalibrator.compute_fk = mock_compute_fk_impl

def test_j5_recovery(injected_offset_deg):
    robot = MockRobot("right")
    dyn_model = robot.get_dynamics()
    arm_idx = robot.model().right_arm_idx
    ee_name = "ee_right"
    
    camera_config = {
        "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, 180.0],
        "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
    }
    
    T_ee_to_marker = BaseCalibrator.make_transform(camera_config["Tf_to_marker_right_v13"])
    R_cam_to_torso = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    T_cam_to_torso = np.eye(4)
    T_cam_to_torso[:3, :3] = R_cam_to_torso
    T_torso_to_cam = np.linalg.inv(T_cam_to_torso)

    # Sweep A: Joint 5 sweep (from -20 to 20)
    dataset_A = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        q_captured = np.zeros(20)
        q_captured[5] = np.radians(angle_deg)
        q_actual = np.zeros(20)
        q_actual[5] = np.radians(angle_deg + injected_offset_deg)
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_A.append((q_captured, T_cam_to_marker))

    # Sweep B: Joint 3 sweep (from -20 to 20)
    dataset_B = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        q_captured = np.zeros(20)
        q_captured[3] = np.radians(angle_deg)
        q_actual = np.zeros(20)
        q_actual[3] = np.radians(angle_deg)
        q_actual[5] = np.radians(injected_offset_deg)
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_B.append((q_captured, T_cam_to_marker))

    # Reference fitting using fit_circle_3d_and_6dof_misalignment
    poses_a_t5 = [T_cam_to_torso @ pose for _, pose in dataset_A]
    poses_b_t5 = [T_cam_to_torso @ pose for _, pose in dataset_B]
    angles_A = [np.degrees(q_full[arm_idx[5]]) for q_full, _ in dataset_A]
    angles_B = [np.degrees(q_full[arm_idx[3]]) for q_full, _ in dataset_B]
    
    a_A_prior = np.array([0.0, 1.0, 0.0])
    a_B_prior = np.array([0.0, 1.0, 0.0])
    
    res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_a_t5, angles_A, axis_prior=a_A_prior)
    res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_b_t5, angles_B, axis_prior=a_B_prior)
    
    r_A = res_A['radius']
    r_B = res_B['radius']
    
    r_3_meas = r_B
    
    # Scale nominal Tf_to_marker
    tf_vec = camera_config["Tf_to_marker_right_v13"]
    tf_vec_mod = list(tf_vec)
    L5_ee = 300.0
    x_m = tf_vec_mod[0] * 1000.0
    z_m = tf_vec_mod[2] * 1000.0
    orig_radius = np.sqrt(x_m**2 + (L5_ee + z_m)**2)
    scale = r_A / orig_radius
    tf_vec_mod[0] = (x_m * scale) / 1000.0
    tf_vec_mod[2] = ((L5_ee + z_m) * scale - L5_ee) / 1000.0
    T_ee_to_marker_scaled = BaseCalibrator.make_transform(tf_vec_mod)

    # Prediction inside optimization loop
    def compute_elbow_radius(delta_deg):
        poses_pred = []
        for q_full, _ in dataset_B:
            q_mod = np.array(q_full)
            q_mod[arm_idx[5]] += np.radians(delta_deg)
            T_t5_to_ee = mock_compute_fk_impl(robot, None, q_mod, ee_name)
            T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker_scaled
            poses_pred.append(T_t5_to_marker)
        
        # Fit predicted using same alignment method!
        res_pred = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
            poses_pred, angles_B, axis_prior=a_B_prior
        )
        return res_pred['radius']

    def loss(delta_deg):
        return (compute_elbow_radius(delta_deg) - r_3_meas) ** 2

    print(f"DEBUG: r_A={r_A}, r_B={r_B}, orig_radius={orig_radius}, scale={scale}")
    print("DEBUG Loss scan:")
    for d in np.linspace(-10.0, 10.0, 21):
        print(f"  delta={d:+.1f} => loss={loss(d):.6f}, radius={compute_elbow_radius(d):.4f}")

    # Run minimize_scalar
    res_opt = minimize_scalar(loss, bounds=(-15.0, 15.0), method='bounded')
    recovered_offset = res_opt.x
    print(f"J5 Test (minimize_scalar + fit_circle_3d_and_6dof_misalignment):")
    print(f"  Injected Offset: {injected_offset_deg:+.4f}°")
    print(f"  Recovered Offset: {recovered_offset:+.4f}°")
    print(f"  Success: {np.isclose(recovered_offset, injected_offset_deg, atol=1e-2)}")

if __name__ == "__main__":
    test_j5_recovery(-4.3)
    test_j5_recovery(1.8)
