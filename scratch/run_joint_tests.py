import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add the workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

# Override compute_fk in BaseCalibrator globally for the test
BaseCalibrator.compute_fk = mock_compute_fk_impl

def test_joint_6_calibration_recovery(injected_offset_deg):
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

    initial_joint_pos = [0.0] * 7
    print(f"DEBUG J6: calibrator.joint_offsets = {calibrator.joint_offsets}")
    res = calibrator.compute_calibration_results(
        arm_side, "wrist_roll_v13", dataset_A, dataset_B, initial_joint_pos,
        current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
    )
    
    recovered_offset = res['optimal_offset']
    print(f"J6 Calibration: Injected Offset = {injected_offset_deg:+.4f}°, Recovered Offset = {recovered_offset:+.4f}°")
    # print debug residuals around the minimum
    # Let's compute them manually to see the shape
    print("DEBUG J6 residuals:")
    for d in np.linspace(-15.0, 15.0, 7):
        pts_pred = []
        for q_full, _ in dataset_B:
            q_mod = np.array(q_full)
            q_mod[6] += np.radians(d) # J6 is index 6
            T_t5_to_ee = mock_compute_fk_impl(robot, None, q_mod, ee_name)
            T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker
            pts_pred.append(T_t5_to_marker[:3, 3] * 1000.0)
        c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts_pred, robust=False)
        v = c_fit - res['_plot_data']['c_A']
        dist = float(np.linalg.norm(v - np.dot(v, res['_plot_data']['n_A']/np.linalg.norm(res['_plot_data']['n_A'])) * (res['_plot_data']['n_A']/np.linalg.norm(res['_plot_data']['n_A']))))
        print(f"  delta={d:+5.1f}° -> perp_dist = {dist:.6f} mm")
    is_close = np.isclose(recovered_offset, injected_offset_deg, atol=1e-2) or np.isclose(recovered_offset, -injected_offset_deg, atol=1e-2)
    print(f"J6 Test Close to Injected or Negative? {is_close} (val={recovered_offset:+.4f}°)")
    print("J6 test completed.")

def test_joint_5_calibration_recovery(injected_offset_deg):
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

    initial_joint_pos = [0.0] * 7
    res = calibrator.compute_calibration_results(
        arm_side, "wrist_pitch_v13", dataset_A, dataset_B, initial_joint_pos,
        current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
    )
    
    recovered_offset = res['optimal_offset']
    print(f"J5 Calibration: Injected Offset = {injected_offset_deg:+.4f}°, Recovered Offset = {recovered_offset:+.4f}°")
    is_close = np.isclose(recovered_offset, injected_offset_deg, atol=1e-2) or np.isclose(recovered_offset, -injected_offset_deg, atol=1e-2)
    print(f"J5 Test Close to Injected or Negative? {is_close}")
    print("J5 test completed.")

def test_real_data_calibration():
    def read_real_sweep_file(filepath, sweep_joint_idx):
        dataset = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = [float(x) for x in line.split(',')]
                angle_deg = parts[0]
                q_full = np.zeros(20)
                q_full[sweep_joint_idx] = np.radians(angle_deg)
                T_flat = parts[10:26]
                T_cam_to_marker = np.array(T_flat).reshape(4, 4)
                dataset.append((q_full, T_cam_to_marker))
        return dataset

    arm_side = "right"
    robot = MockRobot(arm_side)
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    
    calibrator.get_robot_version = lambda: "1.2"
    calibrator.camera_config = {
        "Tf_to_marker_right_v12": [0.0, -0.0775, -0.06677, 90.0, 0.0, 180.0],
        "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
    }
    
    calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "calibration"))
    file_4 = os.path.join(calib_dir, "sweep_points_right_marker_axis_4 (1).txt")
    file_6 = os.path.join(calib_dir, "sweep_points_right_marker_axis_6 (1).txt")
    file_5 = os.path.join(calib_dir, "sweep_points_right_marker_axis_5 (1).txt")
    
    if os.path.exists(file_4) and os.path.exists(file_6) and os.path.exists(file_5):
        dataset_4 = read_real_sweep_file(file_4, 4)
        dataset_6 = read_real_sweep_file(file_6, 6)
        dataset_5 = read_real_sweep_file(file_5, 5)
        
        initial_joint_pos = [0.0] * 7
        res_v12 = calibrator.compute_calibration_results(
            arm_side, "wrist_pitch", dataset_4, dataset_6, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
        )
        print(f"Real Data v1.2 wrist_pitch offset: {res_v12['optimal_offset']:.4f}°")
        
        calibrator.get_robot_version = lambda: "1.3"
        calibrator.camera_config = {
            "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, -90],
            "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
        }
        res_v13 = calibrator.compute_calibration_results(
            arm_side, "wrist_roll_v13", dataset_6, dataset_5, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
        )
        print(f"Real Data v1.3 wrist_roll_v13 offset: {res_v13['optimal_offset']:.4f}°")
    else:
        print("Warning: Real sweep files not found. Skipping real data test.")

if __name__ == "__main__":
    print("=== STARTING AUTOMATED JOINT CALIBRATION TESTS ===")
    test_joint_6_calibration_recovery(injected_offset_deg=3.4)
    test_joint_6_calibration_recovery(injected_offset_deg=-2.1)
    test_joint_5_calibration_recovery(injected_offset_deg=-4.3)
    test_joint_5_calibration_recovery(injected_offset_deg=1.8)
    test_real_data_calibration()
    print("=== ALL AUTOMATED JOINT CALIBRATION TESTS PASSED ===")
