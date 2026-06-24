import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add the calibration directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "calibration")))

from JointCalibrator import JointCalibrator
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
        # We need the transformation between link_arm_3 and link_arm_5 when q=0
        # For L35 calculation
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

# Mock compute_fk to return the actual FK using simple rotation models
# In Joint 6 calibration, Joint 6 is X-axis (roll), Joint 5 is Y-axis (pitch), Joint 4 is Z-axis (yaw)
# In Joint 5 calibration, Joint 5 is Y-axis (pitch), Joint 3 is Y-axis (pitch)
def mock_compute_fk_impl(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
    # ee_link is 'ee_right' or f"link_{arm_side}_arm_5" or f"link_{arm_side}_arm_3"
    # Joint index map for active arm:
    # Joint 3: index 3
    # Joint 4: index 4
    # Joint 5: index 5
    # Joint 6: index 6
    
    T = np.eye(4)
    if "ee" in ee_link:
        # Full FK from torso to end-effector
        # v1.3: Yaw (Z, q4) -> Pitch (Y, q5) -> Roll (Z, q6)
        # Translation of Joint 5 rel to Joint 3 = [0, 0, 0.3] (300mm along Z)
        # Translation of ee rel to Joint 5 = [0, 0, 0.3] (300mm along Z)
        q3, q4, q5, q6 = q[3], q[4], q[5], q[6]
        
        R3 = R_scipy.from_euler('Y', q3).as_matrix()
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5).as_matrix()
        R6 = R_scipy.from_euler('Z', q6).as_matrix()
        
        T[:3, :3] = R3 @ R4 @ R5 @ R6
        # Position of Joint 5 relative to Joint 3 is 300 mm along Joint 3's link
        p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
        # Position of EE relative to Joint 5 is 300 mm along Joint 5's link
        p_ee_rel_j5 = R3 @ R4 @ R5 @ np.array([0.0, 0.0, 0.3])
        T[:3, 3] = p_j5_rel_j3 + p_ee_rel_j5
        
    elif "arm_5" in ee_link:
        # FK up to link_arm_5
        q3, q4, q5 = q[3], q[4], q[5]
        R3 = R_scipy.from_euler('Y', q3).as_matrix()
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5).as_matrix()
        T[:3, :3] = R3 @ R4 @ R5
        p_j5_rel_j3 = R3 @ R4 @ np.array([0.0, 0.0, 0.3])
        T[:3, 3] = p_j5_rel_j3
        
    elif "arm_3" in ee_link:
        # FK up to link_arm_3
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
    calibrator.get_robot_version = lambda: 1.3
    
    # Setup nominal Tf_to_marker (95.0, 0.0, -5.0 mm)
    calibrator.camera_config = {
        "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, 180.0],
        "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
    }
    
    # Generate datasets
    # Injected offset: actual_offset = injected_offset_deg
    # The actual joint angle will have the offset, while q_captured has 0.
    ee_name = f"ee_{arm_side}"
    T_ee_to_marker = calibrator.make_transform(calibrator.camera_config["Tf_to_marker_right_v13"])
    
    # Camera orientation is fixed (R_cam_to_torso)
    R_cam_to_torso = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    T_cam_to_torso = np.eye(4)
    T_cam_to_torso[:3, :3] = R_cam_to_torso
    T_torso_to_cam = np.linalg.inv(T_cam_to_torso)

    # Sweep A: Joint 6 sweep (from -20 to 20)
    dataset_A = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        # Nominal captured joint position (Joint 6 = angle_deg)
        q_captured = np.zeros(20)
        q_captured[6] = np.radians(angle_deg) # J6 index is 6
        
        # Actual joint position with offset injected (J6 = angle_deg + offset)
        q_actual = np.zeros(20)
        q_actual[6] = np.radians(angle_deg + injected_offset_deg)
        
        # T_torso_to_marker
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        
        # T_cam_to_marker
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_A.append((q_captured, T_cam_to_marker))

    # Sweep B: Joint 5 sweep (from -20 to 20)
    dataset_B = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        # Nominal captured joint position (Joint 5 = angle_deg, Joint 6 = 0)
        q_captured = np.zeros(20)
        q_captured[5] = np.radians(angle_deg)
        
        # Actual joint position with offset injected (Joint 5 = angle_deg, Joint 6 = offset)
        q_actual = np.zeros(20)
        q_actual[5] = np.radians(angle_deg)
        q_actual[6] = np.radians(injected_offset_deg)
        
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_B.append((q_captured, T_cam_to_marker))

    # Run the continuous calibration sweep analysis offline
    initial_joint_pos = [0.0] * 7
    res = calibrator.compute_calibration_results(
        arm_side, "wrist_pitch_v13", dataset_A, dataset_B, initial_joint_pos,
        current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
    )
    
    recovered_offset = res['optimal_offset']
    print(f"J6 Calibration: Injected Offset = {injected_offset_deg:+.4f}°, Recovered Offset = {recovered_offset:+.4f}°")
    assert np.isclose(recovered_offset, -injected_offset_deg, atol=1e-2), f"J6 mismatch: {recovered_offset} vs {-injected_offset_deg}"
    print("J6 test passed successfully!")

def test_joint_5_calibration_recovery(injected_offset_deg):
    arm_side = "right"
    robot = MockRobot(arm_side)
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    calibrator.get_robot_version = lambda: 1.3
    
    # Setup nominal Tf_to_marker (95.0, 0.0, -5.0 mm)
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
        # Nominal captured joint position (Joint 5 = angle_deg)
        q_captured = np.zeros(20)
        q_captured[5] = np.radians(angle_deg)
        
        # Actual joint position with offset injected (J5 = angle_deg + offset)
        q_actual = np.zeros(20)
        q_actual[5] = np.radians(angle_deg + injected_offset_deg)
        
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_A.append((q_captured, T_cam_to_marker))

    # Sweep B: Joint 3 sweep (from -20 to 20)
    dataset_B = []
    for angle_deg in np.linspace(-20.0, 20.0, 30):
        # Nominal captured joint position (Joint 3 = angle_deg, Joint 5 = 0)
        q_captured = np.zeros(20)
        q_captured[3] = np.radians(angle_deg)
        
        # Actual joint position with offset injected (Joint 3 = angle_deg, Joint 5 = offset)
        q_actual = np.zeros(20)
        q_actual[3] = np.radians(angle_deg)
        q_actual[5] = np.radians(injected_offset_deg)
        
        T_torso_to_ee = mock_compute_fk_impl(robot, None, q_actual, ee_name)
        T_torso_to_marker = T_torso_to_ee @ T_ee_to_marker
        
        T_cam_to_marker = T_torso_to_cam @ T_torso_to_marker
        dataset_B.append((q_captured, T_cam_to_marker))

    # Run the continuous calibration sweep analysis offline
    initial_joint_pos = [0.0] * 7
    res = calibrator.compute_calibration_results(
        arm_side, "wrist_roll_v13", dataset_A, dataset_B, initial_joint_pos,
        current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
    )
    
    recovered_offset = res['optimal_offset']
    print(f"J5 Calibration: Injected Offset = {injected_offset_deg:+.4f}°, Recovered Offset = {recovered_offset:+.4f}°")
    assert np.isclose(recovered_offset, -injected_offset_deg, atol=0.01), f"J5 mismatch: {recovered_offset} vs {-injected_offset_deg}"
    print("J5 test passed successfully!")

def test_real_data_calibration():
    # Helper to read dataset from text file
    def read_real_sweep_file(filepath, sweep_joint_idx):
        dataset = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = [float(x) for x in line.split(',')]
                angle_deg = parts[0]
                
                # Construct q_full (20 elements)
                q_full = np.zeros(20)
                # Sweep joint index in right arm is sweep_joint_idx (0-indexed)
                q_full[sweep_joint_idx] = np.radians(angle_deg)
                
                # T_cam2marker starts at index 10
                T_flat = parts[10:26]
                T_cam_to_marker = np.array(T_flat).reshape(4, 4)
                
                dataset.append((q_full, T_cam_to_marker))
        return dataset

    arm_side = "right"
    robot = MockRobot(arm_side)
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    
    # 1. Run v1.2 wrist_pitch mode test with real data files (axis_4 and axis_6)
    calibrator.get_robot_version = lambda: 1.2
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
        
        # Run v1.2 wrist_pitch calibration
        initial_joint_pos = [0.0] * 7
        res_v12 = calibrator.compute_calibration_results(
            arm_side, "wrist_pitch", dataset_4, dataset_6, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
        )
        print(f"Real Data v1.2 wrist_pitch offset: {res_v12['optimal_offset']:.4f}° (center_dist={res_v12['center_dist']:.4f} mm)")
        print("Real Data v1.2 wrist_pitch calibration smoke test passed!")

        # Run v1.3 wrist_pitch_v13 calibration (Sweep A = J6, Sweep B = J5)
        calibrator.get_robot_version = lambda: 1.3
        calibrator.camera_config = {
            "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, 180.0],
            "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0]
        }
        res_v13 = calibrator.compute_calibration_results(
            arm_side, "wrist_pitch_v13", dataset_6, dataset_5, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False
        )
        print(f"Real Data v1.3 wrist_pitch_v13 offset: {res_v13['optimal_offset']:.4f}° (center_dist={res_v13['center_dist']:.4f} mm)")
        print("Real Data v1.3 wrist_pitch_v13 calibration smoke test passed!")
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
