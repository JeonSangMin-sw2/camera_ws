import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add the calibration directory to path to import MarkerCalibrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "calibration")))

from MarkerCalibrator import MarkerCalibrator

def test_joint_6_angle_correction_integration(arm_side, roll_offset, pitch_offset, yaw_offset, theta_6_deg):
    calibrator = MarkerCalibrator(marker_st=None, robot=None)
    
    if arm_side == "left":
        nominal_rpy = [90.0, 0.0, 0.0]
    else:
        nominal_rpy = [90.0, 0.0, 180.0]

    # ideal bracket rotation
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

    # actual bracket rotation with offset
    R_offset = R_scipy.from_euler('ZYX', [yaw_offset, pitch_offset, roll_offset], degrees=True).as_matrix()
    R_ee_m_actual_gt = R_offset @ R_ee_m_ideal

    # camera orientation relative to robot base
    R_c_ee_mid = R_scipy.from_euler('ZYX', [10.0, -5.0, 15.0], degrees=True).as_matrix()
    
    # Joint 6 sweep: R_ee = R_y(0.0) @ R_z(theta)
    theta_6_list = np.linspace(-20.0, 20.0, 30)
    poses_6 = []
    for theta in theta_6_list:
        R_ee_rot = R_scipy.from_euler('Y', 0.0, degrees=True).as_matrix() @ R_scipy.from_euler('Z', theta, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_6.append(T)
        
    # Joint 5 sweep: R_ee = R_y(theta) @ R_z(theta_6_deg)
    theta_5_list = np.linspace(-10.0, 10.0, 30)
    poses_5 = []
    for theta in theta_5_list:
        R_ee_rot = R_scipy.from_euler('Y', theta, degrees=True).as_matrix() @ R_scipy.from_euler('Z', theta_6_deg, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_5.append(T)

    # Pack into marker data
    marker_data_6 = {'captured_poses': poses_6, 'radius': 80.0, 'rmse': 0.0}
    # Pass theta_6 in radians directly for testing since robot is None
    marker_data_5 = {
        'captured_poses': poses_5, 
        'radius': 120.0, 
        'rmse': 0.0,
        'theta_6': np.radians(theta_6_deg)
    }

    # Run the actual compute_unified_bracket_calibration function from MarkerCalibrator
    result = calibrator.compute_unified_bracket_calibration(
        marker_data_5, marker_data_6, arm_side=arm_side
    )
    
    roll_e = result['roll_e']
    pitch_e = result['pitch_e']
    yaw_e = result['yaw_e']
    
    # Reconstruct rotation matrix from recovered Euler angles
    R_ee_m_rec = R_scipy.from_euler('ZYX', [yaw_e, pitch_e, roll_e], degrees=True).as_matrix()
    matrix_diff = np.linalg.norm(R_ee_m_rec - R_ee_m_actual_gt)
    
    roll_offset_rec = roll_e - nominal_rpy[0]
    pitch_offset_rec = pitch_e - nominal_rpy[1]
    yaw_diff = yaw_e - nominal_rpy[2]
    if yaw_diff > 180.0: yaw_diff -= 360.0
    elif yaw_diff < -180.0: yaw_diff += 360.0
    yaw_offset_rec = yaw_diff

    print(f"\n--- {arm_side.capitalize()} Arm 2-Axis (Roll & Pitch) Calibration Integration Test (theta_6 = {theta_6_deg}°) ---")
    print(f"Injected Offset:  Roll={roll_offset:+.3f}°, Pitch={pitch_offset:+.3f}°, Yaw={yaw_offset:+.3f}°")
    print(f"Recovered Offset: Roll={roll_offset_rec:+.3f}°, Pitch={pitch_offset_rec:+.3f}°, Yaw={yaw_offset_rec:+.3f}°")
    print(f"Matrix Diff (L2 norm): {matrix_diff:.3e} (Close to 0 means rotation is identical)")
    
    match_roll = np.isclose(roll_offset_rec, roll_offset, atol=1e-3)
    match_pitch = np.isclose(pitch_offset_rec, pitch_offset, atol=1e-3)
    match_yaw = np.isclose(yaw_offset_rec, yaw_offset, atol=1e-3)
    print(f"Matches Injected (Numeric)? Roll: {match_roll}, Pitch: {match_pitch}, Yaw: {match_yaw}")
    print(f"Rotation Matrix Matches? {np.allclose(R_ee_m_rec, R_ee_m_actual_gt, atol=1e-5)}")

def test_3axis_bracket_calibration_integration(arm_side, roll_offset, pitch_offset, yaw_offset, theta_6_deg, theta_6_4_deg):
    calibrator = MarkerCalibrator(marker_st=None, robot=None)
    
    if arm_side == "left":
        nominal_rpy = [90.0, 0.0, 0.0]
    else:
        nominal_rpy = [90.0, 0.0, 180.0]

    # ideal bracket rotation
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

    # actual bracket rotation with offset
    R_offset = R_scipy.from_euler('ZYX', [yaw_offset, pitch_offset, roll_offset], degrees=True).as_matrix()
    R_ee_m_actual_gt = R_offset @ R_ee_m_ideal

    # camera orientation relative to robot base
    R_c_ee_mid = R_scipy.from_euler('ZYX', [10.0, -5.0, 15.0], degrees=True).as_matrix()
    
    # Joint 6 sweep: R_ee = R_z(theta)
    theta_6_list = np.linspace(-20.0, 20.0, 30)
    poses_6 = []
    for theta in theta_6_list:
        R_ee_rot = R_scipy.from_euler('Z', theta, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_6.append(T)
        
    # Joint 5 sweep: R_ee = R_y(theta) @ R_z(theta_6_deg)
    theta_5_list = np.linspace(-10.0, 10.0, 30)
    poses_5 = []
    for theta in theta_5_list:
        R_ee_rot = R_scipy.from_euler('Y', theta, degrees=True).as_matrix() @ R_scipy.from_euler('Z', theta_6_deg, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_5.append(T)

    # Joint 4 sweep: R_ee = R_x(theta) @ R_z(theta_6_4_deg)
    theta_4_list = np.linspace(-10.0, 10.0, 30)
    poses_4 = []
    for theta in theta_4_list:
        R_ee_rot = R_scipy.from_euler('X', theta, degrees=True).as_matrix() @ R_scipy.from_euler('Z', theta_6_4_deg, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_4.append(T)

    # Pack into marker data
    marker_data_6 = {'captured_poses': poses_6, 'radius': 80.0, 'rmse': 0.0}
    marker_data_5 = {
        'captured_poses': poses_5, 
        'radius': 120.0, 
        'rmse': 0.0,
        'theta_6': np.radians(theta_6_deg)
    }
    marker_data_4 = {
        'captured_poses': poses_4,
        'radius': 100.0,
        'rmse': 0.0,
        'theta_6': np.radians(theta_6_4_deg)
    }

    # Run compute_unified_bracket_calibration with marker_data_4
    result = calibrator.compute_unified_bracket_calibration(
        marker_data_5, marker_data_6, arm_side=arm_side, marker_data_4=marker_data_4
    )
    
    roll_e = result['roll_e']
    pitch_e = result['pitch_e']
    yaw_e = result['yaw_e']
    
    # Reconstruct rotation matrix from recovered Euler angles
    R_ee_m_rec = R_scipy.from_euler('ZYX', [yaw_e, pitch_e, roll_e], degrees=True).as_matrix()
    matrix_diff = np.linalg.norm(R_ee_m_rec - R_ee_m_actual_gt)
    
    roll_offset_rec = roll_e - nominal_rpy[0]
    pitch_offset_rec = pitch_e - nominal_rpy[1]
    yaw_diff = yaw_e - nominal_rpy[2]
    if yaw_diff > 180.0: yaw_diff -= 360.0
    elif yaw_diff < -180.0: yaw_diff += 360.0
    yaw_offset_rec = yaw_diff

    print(f"\n--- {arm_side.capitalize()} Arm 3-Axis (Roll, Pitch, Yaw) Calibration Integration Test (theta_6_5={theta_6_deg}°, theta_6_4={theta_6_4_deg}°) ---")
    print(f"Injected Offset:  Roll={roll_offset:+.3f}°, Pitch={pitch_offset:+.3f}°, Yaw={yaw_offset:+.3f}°")
    print(f"Recovered Offset: Roll={roll_offset_rec:+.3f}°, Pitch={pitch_offset_rec:+.3f}°, Yaw={yaw_offset_rec:+.3f}°")
    print(f"Matrix Diff (L2 norm): {matrix_diff:.3e} (Close to 0 means rotation is identical)")
    
    match_roll = np.isclose(roll_offset_rec, roll_offset, atol=1e-3)
    match_pitch = np.isclose(pitch_offset_rec, pitch_offset, atol=1e-3)
    match_yaw = np.isclose(yaw_offset_rec, yaw_offset, atol=1e-3)
    print(f"Matches Injected (Numeric)? Roll: {match_roll}, Pitch: {match_pitch}, Yaw: {match_yaw}")
    print(f"Rotation Matrix Matches? {np.allclose(R_ee_m_rec, R_ee_m_actual_gt, atol=1e-5)}")

def test_v13_bracket_calibration_and_optimization(arm_side, x_e_gt, y_e_gt, z_e_gt, d5_gt_deg, d6_gt_deg):
    calibrator = MarkerCalibrator(marker_st=None, robot=None)
    calibrator.get_robot_version = lambda: 1.3
    calibrator.camera_config = {"mount_to_cam": [0.0, 0.0, 0.0, -90, 0.0, -90]}
    
    # 1. Mock compute_fk to implement the nominal v1.3 kinematics
    def mock_compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        # Joint 4 (yaw): q[4] -> Z-axis
        # Joint 5 (pitch): q[5] -> Y-axis
        # Joint 6 (roll): q[6] -> X-axis
        q4, q5, q6 = q[4], q[5], q[6]
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5).as_matrix()
        R6 = R_scipy.from_euler('X', q6).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R4 @ R5 @ R6
        T[:3, 3] = [0.0, 0.0, 0.3] # L_5_ee = 300mm
        return T

    calibrator.compute_fk = mock_compute_fk

    # 2. Setup ground truth values
    P_ee_gt = np.array([x_e_gt, y_e_gt, z_e_gt]) / 1000.0
    nominal_rpy = [90.0, 0.0, -90.0]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
    T_ee_to_marker_gt = np.eye(4)
    T_ee_to_marker_gt[:3, :3] = R_ee_m_ideal
    T_ee_to_marker_gt[:3, 3] = P_ee_gt

    d5_gt_rad = np.radians(d5_gt_deg)
    d6_gt_rad = np.radians(d6_gt_deg)

    # Helper function to compute actual kinematics (with injected joint offsets)
    def mock_compute_fk_actual(q):
        q4, q5, q6 = q[4], q[5], q[6]
        q5_act = q5 + d5_gt_rad
        q6_act = q6 + d6_gt_rad
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5_act).as_matrix()
        R6 = R_scipy.from_euler('X', q6_act).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R4 @ R5 @ R6
        T[:3, 3] = [0.0, 0.0, 0.3]
        return T

    # 3. Calculate target radii based on gt coordinates (L_5_ee = 300 mm)
    L_5_ee = 300.0
    r6_gt = np.sqrt(y_e_gt**2 + z_e_gt**2)
    p_j6_0 = R_scipy.from_euler('X', d6_gt_rad).as_matrix() @ (P_ee_gt * 1000.0) + [0.0, 0.0, L_5_ee]
    r5_gt = np.sqrt(p_j6_0[0]**2 + p_j6_0[2]**2)
    p_j5_0 = R_scipy.from_euler('Y', d5_gt_rad).as_matrix() @ p_j6_0
    r4_gt = np.sqrt(p_j5_0[0]**2 + p_j5_0[1]**2)

    # 4. Generate J4 sweep dataset
    T_t5_to_cam = calibrator.make_transform([0.0, 0.0, 0.0, -90.0, 0.0, -90.0])
    poses_4 = []
    q_fulls_4 = []
    for q4_deg in np.linspace(-10.0, 10.0, 30):
        q = np.zeros(20)
        q[4] = np.radians(q4_deg)
        T_act = mock_compute_fk_actual(q)
        poses_4.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_4.append(q)
    marker_data_4 = {'captured_poses': poses_4, 'captured_q_full': q_fulls_4, 'radius': r4_gt, 'rmse': 0.0}

    # 5. Generate J5 sweep dataset
    poses_5 = []
    q_fulls_5 = []
    for q5_deg in np.linspace(-10.0, 10.0, 30):
        q = np.zeros(20)
        q[5] = np.radians(q5_deg)
        T_act = mock_compute_fk_actual(q)
        poses_5.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_5.append(q)
    marker_data_5 = {'captured_poses': poses_5, 'captured_q_full': q_fulls_5, 'radius': r5_gt, 'rmse': 0.0}

    # 6. Generate J6 sweep dataset
    poses_6 = []
    q_fulls_6 = []
    for q6_deg in np.linspace(-20.0, 20.0, 30):
        q = np.zeros(20)
        q[6] = np.radians(q6_deg)
        T_act = mock_compute_fk_actual(q)
        poses_6.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_6.append(q)
    marker_data_6 = {'captured_poses': poses_6, 'captured_q_full': q_fulls_6, 'radius': r6_gt, 'rmse': 0.0}

    # 7. Run the calibration optimizer
    res = calibrator.compute_unified_bracket_calibration_v1_3(
        marker_data_5=marker_data_5,
        marker_data_6=marker_data_6,
        arm_side=arm_side,
        marker_data_4=marker_data_4
    )

    opt_d5 = res['opt_delta_5']
    opt_d6 = res['opt_delta_6']
    x_e = res['x_e']
    y_e = res['y_e']
    z_e = res['z_e']
    
    print(f"\n--- v1.3 Joint 4/5/6 Variance Optimization Test ({arm_side.capitalize()} Arm) ---")
    print(f"Injected Offsets:   d5 = {d5_gt_deg:+.3f} deg, d6 = {d6_gt_deg:+.3f} deg")
    print(f"Recovered Offsets:  d5 = {opt_d5:+.3f} deg, d6 = {opt_d6:+.3f} deg")
    print(f"Injected Position:  X={x_e_gt:.2f}, Y={y_e_gt:.2f}, Z={z_e_gt:.2f} mm")
    print(f"Recovered Position: X={x_e:.2f}, Y={y_e:.2f}, Z={z_e:.2f} mm")
    print(f"Min Circle Fitting Radius: {res['min_radius']:.4f} mm")

    assert np.isclose(opt_d5, d5_gt_deg, atol=1e-2), f"d5 mismatch: {opt_d5} vs {d5_gt_deg}"
    assert np.isclose(opt_d6, d6_gt_deg, atol=1e-2), f"d6 mismatch: {opt_d6} vs {d6_gt_deg}"
    assert np.isclose(x_e, x_e_gt, atol=1e-2), f"X mismatch: {x_e} vs {x_e_gt}"
    assert np.isclose(y_e, y_e_gt, atol=1e-2), f"Y mismatch: {y_e} vs {y_e_gt}"
    assert np.isclose(z_e, z_e_gt, atol=1e-2), f"Z mismatch: {z_e} vs {z_e_gt}"
    print("Test passed successfully!")

if __name__ == "__main__":
    # Test Left Arm
    test_joint_6_angle_correction_integration("left", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, theta_6_deg=17.49)
    test_3axis_bracket_calibration_integration("left", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, theta_6_deg=17.49, theta_6_4_deg=10.0)
    test_v13_bracket_calibration_and_optimization("left", x_e_gt=74.8, y_e_gt=0.0, z_e_gt=-50.1, d5_gt_deg=4.5, d6_gt_deg=-6.2)
    
    # Test Right Arm
    test_joint_6_angle_correction_integration("right", roll_offset=2.5, pitch_offset=1.2, yaw_offset=-3.4, theta_6_deg=-15.3)
    test_3axis_bracket_calibration_integration("right", roll_offset=2.5, pitch_offset=1.2, yaw_offset=-3.4, theta_6_deg=-15.3, theta_6_4_deg=-5.0)
    test_v13_bracket_calibration_and_optimization("right", x_e_gt=74.8, y_e_gt=0.0, z_e_gt=-50.1, d5_gt_deg=-3.8, d6_gt_deg=5.4)


