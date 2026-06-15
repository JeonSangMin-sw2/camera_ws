import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add the calibration directory to path to import MarkerCalibrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration")))

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
    
    roll_offset_rec = roll_e - nominal_rpy[0]
    pitch_offset_rec = pitch_e - nominal_rpy[1]
    yaw_diff = yaw_e - nominal_rpy[2]
    if yaw_diff > 180.0: yaw_diff -= 360.0
    elif yaw_diff < -180.0: yaw_diff += 360.0
    yaw_offset_rec = yaw_diff

    print(f"--- Left Arm Bracket Calibration Integration Test (theta_6 = {theta_6_deg}°) ---")
    print(f"Injected Offset:  Roll={roll_offset:+.3f}°, Pitch={pitch_offset:+.3f}°, Yaw={yaw_offset:+.3f}°")
    print(f"Recovered Offset: Roll={roll_offset_rec:+.3f}°, Pitch={pitch_offset_rec:+.3f}°, Yaw={yaw_offset_rec:+.3f}°")
    
    match_roll = np.isclose(roll_offset_rec, roll_offset, atol=1e-3)
    match_pitch = np.isclose(pitch_offset_rec, pitch_offset, atol=1e-3)
    match_yaw = np.isclose(yaw_offset_rec, yaw_offset, atol=1e-3)
    print(f"Matches Injected? Roll: {match_roll}, Pitch: {match_pitch}, Yaw: {match_yaw}")

if __name__ == "__main__":
    test_joint_6_angle_correction_integration("left", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, theta_6_deg=17.49)
