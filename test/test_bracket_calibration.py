import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add the calibration directory to path to import MarkerCalibrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration")))

from MarkerCalibrator import MarkerCalibrator

def test_offset_definition(arm_side, roll_offset, pitch_offset, yaw_offset, offset_frame="ee"):
    calibrator = MarkerCalibrator(marker_st=None, robot=None)
    
    # 1. Define nominal (ideal) bracket angles
    if arm_side == "left":
        nominal_rpy = [90.0, 0.0, 0.0]
    else:
        nominal_rpy = [90.0, 0.0, 180.0]

    # 2. Build R_ee_m_ideal (same as in compute_unified_bracket_calibration)
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

    # 3. Build ground truth actual rotation R_ee_m_actual with injected offset
    R_offset = R_scipy.from_euler('ZYX', [yaw_offset, pitch_offset, roll_offset], degrees=True).as_matrix()
    
    if offset_frame == "ee":
        # Offset applied in the End-Effector frame (tool flange)
        R_ee_m_actual_gt = R_offset @ R_ee_m_ideal
    else:
        # Offset applied in the Marker frame
        R_ee_m_actual_gt = R_ee_m_ideal @ R_offset

    # 4. Generate 30 virtual data points
    theta_6_list = np.linspace(-20.0, 20.0, 30) # in degrees
    poses_6 = []
    
    R_c_ee_mid = R_scipy.from_euler('ZYX', [10.0, -5.0, 15.0], degrees=True).as_matrix() # arbitrary camera pose
    
    for theta in theta_6_list:
        R_ee_rot = R_scipy.from_euler('Z', theta, degrees=True).as_matrix()
        R_c_ee = R_c_ee_mid @ R_ee_rot
        R_c_m = R_c_ee @ R_ee_m_actual_gt
        
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_6.append(T)
        
    theta_5_list = np.linspace(-10.0, 10.0, 30) # in degrees
    poses_5 = []
    for theta in theta_5_list:
        R_ee_rot = R_scipy.from_euler('Y', theta, degrees=True).as_matrix()
        R_c_ee = R_c_ee_mid @ R_ee_rot
        R_c_m = R_c_ee @ R_ee_m_actual_gt
        
        T = np.eye(4)
        T[:3, :3] = R_c_m
        T[:3, 3] = [0.1, 0.2, 0.5]
        poses_5.append(T)

    # Pack into marker data
    marker_data_6 = {'captured_poses': poses_6, 'radius': 80.0, 'rmse': 0.0}
    marker_data_5 = {'captured_poses': poses_5, 'radius': 120.0, 'rmse': 0.0}

    # Run Case A (Current implementation)
    result = calibrator.compute_unified_bracket_calibration(
        marker_data_5, marker_data_6, arm_side=arm_side
    )
    
    roll_e = result['roll_e']
    pitch_e = result['pitch_e']
    yaw_e = result['yaw_e']
    
    # Check recovered offsets relative to nominal
    roll_offset_rec = roll_e - nominal_rpy[0]
    pitch_offset_rec = pitch_e - nominal_rpy[1]
    yaw_diff = yaw_e - nominal_rpy[2]
    if yaw_diff > 180.0: yaw_diff -= 360.0
    elif yaw_diff < -180.0: yaw_diff += 360.0
    yaw_offset_rec = yaw_diff

    # Case B (Alternative: actual n5 based)
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    
    def extract_axis_from_rotations(poses, ideal_axis):
        if len(poses) < 2: return ideal_axis
        mid_idx = len(poses) // 2
        R_ref = poses[mid_idx][:3, :3]
        axes = []
        for i, T in enumerate(poses):
            if i == mid_idx: continue
            R_rel = R_ref.T @ T[:3, :3] 
            rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > np.radians(1.0):
                axis = rotvec / angle
                if np.dot(axis, ideal_axis) < 0: axis = -axis
                axes.append(axis)
        if len(axes) > 0:
            avg_axis = np.mean(axes, axis=0)
            return avg_axis / np.linalg.norm(avg_axis)
        return ideal_axis

    n6_marker_actual = extract_axis_from_rotations(poses_6, z_ee_m_ideal)
    n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    
    z_col = n6_marker_actual
    y_col = n5_marker_actual - np.dot(n5_marker_actual, z_col) * z_col
    y_col /= np.linalg.norm(y_col)
    x_col = np.cross(y_col, z_col)
    
    R_m_ee_actual = np.column_stack((x_col, y_col, z_col))
    R_ee_m_actual = R_m_ee_actual.T
    
    euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
    yaw_e2, pitch_e2, roll_e2 = euler_deg
    if arm_side == "right" and yaw_e2 < 0:
        yaw_e2 += 360.0
        
    roll_offset_rec2 = roll_e2 - nominal_rpy[0]
    pitch_offset_rec2 = pitch_e2 - nominal_rpy[1]
    yaw_diff2 = yaw_e2 - nominal_rpy[2]
    if yaw_diff2 > 180.0: yaw_diff2 -= 360.0
    elif yaw_diff2 < -180.0: yaw_diff2 += 360.0
    yaw_offset_rec2 = yaw_diff2

    print(f"[{offset_frame.upper()} frame offset] arm={arm_side}")
    print(f"  Injected:   Roll={roll_offset:+.3f}°, Pitch={pitch_offset:+.3f}°, Yaw={yaw_offset:+.3f}°")
    print(f"  Case A Rec: Roll={roll_offset_rec:+.3f}°, Pitch={pitch_offset_rec:+.3f}°, Yaw={yaw_offset_rec:+.3f}°")
    print(f"  Case B Rec: Roll={roll_offset_rec2:+.3f}°, Pitch={pitch_offset_rec2:+.3f}°, Yaw={yaw_offset_rec2:+.3f}°")

if __name__ == "__main__":
    print("=== TESTING OFFSETS DEFINED IN EE FRAME ===")
    test_offset_definition("left", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, offset_frame="ee")
    test_offset_definition("right", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, offset_frame="ee")
    
    print("\n=== TESTING OFFSETS DEFINED IN MARKER FRAME ===")
    test_offset_definition("left", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, offset_frame="marker")
    test_offset_definition("right", roll_offset=3.0, pitch_offset=-2.0, yaw_offset=1.5, offset_frame="marker")
