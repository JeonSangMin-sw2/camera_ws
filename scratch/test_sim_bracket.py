import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def make_transform(xyz_rpy):
    x, y, z, r, p, yaw = xyz_rpy
    R = R_scipy.from_euler('ZYX', [yaw, p, r], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def get_T_base_to_ee(q5, q6, delta_5, delta_6):
    # Command angles plus offsets
    q5_act = np.radians(q5 + delta_5)
    q6_act = np.radians(q6 + delta_6)
    
    # Joint 5 rotates around local Y axis (pitch)
    R5 = R_scipy.from_rotvec([0, q5_act, 0]).as_matrix()
    T5 = np.eye(4)
    T5[:3, :3] = R5
    
    # Joint 6 rotates around local Z axis (yaw/torsion)
    R6 = R_scipy.from_rotvec([0, 0, q6_act]).as_matrix()
    T6 = np.eye(4)
    T6[:3, :3] = R6
    
    return T5 @ T6

def extract_axis_from_rotations(poses, ideal_axis):
    mid_idx = len(poses) // 2
    R_ref = poses[mid_idx][:3, :3]
    axes = []
    for i, T in enumerate(poses):
        if i == mid_idx: continue
        R_rel = R_ref.T @ T[:3, :3] 
        rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle > np.radians(0.1):
            axis = rotvec / angle
            if np.dot(axis, ideal_axis) < 0: axis = -axis
            axes.append(axis)
    if len(axes) > 0:
        avg_axis = np.mean(axes, axis=0)
        return avg_axis / np.linalg.norm(avg_axis)
    return ideal_axis

def rodrigues_rotation(v, k, theta):
    # Rotate vector v around unit vector k by angle theta (radians)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1.0 - cos_t)

def run_calibration_simulation(delta_5, delta_6):
    print(f"\n==================================================")
    print(f" Simulating: Joint 5 Offset = {delta_5}°, Joint 6 Offset = {delta_6}°")
    print(f"==================================================")
    
    # Nominal bracket values (ZYX: Yaw=180, Pitch=0, Roll=90)
    T_ee_to_marker_nominal = make_transform([0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
    R_ee_m_ideal = T_ee_to_marker_nominal[:3, :3]
    
    # Ideal axes in marker frame
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    
    # 1. Simulate Joint 6 Sweep (q6: -20 to +20, q5: 0)
    q6_sweep = np.linspace(-20, 20, 21)
    poses_6 = []
    for q6 in q6_sweep:
        T_ee = get_T_base_to_ee(0.0, q6, delta_5, delta_6)
        T_marker = T_ee @ T_ee_to_marker_nominal
        poses_6.append(T_marker)
        
    # 2. Simulate Joint 5 Sweep (q5: -10 to +10, q6: 0)
    q5_sweep = np.linspace(-10, 10, 21)
    poses_5 = []
    for q5 in q5_sweep:
        T_ee = get_T_base_to_ee(q5, 0.0, delta_5, delta_6)
        T_marker = T_ee @ T_ee_to_marker_nominal
        poses_5.append(T_marker)
        
    # Extract actual rotation axes in marker frame
    n6_actual = extract_axis_from_rotations(poses_6, z_ee_m_ideal)
    n5_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    
    # We command Joint 6 to be 0 during the Joint 5 sweep.
    # The actual Joint 6 angle during the Joint 5 sweep is delta_6!
    theta_6_actual = np.radians(delta_6)
    
    # Apply Joint 6 angle correction to the fitted Joint 5 axis
    # (Since Joint 6 rotated by delta_6, we rotate n5_actual back around n6_actual by delta_6)
    z_col = n6_actual
    y_col_rotated = n5_actual - np.dot(n5_actual, z_col) * z_col
    y_col_rotated /= np.linalg.norm(y_col_rotated)
    
    if abs(theta_6_actual) > 1e-5:
        y_col = rodrigues_rotation(y_col_rotated, z_col, -theta_6_actual) # Rotate back by delta_6
    else:
        y_col = y_col_rotated
        
    x_col = np.cross(y_col, z_col)
    x_col /= np.linalg.norm(x_col)
    
    # Gram-Schmidt Orthogonal Matrix
    R_m_ee = np.column_stack((x_col, y_col, z_col))
    R_ee_m_calib = R_m_ee.T
    
    # Extract Euler angles (ZYX)
    euler = R_scipy.from_matrix(R_ee_m_calib).as_euler('ZYX', degrees=True)
    print(f"Calibrated Bracket Rotation Angles (ZYX):")
    print(f"  Yaw (Z):   {euler[0]:.4f}° (Ideal: 180.0000°)")
    print(f"  Pitch (Y): {euler[1]:.4f}° (Ideal: 0.0000°)")
    print(f"  Roll (X):  {euler[2]:.4f}° (Ideal: 90.0000°)")
    
    # Also try without the delta_6 correction to see what happens!
    y_col_uncorrected = y_col_rotated
    x_col_uncorrected = np.cross(y_col_uncorrected, z_col)
    x_col_uncorrected /= np.linalg.norm(x_col_uncorrected)
    R_m_ee_uncorrected = np.column_stack((x_col_uncorrected, y_col_uncorrected, z_col))
    R_ee_m_uncorrected = R_m_ee_uncorrected.T
    euler_unc = R_scipy.from_matrix(R_ee_m_uncorrected).as_euler('ZYX', degrees=True)
    print(f"Without Joint 6 correction:")
    print(f"  Yaw (Z):   {euler_unc[0]:.4f}°")
    print(f"  Pitch (Y): {euler_unc[1]:.4f}°")
    print(f"  Roll (X):  {euler_unc[2]:.4f}°")

if __name__ == "__main__":
    # Case 1: Joint offsets are 0
    run_calibration_simulation(0.0, 0.0)
    
    # Case 2: Joint offsets are 0.5 and 0.3
    run_calibration_simulation(0.5, 0.3)
