import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def run_simulation_comparison(arm_side="left", noise_rot_deg=0.2, noise_trans_mm=1.5):
    # Injected offsets
    roll_offset = 3.0
    pitch_offset = -2.0
    yaw_offset = 1.5
    
    if arm_side == "left":
        nominal_rpy = [90.0, 0.0, 0.0]
    else:
        nominal_rpy = [90.0, 0.0, 180.0]

    # Ground truth bracket rotation
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
    R_offset = R_scipy.from_euler('ZYX', [yaw_offset, pitch_offset, roll_offset], degrees=True).as_matrix()
    R_ee_m_actual_gt = R_offset @ R_ee_m_ideal

    # Camera rotation in robot torso frame
    R_c_ee_mid = R_scipy.from_euler('ZYX', [10.0, -5.0, 15.0], degrees=True).as_matrix()
    
    # 1. Joint 6 Sweep (Roll)
    theta_6_list = np.linspace(-20.0, 20.0, 30)
    poses_6 = []
    R_ee_list_6 = []
    for theta in theta_6_list:
        R_ee_rot = R_scipy.from_euler('Z', theta, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        
        # Add noise to camera measurement
        noise_euler = np.random.normal(0, noise_rot_deg, 3)
        R_noise = R_scipy.from_euler('ZYX', noise_euler, degrees=True).as_matrix()
        R_c_m_noisy = R_noise @ R_c_m
        
        T = np.eye(4)
        T[:3, :3] = R_c_m_noisy
        T[:3, 3] = [0.1, 0.2, 0.5] + np.random.normal(0, noise_trans_mm/1000.0, 3)
        poses_6.append(T)
        R_ee_list_6.append(R_ee_rot)

    # 2. Joint 5 Sweep (Pitch)
    theta_5_list = np.linspace(-10.0, 10.0, 30)
    poses_5 = []
    R_ee_list_5 = []
    for theta in theta_5_list:
        R_ee_rot = R_scipy.from_euler('Y', theta, degrees=True).as_matrix()
        R_c_m = R_c_ee_mid @ R_ee_rot @ R_ee_m_actual_gt
        
        # Add noise to camera measurement
        noise_euler = np.random.normal(0, noise_rot_deg, 3)
        R_noise = R_scipy.from_euler('ZYX', noise_euler, degrees=True).as_matrix()
        R_c_m_noisy = R_noise @ R_c_m
        
        T = np.eye(4)
        T[:3, :3] = R_c_m_noisy
        T[:3, 3] = [0.1, 0.2, 0.5] + np.random.normal(0, noise_trans_mm/1000.0, 3)
        poses_5.append(T)
        R_ee_list_5.append(R_ee_rot)

    # --- METHOD 1: Axis Fitting (SVD / Gram-Schmidt) ---
    def extract_axis(poses, ideal_axis):
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
        return np.mean(axes, axis=0) if len(axes) > 0 else ideal_axis

    # Extract axes in marker frame
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    
    n6 = extract_axis(poses_6, z_ee_m_ideal)
    n6 /= np.linalg.norm(n6)
    
    n5 = extract_axis(poses_5, y_ee_m_ideal)
    n5 /= np.linalg.norm(n5)
    
    # Gram-Schmidt
    z_col = n6
    y_col = n5 - np.dot(n5, z_col) * z_col
    y_col /= np.linalg.norm(y_col)
    x_col = np.cross(y_col, z_col)
    R_ee_m_fit = np.column_stack((x_col, y_col, z_col)).T
    
    # Extract Euler angles from Method 1
    # ZYX Euler angle output is [yaw, pitch, roll]
    yaw_e_fit, pitch_e_fit, roll_e_fit = R_scipy.from_matrix(R_ee_m_fit).as_euler('ZYX', degrees=True)
    roll_offset_fit = roll_e_fit - nominal_rpy[0]
    pitch_offset_fit = pitch_e_fit - nominal_rpy[1]
    yaw_diff_fit = yaw_e_fit - nominal_rpy[2]
    if yaw_diff_fit > 180.0: yaw_diff_fit -= 360.0
    elif yaw_diff_fit < -180.0: yaw_diff_fit += 360.0
    yaw_offset_fit = yaw_diff_fit

    # --- METHOD 2: Direct Kinematic Averaging ---
    R_list = []
    for R_ee, T in zip(R_ee_list_6, poses_6):
        R_c_m = T[:3, :3]
        R_ee_m_i = R_ee.T @ R_c_ee_mid.T @ R_c_m
        R_list.append(R_ee_m_i)
    for R_ee, T in zip(R_ee_list_5, poses_5):
        R_c_m = T[:3, :3]
        R_ee_m_i = R_ee.T @ R_c_ee_mid.T @ R_c_m
        R_list.append(R_ee_m_i)
        
    M = np.mean(R_list, axis=0)
    U, S, Vt = np.linalg.svd(M)
    R_ee_m_avg = U @ Vt
    if np.linalg.det(R_ee_m_avg) < 0:
        U[:, 2] *= -1
        R_ee_m_avg = U @ Vt
        
    yaw_e_avg, pitch_e_avg, roll_e_avg = R_scipy.from_matrix(R_ee_m_avg).as_euler('ZYX', degrees=True)
    roll_offset_avg = roll_e_avg - nominal_rpy[0]
    pitch_offset_avg = pitch_e_avg - nominal_rpy[1]
    yaw_diff_avg = yaw_e_avg - nominal_rpy[2]
    if yaw_diff_avg > 180.0: yaw_diff_avg -= 360.0
    elif yaw_diff_avg < -180.0: yaw_diff_avg += 360.0
    yaw_offset_avg = yaw_diff_avg

    print(f"--- Left Arm Simulation (Noise: Rot={noise_rot_deg}°, Trans={noise_trans_mm}mm) ---")
    print(f"Injected Offset      : Roll={roll_offset:+.3f}°, Pitch={pitch_offset:+.3f}°, Yaw={yaw_offset:+.3f}°")
    print(f"Method 1 (Axis Fit)  : Roll={roll_offset_fit:+.3f}°, Pitch={pitch_offset_fit:+.3f}°, Yaw={yaw_offset_fit:+.3f}°")
    print(f"Method 2 (Kinem Avg) : Roll={roll_offset_avg:+.3f}°, Pitch={pitch_offset_avg:+.3f}°, Yaw={yaw_offset_avg:+.3f}°")
    print(f"Method 1 Error L2    : {np.linalg.norm(R_ee_m_fit - R_ee_m_actual_gt):.4e}")
    print(f"Method 2 Error L2    : {np.linalg.norm(R_ee_m_avg - R_ee_m_actual_gt):.4e}")

if __name__ == "__main__":
    np.random.seed(42)
    run_simulation_comparison()
