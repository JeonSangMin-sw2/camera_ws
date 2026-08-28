import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

txt_dir = '/home/rainbow/camera_ws/result/result_txt'

def parse_sweep_file(filepath):
    if not os.path.exists(filepath):
        return None
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
            poses.append(T_cam2marker)
    return poses

def fit_circle_3d(pts):
    c0 = np.mean(pts, axis=0)
    pts_c = pts - c0
    u, s, vt = np.linalg.svd(pts_c)
    normal = vt[2]
    u_vec = vt[0]
    v_vec = vt[1]
    u_coords = pts_c @ u_vec
    v_coords = pts_c @ v_vec
    A = np.column_stack([u_coords, v_coords, np.ones_like(u_coords)])
    b = u_coords**2 + v_coords**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc = sol[0] / 2.0
    vc = sol[1] / 2.0
    r = np.sqrt(sol[2] + uc**2 + vc**2)
    center_3d = c0 + uc * u_vec + vc * v_vec
    dist = np.sqrt((u_coords - uc)**2 + (v_coords - vc)**2)
    rmse = np.sqrt(np.mean((dist - r)**2))
    pts_2d = np.column_stack([u_coords, v_coords])
    return center_3d, normal, r, rmse, pts_2d, uc, vc

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
            if np.dot(axis, ideal_axis) < 0:
                axis = -axis
            axes.append(axis)
    if len(axes) == 0:
        return ideal_axis
    axis_mean = np.mean(axes, axis=0)
    return axis_mean / np.linalg.norm(axis_mean)

for arm_side in ['right', 'left']:
    print(f"\n=======================================================")
    print(f"       ANALYZING REAL SWEEP DATA FOR {arm_side.upper()} ARM       ")
    print(f"=======================================================")
    p4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_4.txt'))
    p5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_5.txt'))
    p6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_6.txt'))
    
    print(f"Points: Axis 4 = {len(p4)}, Axis 5 = {len(p5)}, Axis 6 = {len(p6)}")
    
    pts4 = np.array([T[:3, 3]*1000 for T in p4])
    pts5 = np.array([T[:3, 3]*1000 for T in p5])
    pts6 = np.array([T[:3, 3]*1000 for T in p6])
    
    c4, n4_cam, r4, rmse4, _, _, _ = fit_circle_3d(pts4)
    c5, n5_cam, r5, rmse5, _, _, _ = fit_circle_3d(pts5)
    c6, n6_cam, r6, rmse6, _, _, _ = fit_circle_3d(pts6)
    
    print(f"Radii (mm): r4 (Yaw) = {r4:.2f}mm, r5 (Pitch) = {r5:.2f}mm, r6 (Roll) = {r6:.2f}mm")
    print(f"Fit RMSE  : Axis 4 = {rmse4:.3f}mm, Axis 5 = {rmse5:.3f}mm, Axis 6 = {rmse6:.3f}mm")
    
    # Check angles in camera frame:
    dot_45_cam = np.dot(n4_cam, n5_cam)
    ang_45_cam = np.degrees(np.arccos(np.clip(abs(dot_45_cam), -1.0, 1.0)))
    dot_56_cam = np.dot(n5_cam, n6_cam)
    ang_56_cam = np.degrees(np.arccos(np.clip(abs(dot_56_cam), -1.0, 1.0)))
    dot_46_cam = np.dot(n4_cam, n6_cam)
    ang_46_cam = np.degrees(np.arccos(np.clip(abs(dot_46_cam), -1.0, 1.0)))
    
    print(f"\nOrthogonality of Fitted Circle Normals in Camera Frame:")
    print(f"  * J4 - J5 Angle: {ang_45_cam:.3f}° (Deviation from 90°: {abs(ang_45_cam - 90.0):.3f}°)")
    print(f"  * J5 - J6 Angle: {ang_56_cam:.3f}° (Deviation from 90°: {abs(ang_56_cam - 90.0):.3f}°)")
    print(f"  * J4 - J6 Angle: {ang_46_cam:.3f}° (Deviation from 90°: {abs(ang_46_cam - 90.0):.3f}°)")
    
    # Ideal axes
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [-90.0, 0.0, 90.0], degrees=True).as_matrix()
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    
    n6_m = extract_axis_from_rotations(p6, x_ee_m_ideal)
    n5_m = extract_axis_from_rotations(p5, y_ee_m_ideal)
    n4_m = extract_axis_from_rotations(p4, z_ee_m_ideal)
    
    ang_45_m = np.degrees(np.arccos(np.clip(np.dot(n4_m, n5_m), -1.0, 1.0)))
    ang_56_m = np.degrees(np.arccos(np.clip(np.dot(n5_m, n6_m), -1.0, 1.0)))
    ang_46_m = np.degrees(np.arccos(np.clip(np.dot(n4_m, n6_m), -1.0, 1.0)))
    
    print(f"\nOrthogonality of Rotation Axes in Marker Frame:")
    print(f"  * J4 - J5 Angle: {ang_45_m:.3f}° (Deviation from 90°: {abs(ang_45_m - 90.0):.3f}°)")
    print(f"  * J5 - J6 Angle: {ang_56_m:.3f}° (Deviation from 90°: {abs(ang_56_m - 90.0):.3f}°)")
    print(f"  * J4 - J6 Angle: {ang_46_m:.3f}° (Deviation from 90°: {abs(ang_46_m - 90.0):.3f}°)")
