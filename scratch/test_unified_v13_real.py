import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

txt_dir = '/home/rainbow/camera_ws/result_1_3/result_txt'

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
    # residuals
    pts_2d = np.column_stack([u_coords, v_coords])
    dist = np.sqrt((u_coords - uc)**2 + (v_coords - vc)**2)
    rmse = np.sqrt(np.mean((dist - r)**2))
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
            if np.dot(axis, ideal_axis) < 0: axis = -axis
            axes.append(axis)
    if len(axes) > 0:
        avg_axis = np.mean(axes, axis=0)
        return avg_axis / np.linalg.norm(avg_axis)
    return ideal_axis

def compute_v13_unified(arm_side, poses_4, poses_5, poses_6, current_offsets={'joint5': 0.0, 'joint6': 0.0}):
    # Nominal bracket template
    nominal_vec = [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
    x_nom, y_nom, z_nom = nominal_vec[:3]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_vec[5], nominal_vec[4], nominal_vec[3]], degrees=True).as_matrix()
    
    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    
    # 1. Fit circles and extract axes in marker frame
    c4, n4_cam, r4, rmse4, _, _, _ = fit_circle_3d(np.array([T[:3, 3]*1000 for T in poses_4]))
    c5, n5_cam, r5, rmse5, _, _, _ = fit_circle_3d(np.array([T[:3, 3]*1000 for T in poses_5]))
    c6, n6_cam, r6, rmse6, _, _, _ = fit_circle_3d(np.array([T[:3, 3]*1000 for T in poses_6]))
    
    n6_m = extract_axis_from_rotations(poses_6, x_ee_m_ideal)
    n5_m = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    n4_m = extract_axis_from_rotations(poses_4, z_ee_m_ideal)
    
    # Check orthogonality
    ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n5_m)), -1, 1)))
    ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(n5_m, n6_m)), -1, 1)))
    ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n6_m)), -1, 1)))
    
    # Gram-Schmidt / SVD to build ideal orthogonal basis R_m_ee:
    # J6 is X-axis (Roll)
    x_col = n6_m / np.linalg.norm(n6_m)
    # Project J5 onto plane perpendicular to X
    y_proj = n5_m - np.dot(n5_m, x_col) * x_col
    y_col = y_proj / np.linalg.norm(y_proj)
    # Z axis is X x Y
    z_col = np.cross(x_col, y_col)
    z_col /= np.linalg.norm(z_col)
    
    # Build rotation matrix from marker to EE frame:
    R_m_ee = np.column_stack((x_col, y_col, z_col))
    R_ee_m = R_m_ee.T
    
    euler_deg = R_scipy.from_matrix(R_ee_m).as_euler('ZYX', degrees=True)
    yaw_e, pitch_e, roll_e = euler_deg
    if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
        yaw_e += 360.0

    # Solve for Joint 6 offset:
    # y_col is the projected J5 axis in marker frame.
    # In ideal marker frame, J5 axis is y_ee_m_ideal.
    # The angle between y_ee_m_ideal and y_col around x_col is the Joint 6 offset delta!
    ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
    ref_y /= np.linalg.norm(ref_y)
    ref_z = np.cross(x_col, ref_y)
    
    roll_err_rad = np.arctan2(np.dot(y_col, ref_z), np.dot(y_col, ref_y))
    opt_d6_deg = -np.degrees(roll_err_rad) + current_offsets.get('joint6', 0.0)
    
    # Solve for Joint 5 offset:
    # ang_46 is the angle between J4 (Yaw) and J6 (Roll).
    # Nominally 90.0°. Deviation from 90° is directly the Pitch angle error!
    pitch_err_deg = (ang_46 - 90.0)
    opt_d5_deg = -pitch_err_deg + current_offsets.get('joint5', 0.0)
    
    # Solve for translation (x_e, y_e, z_e) in mm:
    L_nom = 125.0 # mm
    def res_trans(p):
        xe, ye, ze = p
        Zs = ze - L_nom
        r6_p = np.sqrt(ye**2 + Zs**2)
        r5_p = np.sqrt(xe**2 + Zs**2)
        r4_p = np.sqrt(xe**2 + ye**2)
        return [r6_p - r6, r5_p - r5, r4_p - r4]
        
    x0 = [x_nom*1000.0, y_nom*1000.0, z_nom*1000.0]
    opt_t = least_squares(res_trans, x0, bounds=([20, -30, -50], [120, 30, 50]))
    xe_opt, ye_opt, ze_opt = opt_t.x
    
    # Verification metrics
    ortho_err = max(abs(ang_45 - 90.0), abs(ang_56 - 90.0), abs(ang_46 - 90.0))
    r_res = res_trans(opt_t.x)
    
    return {
        'arm_side': arm_side,
        'd5_opt_deg': opt_d5_deg,
        'd6_opt_deg': opt_d6_deg,
        'x_e': xe_opt, 'y_e': ye_opt, 'z_e': ze_opt,
        'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
        'ortho_err': ortho_err,
        'ang_45': ang_45, 'ang_56': ang_56, 'ang_46': ang_46,
        'r4': r4, 'r5': r5, 'r6': r6,
        'r_res': r_res,
        'rmse4': rmse4, 'rmse5': rmse5, 'rmse6': rmse6
    }

for arm in ['left', 'right']:
    p4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_4.txt'))
    p5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_5.txt'))
    p6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_6.txt'))
    res = compute_v13_unified(arm, p4, p5, p6)
    print(f"\n==================== {arm.upper()} ARM UNIFIED CALIBRATION RESULT ====================")
    print(f"  * Recommended Joint 5 (Pitch) Offset : {res['d5_opt_deg']:.4f}°")
    print(f"  * Recommended Joint 6 (Roll)  Offset : {res['d6_opt_deg']:.4f}°")
    print(f"  * Marker Bracket Position (x, y, z)  : [{res['x_e']:.2f}, {res['y_e']:.2f}, {res['z_e']:.2f}] mm (Nominal: [67.0, 0.0, 0.0])")
    print(f"  * Marker Bracket Orientation (R,P,Y) : [{res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]° (Nominal: [90.0, 0.0, -90.0])")
    print(f"  * Orthogonality Check (Target 90.0°):")
    print(f"      - J4(Yaw) - J5(Pitch) : {res['ang_45']:.3f}° (Dev: {abs(res['ang_45']-90.0):.3f}°)")
    print(f"      - J5(Pitch) - J6(Roll): {res['ang_56']:.3f}° (Dev: {abs(res['ang_56']-90.0):.3f}°)")
    print(f"      - J4(Yaw) - J6(Roll)  : {res['ang_46']:.3f}° (Dev: {abs(res['ang_46']-90.0):.3f}°)")
    print(f"  * Radii Residuals                    : {[f'{x:.3f}mm' for x in res['r_res']]}")
