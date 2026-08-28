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
            if np.dot(axis, ideal_axis) < 0: axis = -axis
            axes.append(axis)
    if len(axes) > 0:
        avg_axis = np.mean(axes, axis=0)
        return avg_axis / np.linalg.norm(avg_axis)
    return ideal_axis

def compute_unified_bracket_v13_polished(marker_data_5, marker_data_6, arm_side, marker_data_4=None, calib_pitch_deg=None, calib_roll_deg=None):
    nominal_vec = [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
    x_nom, y_nom, z_nom = nominal_vec[:3]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_vec[5], nominal_vec[4], nominal_vec[3]], degrees=True).as_matrix()
    
    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    
    poses_6 = marker_data_6.get('captured_poses', [])
    poses_5 = marker_data_5.get('captured_poses', [])
    poses_4 = marker_data_4.get('captured_poses', []) if marker_data_4 else []
    
    n6_m = extract_axis_from_rotations(poses_6, x_ee_m_ideal)
    n5_m = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    n4_m = extract_axis_from_rotations(poses_4, z_ee_m_ideal) if poses_4 else z_ee_m_ideal
    
    radius_6 = marker_data_6.get('radius', 0.0)
    radius_5 = marker_data_5.get('radius', 0.0)
    radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 else 0.0
    
    # Orthogonality checks
    ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n5_m)), -1.0, 1.0)))
    ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(n5_m, n6_m)), -1.0, 1.0)))
    ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n6_m)), -1.0, 1.0)))
    ortho_err = max(abs(ang_45 - 90.0), abs(ang_56 - 90.0), abs(ang_46 - 90.0))
    
    # Decoupled SVD basis construction
    x_col = n6_m / np.linalg.norm(n6_m)
    y_proj = n5_m - np.dot(n5_m, x_col) * x_col
    y_col = y_proj / np.linalg.norm(y_proj)
    z_col = np.cross(x_col, y_col)
    z_col /= np.linalg.norm(z_col)
    
    M = np.column_stack((x_col, y_col, z_col))
    U, S, Vt = np.linalg.svd(M)
    R_m_ee = U @ Vt
    if np.linalg.det(R_m_ee) < 0:
        U[:, 2] *= -1
        R_m_ee = U @ Vt
    R_ee_m = R_m_ee.T
    
    euler_deg = R_scipy.from_matrix(R_ee_m).as_euler('ZYX', degrees=True)
    yaw_e, pitch_e, roll_e = euler_deg
    if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
        yaw_e += 360.0
        
    rot_err_mat = R_ee_m.T @ R_ee_m_ideal
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
    
    # Joint offsets
    ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
    ref_y /= np.linalg.norm(ref_y)
    ref_z = np.cross(x_col, ref_y)
    diff_angle_6 = np.arctan2(np.dot(n5_m, ref_z), np.dot(n5_m, ref_y))
    opt_delta_6 = -float(np.degrees(diff_angle_6))
    if calib_roll_deg is not None:
        opt_delta_6 += calib_roll_deg
        
    cross_64 = np.cross(n6_m, n4_m)
    sign_5 = np.sign(np.dot(n5_m, cross_64)) if np.linalg.norm(cross_64) > 1e-4 else 1.0
    opt_delta_5 = -float((ang_46 - 90.0) * sign_5)
    if calib_pitch_deg is not None:
        opt_delta_5 += calib_pitch_deg
        
    # Translation solver (XYZ)
    L_5_ee = 125.0 # mm
    has_j4 = (marker_data_4 is not None and radius_4 > 1.0)
    
    def residuals_trans(params):
        xe, ye, ze = params
        Z_prime = ze - L_5_ee
        r6_pred = np.sqrt(ye**2 + Z_prime**2)
        r5_pred = np.sqrt(xe**2 + Z_prime**2)
        r4_pred = np.sqrt(xe**2 + ye**2)
        
        res = [
            (r6_pred - radius_6),
            (r5_pred - radius_5)
        ]
        if has_j4:
            res.append(r4_pred - radius_4)
        reg = 1e-3
        res.append(reg * (xe - x_nom * 1000.0))
        res.append(reg * (ye - y_nom * 1000.0))
        res.append(reg * (ze - z_nom * 1000.0))
        return res

    x_init = [x_nom * 1000.0, y_nom * 1000.0, z_nom * 1000.0]
    opt_res = least_squares(residuals_trans, x_init, bounds=([20.0, -30.0, -50.0], [120.0, 30.0, 50.0]), loss='huber')
    xe_opt, ye_opt, ze_opt = opt_res.x
    
    Z_prime_opt = ze_opt - L_5_ee
    r6_err = abs(radius_6 - np.sqrt(ye_opt**2 + Z_prime_opt**2))
    r5_err = abs(radius_5 - np.sqrt(xe_opt**2 + Z_prime_opt**2))
    r4_err = abs(radius_4 - np.sqrt(xe_opt**2 + ye_opt**2)) if has_j4 else 0.0
    max_radius_err = max(r6_err, r5_err, r4_err)
    
    pos_diff_mm = float(np.linalg.norm([xe_opt - x_nom*1000.0, ye_opt - y_nom*1000.0, ze_opt - z_nom*1000.0]))
    
    return {
        'converged': True,
        'x_e': xe_opt, 'y_e': ye_opt, 'z_e': ze_opt,
        'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
        'opt_delta_5': opt_delta_5,
        'opt_delta_6': opt_delta_6,
        'd5_opt_deg': opt_delta_5,
        'd6_opt_deg': opt_delta_6,
        'recommended_joint_offset_5': opt_delta_5,
        'recommended_joint_offset_6': opt_delta_6,
        'L_5_ee': L_5_ee,
        'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
        'ortho_err': ortho_err,
        'ang_45': ang_45, 'ang_56': ang_56, 'ang_46': ang_46,
        'r6_err': r6_err, 'r5_err': r5_err, 'r4_err': r4_err,
        'max_radius_err': max_radius_err,
        'rot_err_deg': rot_err_deg,
        'pos_diff_mm': pos_diff_mm,
        'warn_large_angle': rot_err_deg > 15.0,
        'warn_large_pos': pos_diff_mm > 40.0,
        'rmse_6': marker_data_6.get('rmse', 0.0),
        'rmse_5': marker_data_5.get('rmse', 0.0),
        'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 else 0.0,
        'n6_marker_actual': n6_m,
        'n5_marker_actual': n5_m,
        'n4_marker_actual': n4_m,
        'y_ee_m_ideal': y_ee_m_ideal
    }

for arm in ['left', 'right']:
    p4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_4.txt'))
    p5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_5.txt'))
    p6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_6.txt'))
    
    m4 = {'captured_poses': p4, 'radius': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p4]))[2], 'rmse': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p4]))[3]}
    m5 = {'captured_poses': p5, 'radius': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p5]))[2], 'rmse': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p5]))[3]}
    m6 = {'captured_poses': p6, 'radius': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p6]))[2], 'rmse': fit_circle_3d(np.array([T[:3, 3]*1000 for T in p6]))[3]}
    
    out = compute_unified_bracket_v13_polished(m5, m6, arm, marker_data_4=m4)
    print(f"=== {arm.upper()} ARM POLISHED TEST ===")
    print(f"  * Opt Delta 5 (Pitch): {out['opt_delta_5']:.4f}°")
    print(f"  * Opt Delta 6 (Roll) : {out['opt_delta_6']:.4f}°")
    print(f"  * Bracket XYZ (mm)   : [{out['x_e']:.2f}, {out['y_e']:.2f}, {out['z_e']:.2f}]")
    print(f"  * Bracket RPY (deg)  : [{out['roll_e']:.2f}, {out['pitch_e']:.2f}, {out['yaw_e']:.2f}]")
    print(f"  * Ortho Error        : {out['ortho_err']:.4f}°")
    print(f"  * Max Radius Error   : {out['max_radius_err']:.4f} mm")
