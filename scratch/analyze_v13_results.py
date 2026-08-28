import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

txt_dir = '/home/rainbow/camera_ws/result_1_3/result_txt'

def parse_sweep_file(filepath):
    """
    Parses sweep_points file.
    Lines starting with '#' or '===' are skipped.
    Columns:
    angle, cam_x, cam_y, cam_z, torso_x, torso_y, torso_z, ee_x, ee_y, ee_z, T_cam2marker(16), T_torso2marker(16), T_ee2marker(16)
    """
    if not os.path.exists(filepath):
        return None
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            angle = parts[0]
            cam_xyz = np.array(parts[1:4])
            torso_xyz = np.array(parts[4:7])
            ee_xyz = np.array(parts[7:10])
            T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
            T_torso2marker = np.array(parts[26:42]).reshape((4, 4))
            T_ee2marker = np.array(parts[42:58]).reshape((4, 4))
            data.append({
                'angle': angle,
                'cam_xyz': cam_xyz,
                'torso_xyz': torso_xyz,
                'ee_xyz': ee_xyz,
                'T_cam2marker': T_cam2marker,
                'T_torso2marker': T_torso2marker,
                'T_ee2marker': T_ee2marker
            })
    return data

def fit_circle_3d(pts):
    # Centroid
    c0 = np.mean(pts, axis=0)
    pts_c = pts - c0
    # SVD for plane normal
    u, s, vt = np.linalg.svd(pts_c)
    normal = vt[2]
    # Basis on plane
    u_vec = vt[0]
    v_vec = vt[1]
    # Project to 2D
    u_coords = pts_c @ u_vec
    v_coords = pts_c @ v_vec
    # 2D circle fit: u^2 + v^2 + D*u + E*v + F = 0
    A = np.column_stack([u_coords, v_coords, np.ones_like(u_coords)])
    b = u_coords**2 + v_coords**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc = sol[0] / 2.0
    vc = sol[1] / 2.0
    r = np.sqrt(sol[2] + uc**2 + vc**2)
    center_3d = c0 + uc * u_vec + vc * v_vec
    return center_3d, normal, r

print("=== PARSING SWEEP FILES ===")
for side in ['left', 'right']:
    print(f"\n====================== {side.upper()} ARM ======================")
    # Marker sweep points
    m4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_marker_axis_4.txt'))
    m5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_marker_axis_5.txt'))
    m6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_marker_axis_6.txt'))
    
    # Joint sweep points
    j6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_joint_A_axis_6.txt'))
    j5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_joint_B_axis_5.txt'))
    j4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{side}_joint_B_axis_4.txt'))
    
    for name, data in [('m4', m4), ('m5', m5), ('m6', m6), ('j6', j6), ('j5', j5), ('j4', j4)]:
        if data is not None:
            pts_cam = np.array([d['cam_xyz'] for d in data])
            c, n, r = fit_circle_3d(pts_cam)
            print(f"{name}: pts={len(data)}, r={r:.2f}mm, normal={n.round(4)}, center={c.round(1)}")

    # Check angles between normals
    if j6 is not None and j4 is not None:
        _, n6, _ = fit_circle_3d(np.array([d['cam_xyz'] for d in j6]))
        _, n4, _ = fit_circle_3d(np.array([d['cam_xyz'] for d in j4]))
        ang_64 = np.degrees(np.arccos(np.clip(abs(np.dot(n6, n4)), -1, 1)))
        print(f"Angle between J6 and J4 in camera frame: {ang_64:.4f}° (nominally 90°)")

    if j6 is not None and j5 is not None:
        _, n6, _ = fit_circle_3d(np.array([d['cam_xyz'] for d in j6]))
        _, n5, _ = fit_circle_3d(np.array([d['cam_xyz'] for d in j5]))
        ang_65 = np.degrees(np.arccos(np.clip(abs(np.dot(n6, n5)), -1, 1)))
        print(f"Angle between J6 and J5 in camera frame: {ang_65:.4f}° (nominally 90°)")
