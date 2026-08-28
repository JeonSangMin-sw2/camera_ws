import os
import sys
import numpy as np

sys.path.insert(0, '/home/rainbow/camera_ws')

from core.calibration.MarkerCalibrator import MarkerCalibrator
from core.calibration.CalibratorBase import BaseCalibrator
from scipy.spatial.transform import Rotation as R_scipy

txt_dir = '/home/rainbow/camera_ws/result_1_3/result_txt'

def parse_sweep_file(filepath):
    if not os.path.exists(filepath):
        return None
    poses = []
    q_fulls = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            angle = parts[0]
            T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
            poses.append(T_cam2marker)
            q_fulls.append([0.0]*16)
    return poses, q_fulls

mc = MarkerCalibrator(marker_st=None, robot=None)
mc.robot_version = "1.3"

for arm in ['left', 'right']:
    print(f"\n==================== {arm.upper()} ARM SIMULTANEOUS CALIBRATION ====================")
    poses_4, q_4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_4.txt'))
    poses_5, q_5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_5.txt'))
    poses_6, q_6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm}_marker_axis_6.txt'))
    
    def get_marker_data(poses, joint_idx):
        pts = np.array([T[:3, 3] * 1000.0 for T in poses])
        c, R_fit, r, rmse, _, _, _ = BaseCalibrator.fit_circle_3d(pts)
        n = R_fit[:, 2]
        return {
            'captured_poses': poses,
            'captured_q_full': [[0.0]*16]*len(poses),
            'axis_opt': n,
            'radius': r,
            'c_opt': c,
            'rmse': rmse
        }
        
    m4_data = get_marker_data(poses_4, 4)
    m5_data = get_marker_data(poses_5, 5)
    m6_data = get_marker_data(poses_6, 6)
    
    print(f"Sweep Radii: R4={m4_data['radius']:.2f}mm, R5={m5_data['radius']:.2f}mm, R6={m6_data['radius']:.2f}mm")
    
    # We test the QP / SVD formulation
    # Let's inspect axes extracted from rotations
    ver_key = "1.3"
    nominal_rpy = mc.NOMINAL_BRACKET_TEMPLATES[ver_key][arm][3:6]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
    
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

    n6_m = extract_axis_from_rotations(poses_6, x_ee_m_ideal)
    n5_m = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    n4_m = extract_axis_from_rotations(poses_4, z_ee_m_ideal)
    
    print(f"Extracted axes in marker frame:")
    print(f"  * n6 (J6 Roll axis, ideal [1,0,0] in EE): {n6_m.round(4)}")
    print(f"  * n5 (J5 Pitch axis, ideal [0,1,0] in EE): {n5_m.round(4)}")
    print(f"  * n4 (J4 Yaw axis, ideal [0,0,1] in EE): {n4_m.round(4)}")
    
    ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n5_m)), -1, 1)))
    ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(n5_m, n6_m)), -1, 1)))
    ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_m, n6_m)), -1, 1)))
    print(f"  * Orthogonality check: J4-J5={ang_45:.3f}°, J5-J6={ang_56:.3f}°, J4-J6={ang_46:.3f}°")
    
    # 3-axis SVD alignment to build R_m_ee:
    # J6 is X, J5 is Y, J4 is Z
    M = np.column_stack((n6_m, n5_m, n4_m))
    U, S, Vt = np.linalg.svd(M)
    R_m_ee = U @ Vt
    if np.linalg.det(R_m_ee) < 0:
        U[:, 2] *= -1
        R_m_ee = U @ Vt
    R_ee_m = R_m_ee.T
    
    # Orientation of bracket:
    euler_deg = R_scipy.from_matrix(R_ee_m).as_euler('ZYX', degrees=True)
    rot_err_mat = R_ee_m.T @ R_ee_m_ideal
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
    print(f"\n[3-Axis SVD Bracket Orientation]")
    print(f"  * RPY in EE: Roll={euler_deg[2]:.3f}°, Pitch={euler_deg[1]:.3f}°, Yaw={euler_deg[0]:.3f}°")
    print(f"  * Ideal RPY: {nominal_rpy}")
    print(f"  * Rotation Error vs Ideal: {rot_err_deg:.3f}°")
