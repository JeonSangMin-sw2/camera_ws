import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append('/home/rainbow/camera_ws')

from core.calibration.MarkerCalibrator import MarkerCalibrator

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

class MockSimRobot:
    def get_dynamics(self):
        class Dyn:
            pass
        return Dyn()
    def get_state(self):
        class State:
            position = [0.0]*32
        return State()
    def model(self):
        class Model:
            left_arm_idx = list(range(7))
            right_arm_idx = list(range(7, 14))
        return Model()

# Create dummy calibrator
mc = MarkerCalibrator(None, MockSimRobot())
mc.robot_version = "1.3"
# override get_link_length and get_z_sign for offline verification
mc.get_link_length = lambda arm: 125.0
mc.get_z_sign = lambda arm: -1.0

p4 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_4.txt'))
p5 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_5.txt'))
p6 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_6.txt'))

print(f"Loaded points: p4={len(p4)}, p5={len(p5)}, p6={len(p6)}")

c4, n4, r4, rmse4, pts4, uc4, vc4 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p4]))
c5, n5, r5, rmse5, pts5, uc5, vc5 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p5]))
c6, n6, r6, rmse6, pts6, uc6, vc6 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p6]))

m4 = {'captured_poses': p4, 'radius': r4, 'rmse': rmse4, 'pts_2d': pts4, 'uc_opt': uc4, 'vc_opt': vc4}
m5 = {'captured_poses': p5, 'radius': r5, 'rmse': rmse5, 'pts_2d': pts5, 'uc_opt': uc5, 'vc_opt': vc5}
m6 = {'captured_poses': p6, 'radius': r6, 'rmse': rmse6, 'pts_2d': pts6, 'uc_opt': uc6, 'vc_opt': vc6}

# Pass 1 computation
res_pass1 = mc.compute_unified_bracket_calibration_v1_3(m5, m6, "right", marker_data_4=m4)
print("\n=== PASS 1 COMPUTATION ===")
print(f"  * Recommended J5 (Pitch): {res_pass1['d5_opt_deg']:.4f}° (GT: -2.10°)")
print(f"  * Recommended J6 (Roll) : {res_pass1['d6_opt_deg']:.4f}° (GT: +2.30°)")
print(f"  * Ortho Residual        : {res_pass1['ortho_err']:.4f}°")

# Pass 2 computation (with same physical data)
res_pass2 = mc.compute_unified_bracket_calibration_v1_3(m5, m6, "right", marker_data_4=m4)
print("\n=== PASS 2 COMPUTATION ===")
print(f"  * Recommended J5 (Pitch): {res_pass2['d5_opt_deg']:.4f}° (GT: -2.10°)")
print(f"  * Recommended J6 (Roll) : {res_pass2['d6_opt_deg']:.4f}° (GT: +2.30°)")
print(f"  * Difference Pass 1->2  : J5: {abs(res_pass2['d5_opt_deg'] - res_pass1['d5_opt_deg']):.4f}°, J6: {abs(res_pass2['d6_opt_deg'] - res_pass1['d6_opt_deg']):.4f}°")
