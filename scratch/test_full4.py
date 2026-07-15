import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

theta_6_enc = -10.0
true_offset = 2.3 # Joint 6 right
theta_6_phys = theta_6_enc + true_offset

R_bracket = R_scipy.from_euler('ZYX', [0.05, -0.1, -0.1], degrees=True).as_matrix() # bracket_rpy: R: -0.1, P: -0.1, Y: 0.05
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix() # for Right arm, it's 180! Wait, NOMINAL_BRACKET_TEMPLATES["1.2"]["right"] = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
# RPY for Right arm 1.2 is 180.0, 0.0, 90.0
# Wait: Z, Y, X order: Yaw=180, Pitch=0, Roll=90
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()
R_ee_m_gt = R_bracket @ R_ee_m_ideal

theta_5_sweep = np.linspace(-20, 20, 50)
positions = []
Rs = []
for th5 in theta_5_sweep:
    R_j6 = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()
    R_j5 = R_scipy.from_euler('y', th5, degrees=True).as_matrix()
    R_cam_to_ee = R_j5 @ R_j6
    T_cam_to_m = R_cam_to_ee @ R_ee_m_gt
    pos_m = R_cam_to_ee @ np.array([0.0005, 0.0, 0.002])
    positions.append(pos_m)
    Rs.append(T_cam_to_m)

positions = np.array(positions)
centroid = np.mean(positions, axis=0)
centered = positions - centroid
U, S, Vt = np.linalg.svd(centered)
n5_cam = Vt[-1]

if np.dot(n5_cam, [0, 1, 0]) < 0:
    n5_cam = -n5_cam

R_ref_5 = Rs[len(Rs)//2]

n5_marker_actual = R_ref_5.T @ n5_cam

y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle (Right Arm):", diff_angle)

