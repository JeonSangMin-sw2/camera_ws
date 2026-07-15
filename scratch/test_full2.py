import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

theta_6_enc = -10.0
true_offset = 0.0 # because J7 has 0.0 offset in mock
theta_6_phys = theta_6_enc + true_offset

R_cam_to_ee_phys = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()

# Simulated Bracket Offset (bracket_rpy = [-0.1, -0.1, 0.05])
R_bracket = R_scipy.from_euler('ZYX', [0.05, -0.1, -0.1], degrees=True).as_matrix()
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()

# R_ee_m_gt includes bracket offset
R_ee_m_gt = R_bracket @ R_ee_m_ideal

R_cam_to_m_phys = R_cam_to_ee_phys @ R_ee_m_gt

# J5 axis in cam frame
n5_cam = R_cam_to_ee_phys @ np.array([0, 1.0, 0])

# R_ref_5 is the measured marker orientation at the center of the sweep
R_ref_5 = R_cam_to_m_phys

# n5_marker_actual
n5_marker_actual = R_ref_5.T @ n5_cam

# JoinCalibrator logic
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle:", diff_angle)

