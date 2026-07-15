import sys
import numpy as np
import scipy.spatial.transform.Rotation as R_scipy

theta_6_enc = -10.0
true_offset = 2.3
theta_6_phys = theta_6_enc + true_offset

n5_ee = np.array([-np.sin(np.radians(theta_6_phys)), np.cos(np.radians(theta_6_phys)), 0])
R_bracket = R_scipy.Rotation.from_euler('ZYX', [0.05, -0.10, -0.10], degrees=True).as_matrix()
R_ideal = R_scipy.Rotation.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()
R_ee_m_gt = R_bracket @ R_ideal

n5_marker = R_ee_m_gt.T @ n5_ee
z_axis = R_ee_m_gt.T @ np.array([0,0,1])
ref_y = R_ideal.T @ np.array([0,1,0])

n5_proj = n5_marker - np.dot(n5_marker, z_axis)*z_axis
n5_proj /= np.linalg.norm(n5_proj)
ref_x = np.cross(z_axis, ref_y)
ref_x /= np.linalg.norm(ref_x)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))

# Suppose diff_angle is -theta_6_phys = - (-10.0 + 2.3) = 7.7
print("diff_angle:", diff_angle)
print("diff_angle + theta_6_enc:", diff_angle + theta_6_enc)
print("diff_angle - theta_6_enc:", diff_angle - theta_6_enc)
print("-diff_angle - theta_6_enc:", -diff_angle - theta_6_enc)

