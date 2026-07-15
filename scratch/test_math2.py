import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()

theta_6_phys = -10.0
# True J6 to EE rotation:
R_j6_ee_phys = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()

# Physical J5 axis in Cam frame:
n5_cam = np.array([0, 1.0, 0])

# Wait, if n5_cam is the J5 axis in the camera frame,
# R_ref_5 is the rotation from EE to Camera frame (computed from FK)
# So R_ref_5 = R_cam_to_ee_phys
# Let's say R_cam_to_j6 is Identity.
R_cam_to_j6 = np.eye(3)
R_cam_to_ee_phys = R_cam_to_j6 @ R_j6_ee_phys

# Then n5_marker_actual computed by the code is:
# n5_marker_actual = R_ref_5.T @ n5_cam = R_cam_to_ee_phys.T @ n5_cam
n5_marker_actual = R_cam_to_ee_phys.T @ n5_cam

print("n5_marker_actual:")
print(n5_marker_actual)

y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])

z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle:", diff_angle)

