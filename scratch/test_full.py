import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# 1. Physics
theta_6_enc = -10.0
true_offset = 0.0 # because J7 (index 6) has 0.0 offset in mock
theta_6_phys = theta_6_enc + true_offset

# Physical EE rotation in Cam frame:
# R_cam_to_j6 = Identity
R_cam_to_ee_phys = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()

# Ideal Marker rotation in EE frame
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()

# Physical Marker rotation in Cam frame
R_cam_to_m_phys = R_cam_to_ee_phys @ R_ee_m_ideal

# 2. Sweep 5 (J6) creates a circle
# J5 axis in J6 frame is [0,1,0].
# In Cam frame, the axis is:
n5_cam = R_cam_to_ee_phys @ np.array([0, 1.0, 0])

# 3. MarkerCalibrator uses nominal FK to find R_ref_5
R_cam_to_ee_nom = R_scipy.from_euler('z', theta_6_enc, degrees=True).as_matrix()
R_ref_5 = R_cam_to_ee_nom

# 4. Compute n5_marker_actual
n5_marker_actual = R_ref_5.T @ n5_cam

print("n5_marker_actual:", n5_marker_actual)

# 5. JointCalibrator computes diff_angle
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle:", diff_angle)

