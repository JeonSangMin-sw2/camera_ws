import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Ideal marker orientation in EE frame (assume identity for simplicity, or 90 deg rotation)
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()

# 1. Physics: The robot is at J7 = -10.0 (ready pose). No physical offset.
theta_6_phys = -10.0
# The physical EE frame is rotated around Z by -10 degrees relative to J6
R_j6_ee = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()

# J5 rotates around Y axis of J6. In EE frame, J5 axis is R_j6_ee.T @ [0,1,0]
# Wait, if J5 rotates around Y, then J5 axis in J6 frame is [0,1,0].
n5_j6 = np.array([0, 1.0, 0])
n5_ee = R_j6_ee.T @ n5_j6 # J5 axis in EE frame

# In Marker frame, n5_marker = R_ee_m_ideal.T @ n5_ee
n5_marker = R_ee_m_ideal.T @ n5_ee

print("n5_marker (Physical J5 axis in Marker Frame):")
print(n5_marker)

# 2. Calibration Algorithm
# The algorithm measures n5_marker. Then it compares it to y_ee_m_ideal.
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
print("y_ee_m_ideal (Ideal J5 axis in Marker Frame):")
print(y_ee_m_ideal)

# difference angle between n5_marker and y_ee_m_ideal
z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker - np.dot(n5_marker, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle:", diff_angle)
