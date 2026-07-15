import numpy as np

# R.T for right arm v1.2
R_T = np.array([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

z_axis = R_T @ np.array([0, 0, 1])  # n6_marker_actual = [0, 1, 0]
ref_y = R_T @ np.array([0, 1, 0])   # y_ee_m_ideal = [0, 0, 1]
ref_x = np.cross(z_axis, ref_y)     # [1, 0, 0]

# Suppose n5_ee has a 45 degree offset and 2.3 offset
angle_ee = np.radians(-45 + 2.3)
n5_ee = np.array([-np.sin(angle_ee), np.cos(angle_ee), 0])
n5_act = R_T @ n5_ee

n5_proj = n5_act - np.dot(n5_act, z_axis) * z_axis
diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))

print("Diff Angle:", diff_angle)

