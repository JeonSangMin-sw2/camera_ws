import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

theta_6 = 2.3
R_J6 = R_scipy.from_euler('Z', theta_6, degrees=True).as_matrix()
z_ee_meas = R_J6 @ np.array([0, 0, 1])
y_ee_meas = R_J6 @ np.array([0, 1, 0])

R_ee_m_ideal = R_scipy.from_euler('ZYX', [180.0, 0.0, 90.0], degrees=True).as_matrix()
R_ideal = R_ee_m_ideal.T

z_axis_meas = R_ideal @ z_ee_meas
y_axis_meas = R_ideal @ y_ee_meas

# Old Logic
z_axis = z_axis_meas
n5_act = y_axis_meas
ref_y = R_ideal @ np.array([0, 1, 0])
n5_proj = n5_act - np.dot(n5_act, z_axis) * z_axis
ref_x = np.cross(z_axis, ref_y)
ref_x /= np.linalg.norm(ref_x)
diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("Old logic diff angle:", diff_angle)

# New Logic
x_axis_meas = np.cross(y_axis_meas, z_axis_meas)
y_axis_meas = np.cross(z_axis_meas, x_axis_meas)
R_meas = np.column_stack((x_axis_meas, y_axis_meas, z_axis_meas))
R_err_ee = R_ee_m_ideal @ R_meas
euler = R_scipy.from_matrix(R_err_ee).as_euler('XYZ', degrees=True)
print("New logic extracted Z rotation:", euler[2])
