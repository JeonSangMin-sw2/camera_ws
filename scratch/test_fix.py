import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Let's say theta_6 is 42.4
theta_6 = 42.4

# J6 rotates by theta around Z in ee frame
R_J6 = R_scipy.from_euler('Z', theta_6, degrees=True).as_matrix()

# Simulated measured axes in ee frame (rotated by theta around Z)
z_ee_meas = R_J6 @ np.array([0, 0, 1])
y_ee_meas = R_J6 @ np.array([0, 1, 0])

# Right Arm ideal R_ee_m_ideal
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180.0, 0.0, 90.0], degrees=True).as_matrix()
R_ideal = R_ee_m_ideal.T

# Simulated measured axes in marker frame
z_axis_meas = R_ideal @ z_ee_meas
y_axis_meas = R_ideal @ y_ee_meas

# New Logic:
x_axis_meas = np.cross(y_axis_meas, z_axis_meas)
x_axis_meas /= np.linalg.norm(x_axis_meas)
y_axis_meas = np.cross(z_axis_meas, x_axis_meas)
R_meas = np.column_stack((x_axis_meas, y_axis_meas, z_axis_meas))

R_err_ee = R_ee_m_ideal @ R_meas
euler = R_scipy.from_matrix(R_err_ee).as_euler('XYZ', degrees=True)
extracted_theta = euler[2]

print("Extracted Z rotation:", extracted_theta)

