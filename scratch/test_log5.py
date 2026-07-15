import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

n6_marker_actual = np.array([-0.00180235,  0.99999431, -0.00280755])
n5_marker_actual = np.array([0.17393784, 0.00211537, 0.98475444])

z_meas = n6_marker_actual
y_meas = n5_marker_actual
x_meas = np.cross(y_meas, z_meas)
x_meas /= np.linalg.norm(x_meas)
y_meas = np.cross(z_meas, x_meas)
R_meas = np.column_stack((x_meas, y_meas, z_meas))

nominal_vec = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
R_ideal = R_scipy.from_euler('ZYX', [nominal_vec[5], nominal_vec[4], nominal_vec[3]], degrees=True).as_matrix()

R_err_1 = R_ideal @ R_meas
euler_1 = R_scipy.from_matrix(R_err_1).as_euler('XYZ', degrees=True)

R_err_2 = R_meas @ R_ideal.T
euler_2 = R_scipy.from_matrix(R_err_2).as_euler('XYZ', degrees=True)

R_err_3 = R_ideal.T @ R_meas
euler_3 = R_scipy.from_matrix(R_err_3).as_euler('XYZ', degrees=True)

print(f"euler_1 (R_ideal @ R_meas): {euler_1}")
print(f"euler_2 (R_meas @ R_ideal.T): {euler_2}")
print(f"euler_3 (R_ideal.T @ R_meas): {euler_3}")
