import sys
import numpy as np
import json
sys.path.append('/home/rainbow/camera_ws')
import core.calibration.MarkerCalibrator as mc
from scipy.spatial.transform import Rotation as R_scipy

n6_marker_actual = np.array([-0.00180235,  0.99999431, -0.00280755])
n5_marker_actual = np.array([0.17393784, 0.00211537, 0.98475444])
initial_joint_pos_6 = np.radians(-10.0)

z_axis_meas = n6_marker_actual
y_axis_meas = n5_marker_actual
x_axis_meas = np.cross(y_axis_meas, z_axis_meas)
x_axis_meas /= np.linalg.norm(x_axis_meas)
y_axis_meas = np.cross(z_axis_meas, x_axis_meas)
R_meas = np.column_stack((x_axis_meas, y_axis_meas, z_axis_meas))
nominal_vec = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_vec[5], nominal_vec[4], nominal_vec[3]], degrees=True).as_matrix()

R_err_ee = R_ee_m_ideal @ R_meas
euler = R_scipy.from_matrix(R_err_ee).as_euler('XYZ', degrees=True)
extracted_theta = euler[2]
diff_angle = np.radians(extracted_theta) - initial_joint_pos_6
optimal_offset_deg = np.degrees(diff_angle)

print(f"extracted_theta: {extracted_theta}")
print(f"optimal_offset_deg: {optimal_offset_deg}")
