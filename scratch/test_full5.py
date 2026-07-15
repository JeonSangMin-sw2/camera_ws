import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

theta_6_enc = -10.0
true_offset = 2.3 
theta_6_phys = theta_6_enc + true_offset

R_bracket = R_scipy.from_euler('ZYX', [0.05, -0.1, -0.1], degrees=True).as_matrix()
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()
R_ee_m_gt = R_bracket @ R_ee_m_ideal

# J7 (axis_6) Sweep
theta_6_sweep = np.linspace(-30, 10, 50)
positions_6 = []
Rs_6 = []
for th6 in theta_6_sweep:
    R_j6 = R_scipy.from_euler('z', th6 + true_offset, degrees=True).as_matrix()
    R_j5 = np.eye(3) # th5 is 0
    R_cam_to_ee = R_j5 @ R_j6
    pos_ee = R_cam_to_ee @ np.array([0,0,0])
    T_cam_to_m = R_cam_to_ee @ R_ee_m_gt
    pos_m = pos_ee + R_cam_to_ee @ np.array([0.0005, 0.0, 0.002])
    positions_6.append(pos_m)
    Rs_6.append(T_cam_to_m)

positions_6 = np.array(positions_6)
U, S, Vt = np.linalg.svd(positions_6 - np.mean(positions_6, axis=0))
n6_cam = Vt[-1]

# R_ref_6 is center
R_ref_6 = Rs_6[len(Rs_6)//2]
n6_marker_actual = R_ref_6.T @ n6_cam

# J5 (axis_5) Sweep
theta_5_sweep = np.linspace(-20, 20, 50)
positions_5 = []
Rs_5 = []
for th5 in theta_5_sweep:
    R_j6 = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()
    R_j5 = R_scipy.from_euler('y', th5, degrees=True).as_matrix()
    R_cam_to_ee = R_j5 @ R_j6
    T_cam_to_m = R_cam_to_ee @ R_ee_m_gt
    pos_m = R_cam_to_ee @ np.array([0.0005, 0.0, 0.002])
    positions_5.append(pos_m)
    Rs_5.append(T_cam_to_m)

positions_5 = np.array(positions_5)
U, S, Vt = np.linalg.svd(positions_5 - np.mean(positions_5, axis=0))
n5_cam = Vt[-1]
if np.dot(n5_cam, [0, 1, 0]) < 0: n5_cam = -n5_cam

R_ref_5 = Rs_5[len(Rs_5)//2]
n5_marker_actual = R_ref_5.T @ n5_cam

# JoinCalibrator logic
z_axis = n6_marker_actual
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle (Right Arm with n6):", diff_angle)


# In MarkerCalibrator:
theta_6_enc = -10.0
y_col_rot = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
y_col_rot /= np.linalg.norm(y_col_rot)

def rodrigues(v, k, theta_deg):
    theta = np.radians(theta_deg)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

# Rotate back by theta_6_enc
y_col = rodrigues(y_col_rot, z_axis, theta_6_enc)

# Then y_col is used to construct R_ee_m_actual...
# But n5_marker_actual is returned AS IS!
# Wait! n5_marker_actual is returned BEFORE applying rodrigues!
# Let me check MarkerCalibrator.py!
