import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def load_poses(filename):
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                vals = list(map(float, line.strip().split(',')))
                # T_cam2marker_flat starts at index 10 (0-indexed)
                # Let's verify: 
                # vals[0]: J_deg
                # vals[1:4]: Cam_XYZ
                # vals[4:7]: Torso_XYZ
                # vals[7:10]: EE_XYZ
                # vals[10:26]: T_cam2marker_flat
                T = np.array(vals[10:26]).reshape(4, 4)
                # Some matrices might be row-major or col-major, let's assume row-major
                poses.append(T)
    return poses

poses_6 = load_poses('/home/rainbow/camera_ws/core/calibration/result_txt/sweep_points_right_marker_axis_6.txt')
poses_5 = load_poses('/home/rainbow/camera_ws/core/calibration/result_txt/sweep_points_right_marker_axis_5.txt')

def fit_circle(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    return Vt[2, :]

c_6 = np.array([T[:3, 3] for T in poses_6])
c_5 = np.array([T[:3, 3] for T in poses_5])

n6_cam = fit_circle(c_6)
n5_cam = fit_circle(c_5)

R_ref_6 = poses_6[len(poses_6)//2][:3, :3]
R_ref_5 = poses_5[len(poses_5)//2][:3, :3]

n6_marker_actual = R_ref_6.T @ n6_cam
n5_marker_actual = R_ref_5.T @ n5_cam

z_ee_m_ideal = np.array([0, 1, 0])
y_ee_m_ideal = np.array([0, 0, 1])

if np.dot(n6_marker_actual, z_ee_m_ideal) < 0: n6_marker_actual = -n6_marker_actual
if np.dot(n5_marker_actual, y_ee_m_ideal) < 0: n5_marker_actual = -n5_marker_actual

print("n6_marker:", n6_marker_actual)
print("n5_marker:", n5_marker_actual)

z_axis = n6_marker_actual
y_axis = n5_marker_actual
x_axis = np.cross(y_axis, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)

R_meas = np.column_stack((x_axis, y_axis, z_axis))

nominal_vec = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_vec[5], nominal_vec[4], nominal_vec[3]], degrees=True).as_matrix()

R_err = R_ee_m_ideal @ R_meas
euler = R_scipy.from_matrix(R_err).as_euler('XYZ', degrees=True)
print("R_err_ee:\n", R_err)
print("Euler:", euler)
