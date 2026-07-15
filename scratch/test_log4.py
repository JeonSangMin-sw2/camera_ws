import sys
import numpy as np
sys.path.append('/home/rainbow/camera_ws')
from scipy.spatial.transform import Rotation as R_scipy

def load_poses(filename):
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                vals = list(map(float, line.strip().split(',')))
                T = np.array(vals[10:26]).reshape(4, 4)
                poses.append(T)
    return poses

poses_6 = load_poses('/home/rainbow/camera_ws/core/calibration/result_txt/sweep_points_right_marker_axis_6.txt')
poses_5 = load_poses('/home/rainbow/camera_ws/core/calibration/result_txt/sweep_points_right_marker_axis_5.txt')

def fit_circle(points):
    centroid = np.mean(points, axis=0)
    U, S, Vt = np.linalg.svd(points - centroid)
    return Vt[2, :]

n6_cam = fit_circle(np.array([T[:3, 3] for T in poses_6]))
n5_cam = fit_circle(np.array([T[:3, 3] for T in poses_5]))

R_ref_6 = poses_6[len(poses_6)//2][:3, :3]
R_ref_5 = poses_5[len(poses_5)//2][:3, :3]

n6_m = R_ref_6.T @ n6_cam
n5_m = R_ref_5.T @ n5_cam

if np.dot(n6_m, [0, 1, 0]) < 0: n6_m = -n6_m
if np.dot(n5_m, [0, 0, 1]) < 0: n5_m = -n5_m

z_meas = n6_m
y_meas = n5_m
x_meas = np.cross(y_meas, z_meas)
x_meas /= np.linalg.norm(x_meas)
y_meas = np.cross(z_meas, x_meas)

R_meas = np.column_stack((x_meas, y_meas, z_meas))
R_ideal = R_scipy.from_euler('ZYX', [180.0, 0.0, 90.0], degrees=True).as_matrix()

R_err = R_ideal @ R_meas
euler = R_scipy.from_matrix(R_err).as_euler('XYZ', degrees=True)
ext = euler[2]
print(f"Extracted theta: {ext}")
print(f"Resulting offset if init_pos[6]=-10.0: {ext - (-10.0)}")
