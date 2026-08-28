import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')
from core.calibration.CalibratorBase import BaseCalibrator

txt_path = '/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_5.txt'
poses = []
angles = []
with open(txt_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('='):
            continue
        parts = [float(x.strip()) for x in line.split(',')]
        angles.append(parts[0])
        T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
        poses.append(T_cam2marker)

print(f"Loaded {len(poses)} poses.")

# Run standard fit_circle_3d
pts = np.array([T[:3, 3]*1000 for T in poses])
c3d, R3d, r3d, rmse3d, _, _, _ = BaseCalibrator.fit_circle_3d(pts, robust=False)
print(f"BaseCalibrator.fit_circle_3d: radius = {r3d:.2f}mm, rmse = {rmse3d:.3f}mm")

# Run fit_circle_3d_and_6dof_misalignment with axis_prior=[0, 1, 0]
res_with_prior = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, axis_prior=[0.0, 1.0, 0.0], robust=False)
print(f"fit_circle_3d_and_6dof_misalignment (axis_prior=[0,1,0]): radius = {res_with_prior['radius']:.2f}mm, rmse = {res_with_prior['rmse']:.3f}mm")

# Run fit_circle_3d_and_6dof_misalignment with axis_prior=None
res_no_prior = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, axis_prior=None, robust=False)
print(f"fit_circle_3d_and_6dof_misalignment (axis_prior=None): radius = {res_no_prior['radius']:.2f}mm, rmse = {res_no_prior['rmse']:.3f}mm")
