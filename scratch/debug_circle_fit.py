import numpy as np

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

points = np.array([T[:3, 3]*1000 for T in poses])
centroid = np.mean(points, axis=0)
pts_centered = points - centroid
_, _, vh = np.linalg.svd(pts_centered)
normal = vh[2, :]
ex = vh[0, :]
ey = vh[1, :]
pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
uc, vc = res[0], res[1]
center_3d = centroid + uc * ex + vc * ey
print("uc, vc:", uc, vc)
print("center_3d:", center_3d)
dists = np.linalg.norm(points - center_3d, axis=1)
print("dists min, max, mean:", dists.min(), dists.max(), dists.mean())
