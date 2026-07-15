import numpy as np
import sys

def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    basis1 = Vt[0, :]
    basis2 = Vt[1, :]
    points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A_mat = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    c, resid, rank, s = np.linalg.lstsq(A_mat, b, rcond=None)
    xc, yc = c[0]/2, c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return r

for name in ['sweep_points_right_joint_A_axis_2.txt', 'sweep_points_right_joint_B_axis_4.txt']:
    data = []
    with open(f"result/result_real/result/result_txt/{name}", "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(',')
            if len(parts) > 3:
                pt = [float(p) for p in parts[1:4]]
                data.append(pt)
    data = np.array(data)
    r = fit_circle_3d(data)
    print(f"{name}: R = {r:.2f} mm")
