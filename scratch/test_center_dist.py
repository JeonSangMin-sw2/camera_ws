import numpy as np

# Load circle centers and axis from the debug files
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
    center_3d = centroid + xc * basis1 + yc * basis2
    return center_3d, r, normal

def load_pts(name):
    data = []
    with open(f"result/result_real/result/result_txt/{name}", "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(',')
            if len(parts) > 3:
                pt = [float(p) for p in parts[1:4]]
                data.append(pt)
    return np.array(data)

pts_A = load_pts("sweep_points_right_joint_A_axis_2.txt")
pts_B = load_pts("sweep_points_right_joint_B_axis_4.txt")

c_A, r_A, n_A = fit_circle_3d(pts_A)
c_B, r_B, n_B = fit_circle_3d(pts_B)

print(f"c_A: {c_A}, r_A: {r_A}")
print(f"c_B: {c_B}, r_B: {r_B}")
print(f"center_dist: {np.linalg.norm(c_B - c_A):.4f}")

