import numpy as np
import os
from scipy.optimize import least_squares

def fit_circle_3d(pts):
    # pts: (N, 3)
    # Fit a plane, then project and fit a circle
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean
    uu, dd, vv = np.linalg.svd(pts_centered)
    normal = vv[2, :] # normal vector of the plane
    
    # Project to 2D plane coordinates
    u = vv[0, :]
    v = vv[1, :]
    pts_2d = np.zeros((pts.shape[0], 2))
    for i in range(pts.shape[0]):
        pts_2d[i, 0] = np.dot(pts_centered[i, :], u)
        pts_2d[i, 1] = np.dot(pts_centered[i, :], v)
        
    # Fit circle in 2D
    # (x - cx)^2 + (y - cy)^2 = R^2
    # x^2 - 2*cx*x + cx^2 + y^2 - 2*cy*y + cy^2 = R^2
    # 2*cx*x + 2*cy*y + (R^2 - cx^2 - cy^2) = x^2 + y^2
    A = np.zeros((pts.shape[0], 3))
    B = np.zeros(pts.shape[0])
    for i in range(pts.shape[0]):
        A[i, 0] = 2 * pts_2d[i, 0]
        A[i, 1] = 2 * pts_2d[i, 1]
        A[i, 2] = 1
        B[i] = pts_2d[i, 0]**2 + pts_2d[i, 1]**2
        
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    cx_2d = res[0]
    cy_2d = res[1]
    R = np.sqrt(res[2] + cx_2d**2 + cy_2d**2)
    
    # Residuals
    dists = np.sqrt((pts_2d[:, 0] - cx_2d)**2 + (pts_2d[:, 1] - cy_2d)**2)
    rmse = np.sqrt(np.mean((dists - R)**2))
    
    return R * 1000.0, rmse * 1000.0 # to mm

def load_pts(filename):
    path = os.path.join("/home/rainbow/camera_ws/result/result_txt", filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "===" in line:
                continue
            try:
                parts = [float(p.strip()) for p in line.split(",") if p.strip()]
                if len(parts) >= 4:
                    pts.append([parts[1]/1000.0, parts[2]/1000.0, parts[3]/1000.0])
            except ValueError:
                continue
    return np.array(pts)

# Fit circles for right and left arms
print("--- CIRCLE FITTING FROM SWEEP FILES ---")
for side in ["right", "left"]:
    print(f"\n[{side.upper()} ARM]")
    pts_6 = load_pts(f"sweep_points_{side}_marker_axis_6.txt")
    pts_5 = load_pts(f"sweep_points_{side}_marker_axis_5.txt")
    pts_4 = load_pts(f"sweep_points_{side}_marker_axis_4.txt")
    
    r6, rmse6 = fit_circle_3d(pts_6) if pts_6 is not None else (0, 0)
    r5, rmse5 = fit_circle_3d(pts_5) if pts_5 is not None else (0, 0)
    r4, rmse4 = fit_circle_3d(pts_4) if pts_4 is not None else (0, 0)
    
    print(f"Axis 6 fit: Radius = {r6:.4f} mm, RMSE = {rmse6:.4f} mm")
    print(f"Axis 5 fit: Radius = {r5:.4f} mm, RMSE = {rmse5:.4f} mm")
    if pts_4 is not None:
        print(f"Axis 4 fit: Radius = {r4:.4f} mm, RMSE = {rmse4:.4f} mm")
        
    # Solve least_squares for bracket
    L_5_ee = 126.1
    z_sign = -1.0
    x_nom = 0.0
    y_nom = -54.0 if side == "right" else 54.0
    z_nom = -48.0
    
    if pts_4 is not None:
        def residuals_trans(params):
            xe, ye, ze = params
            r6_pred = np.sqrt(xe**2 + ye**2)
            Z_prime = ze + z_sign * L_5_ee
            r5_pred = np.sqrt(xe**2 + Z_prime**2)
            r4_pred = np.sqrt((ze + z_sign * L_5_ee)**2 + ye**2)
            res = [
                r6_pred - r6,
                r5_pred - r5,
                r4_pred - r4
            ]
            reg_weight = 1e-7
            res.append(reg_weight * (xe - x_nom))
            res.append(reg_weight * (ye - y_nom))
            res.append(reg_weight * (ze - z_nom))
            return res

        initial_guess = [x_nom, y_nom, z_nom]
        lower_bounds = [x_nom - 30.0, y_nom - 30.0, -250.0]
        upper_bounds = [x_nom + 30.0, y_nom + 30.0, 10.0]
        opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
        print(f"Optimized xe, ye, ze: {opt_res.x}")
    else:
        def residuals_trans(params):
            ye, ze = params
            xe = 0.0
            r6_pred = np.sqrt(xe**2 + ye**2)
            Z_prime = ze + z_sign * L_5_ee
            r5_pred = np.sqrt(xe**2 + Z_prime**2)
            res = [
                r6_pred - r6,
                r5_pred - r5
            ]
            reg_weight = 1e-7
            res.append(reg_weight * (ye - y_nom))
            res.append(reg_weight * (ze - z_nom))
            return res

        initial_guess = [y_nom, z_nom]
        lower_bounds = [y_nom - 30.0, -250.0]
        upper_bounds = [y_nom + 30.0, 10.0]
        opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
        print(f"Optimized ye, ze: {opt_res.x}")
