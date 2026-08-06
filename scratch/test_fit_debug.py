import numpy as np

for axis in [4, 5, 6]:
    fpath = f"/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_{axis}.txt"
    angles = []
    cam_pts = []
    with open(fpath, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("#") or not line_str or "=" in line_str:
                continue
            try:
                parts = [float(x.strip()) for x in line_str.split(",")]
                angles.append(parts[0])
                cam_pts.append(parts[1:4])
            except ValueError:
                continue

    angles = np.array(angles)
    cam_pts = np.array(cam_pts)

    first_iter_len = 0
    for i in range(1, len(angles)):
        if angles[i] < angles[i-1] - 5.0:
            first_iter_len = i
            break
    if first_iter_len == 0:
        first_iter_len = len(angles)

    cam_pts_first = cam_pts[:first_iter_len]
    angles_first = angles[:first_iter_len]
    
    centroid = np.mean(cam_pts_first, axis=0)
    centered = cam_pts_first - centroid
    U, S, Vt = np.linalg.svd(centered)
    basis1, basis2 = Vt[0, :], Vt[1, :]
    points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
    A = np.column_stack((points_2d[:, 0], points_2d[:, 1], np.ones_like(points_2d[:, 0])))
    b = points_2d[:, 0]**2 + points_2d[:, 1]**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    print(f"\nAxis {axis}:")
    print(f"  Points: {len(angles_first)}")
    print(f"  Angle range: {np.min(angles_first):.2f} to {np.max(angles_first):.2f} deg")
    print(f"  Euclidean dist first-last: {np.linalg.norm(cam_pts_first[0] - cam_pts_first[-1]):.2f} mm")
    print(f"  Algebraic fit radius: {r:.2f} mm")
