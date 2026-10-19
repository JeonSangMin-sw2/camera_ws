"""Image margin and marker tilt over recorded v1.3 wrist-stage sweeps (offline).

Reads sweep_points_<arm>_<mode>_joint_<A|B>_axis_<n>.txt (marker pose in the camera frame per
frame) from a run folder, projects the 80 mm plate's corners with the D405 intrinsics
(config/camera_intrinsics_d405.yaml, 1280x720) and reports per file: sweep angle range, the smallest
distance of a corner to the image border (px) and the largest tilt of the plate against the line of
sight (deg). Detection margins used before: >= 60 px, tilt <= 30-40 deg.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_sweep_view_margin.py <run folder with result/result_txt>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, HALF = 1280, 720, 0.040
K = np.array(yaml.safe_load((ROOT / "config" / "camera_intrinsics_d405.yaml").read_text(encoding="utf-8"))["camera_matrix"])
CORNERS = np.array([[x, y, 0.0, 1.0] for x in (-HALF, HALF) for y in (-HALF, HALF)])


def blocks(path):
    rows, out = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            if rows:
                out.append(rows)
            rows = []
        elif line and not line.startswith("#"):
            values = [float(v) for v in line.split(",") if v.strip()]
            if len(values) >= 26:
                rows.append((values[0], np.array(values[10:26]).reshape(4, 4)))
    if rows:
        out.append(rows)
    return out


def view(T):
    pixels = (K @ (T @ CORNERS.T)[:3]).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    margin = min(pixels[:, 0].min(), WIDTH - pixels[:, 0].max(), pixels[:, 1].min(), HEIGHT - pixels[:, 1].max())
    ray = T[:3, 3] / np.linalg.norm(T[:3, 3])
    tilt = np.degrees(np.arccos(abs(float(T[:3, 2] @ ray))))
    return margin, tilt


folder = Path(sys.argv[1]) / "result" / "result_txt"
print(f"{folder}")
print("file                                                  sweeps  angle range (deg) | min margin (px) | max tilt (deg)")
for path in sorted(folder.glob(sys.argv[2] if len(sys.argv) > 2 else "sweep_points_*_wrist_*_v13_joint_*_axis_*.txt")):
    data = [row for block in blocks(path) for row in block]
    if not data:
        continue
    angles = [a for a, _ in data]
    views = [view(T) for _, T in data]
    print(f"{path.name:54s} {len(blocks(path)):5d}  {min(angles):+6.1f} .. {max(angles):+6.1f}   | "
          f"{min(v[0] for v in views):14.0f} | {max(v[1] for v in views):13.1f}")
