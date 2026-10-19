"""Would the marker stay in view if the v1.3 J4 / J6 sweeps were wider? (offline, no robot)

Takes the marker bracket sweeps a run left in result_txt (sweep_points_<arm>_marker_axis_<4|6>.txt,
camera-frame marker poses against the joint angle), continues the measured rotation to wider
angles, and reports for each half arc the worst marker tilt to the view ray, the smallest distance
of a marker corner to the image edge and the range. The pinhole model is the D405 file in config/
at 1280x720; lens distortion and detection quality are not modelled.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_sweep_view_check.py <result_txt folder>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import expm, logm

ROOT = Path(__file__).resolve().parents[1]
K = np.array(yaml.safe_load((ROOT / "config" / "camera_intrinsics_d405.yaml").read_text(encoding="utf-8"))["camera_matrix"])
WIDTH, HEIGHT, MARKER_HALF_M = 1280, 720, 0.032
HALF_ARCS = (15.0, 20.0, 22.5, 30.0, 40.0)


def last_block(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            rows = []
        elif line and not line.startswith("#"):
            rows.append([float(v) for v in line.split(",")])
    return np.array(rows)


def view(pose):
    position = pose[:3, 3]
    tilt = np.degrees(np.arccos(abs(pose[:3, 2] @ position) / np.linalg.norm(position)))
    corners = [pose[:3, :3] @ np.array([sx * MARKER_HALF_M, sy * MARKER_HALF_M, 0.0]) + position
               for sx in (-1, 1) for sy in (-1, 1)]
    uv = np.array([[K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]] for c in corners])
    margin = min(uv[:, 0].min(), WIDTH - uv[:, 0].max(), uv[:, 1].min(), HEIGHT - uv[:, 1].max())
    return tilt, margin, np.linalg.norm(position) * 1000.0


def main(argv):
    folder = Path(argv[0])
    print(" arm   axis | measured sweep (deg) | half arc: worst tilt (deg) / smallest image margin (px) / range (mm)")
    for arm in ("right", "left"):
        for axis in (4, 6):
            block = last_block(folder / f"sweep_points_{arm}_marker_axis_{axis}.txt")
            angles = block[:, 0]
            poses = block[:, 10:26].reshape(-1, 4, 4)
            first, last = int(np.argmin(angles)), int(np.argmax(angles))
            # Camera-frame motion per degree of the joint, from the two ends of the measured sweep.
            twist = np.real(logm(poses[last] @ np.linalg.inv(poses[first]))) / (angles[last] - angles[first])
            centre_angle = 0.5 * (angles[first] + angles[last])
            cells = []
            for half in HALF_ARCS:
                values = np.array([view(expm(twist * (centre_angle + a - angles[first])) @ poses[first])
                                   for a in np.linspace(-half, half, 41)])
                cells.append(f"+-{half:g}: {values[:, 0].max():4.1f} / {values[:, 1].min():4.0f} / "
                             f"{values[:, 2].min():.0f}..{values[:, 2].max():.0f}")
            print(f" {arm:5s} J{axis}   | {angles[first]:+6.1f} .. {angles[last]:+6.1f}      | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
