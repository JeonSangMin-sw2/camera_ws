"""Per-sweep scatter of the two circle axes a joint calibration is solved from (offline, no robot).

Reads the sweep_points_<arm>_<mode>_joint_A_axis_<a>.txt / ..._joint_B_axis_<b>.txt blocks a joint
calibration leaves in result_txt (one block per sweep), fits each block with the production
circle fit, and prints per sweep: frames, arc, radius, fit RMSE, the angle between the two axes,
and how far each axis is from its mean over the sweeps. Files written before 2026-10-06 have no
mode in the name (sweep_points_<arm>_joint_A_axis_<a>.txt) and are read when the new name is absent.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\joint_sweep_scatter.py <result_txt folder> <arm> <mode> [last N blocks]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402


def load_blocks(path):
    blocks, rows, note = [], [], ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            if rows:
                blocks.append((note, np.array(rows)))
            rows, note = [], ""
        elif line.startswith("# pass"):
            note = line[2:]
        elif line and not line.startswith("#"):
            rows.append([float(v) for v in line.split(",")])
    if rows:
        blocks.append((note, np.array(rows)))
    return blocks


def fit(block):
    angles = block[:, 0]
    poses = [row[10:26].reshape(4, 4) for row in block]
    result = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, robust=True)
    return result["axis_opt"] / np.linalg.norm(result["axis_opt"]), result["radius"], result["rmse"], np.ptp(angles)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def main(argv):
    folder, arm, mode = Path(argv[0]), argv[1], argv[2]
    last = int(argv[3]) if len(argv) > 3 else None
    config = BaseCalibrator.JOINT_CONFIGS[mode]
    axis_a, axis_b = config["sweep_joint_A"], config["sweep_joint_B"]
    fits = {}
    for key, axis in (("A", axis_a), ("B", axis_b)):
        path = folder / f"sweep_points_{arm}_{mode}_joint_{key}_axis_{axis}.txt"
        if not path.exists():
            path = folder / f"sweep_points_{arm}_joint_{key}_axis_{axis}.txt"
        blocks = load_blocks(path)
        blocks = blocks[-last:] if last else blocks
        fits[key] = [(note, len(block)) + fit(block) for note, block in blocks]
    count = min(len(fits["A"]), len(fits["B"]))
    for key in ("A", "B"):
        axes = np.array([f[2] for f in fits[key][:count]])
        axes *= np.sign(axes @ axes[0])[:, None]
        mean = axes.mean(axis=0)
        mean /= np.linalg.norm(mean)
        fits[key + "_dev"] = [angle_deg(a, mean) for a in axes]
    print(f"{arm} arm, {mode}: axis A = J{axis_a}, axis B = J{axis_b}, {count} sweeps")
    print(" sweep                                   | frames arc(deg) rA(mm) rmseA  devA(deg) | frames arc(deg) rB(mm) rmseB  devB(deg) | A^B(deg)")
    for i in range(count):
        (note, n_a, ax_a, r_a, e_a, arc_a), (_, n_b, ax_b, r_b, e_b, arc_b) = fits["A"][i], fits["B"][i]
        between = float(np.degrees(np.arccos(np.clip(np.dot(ax_a, ax_b), -1.0, 1.0))))
        print(f" {note:39s} | {n_a:6d} {arc_a:8.1f} {r_a:6.1f} {e_a:5.3f} {fits['A_dev'][i]:9.3f} "
              f"| {n_b:6d} {arc_b:8.1f} {r_b:6.1f} {e_b:5.3f} {fits['B_dev'][i]:9.3f} | {between:8.3f}")
    print(f" axis scatter (rms from mean): A {np.sqrt(np.mean(np.square(fits['A_dev']))):.3f} deg, "
          f"B {np.sqrt(np.mean(np.square(fits['B_dev']))):.3f} deg")


if __name__ == "__main__":
    main(sys.argv[1:])
