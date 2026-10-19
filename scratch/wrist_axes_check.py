"""Offline: where along the right arm did the geometry change? Saved Step 1 sweeps, tool's own fit.

Per sweep the turned joint's axis is a line in the camera frame. Joint-zero errors do not change
(a) the angle / gap between neighbouring axes, or (b) an axis expressed in the marker frame, so those
are compared between 2026-09-29 (before the fall) and the run after the fall. Read-only.
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\User\Desktop\camera_ws")
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from core.calibration.CalibratorBase import BaseCalibrator

SETS = {
    "09-29 12:08": ROOT / "archive/runs/20260929_d405/sweeps_1208",
    "09-29 15:00": ROOT / "archive/runs/20260929_d405/sweeps_1500",
    "10-01 run2 ": ROOT / "archive/runs/20261001_after_fall_run2",
}
out = sys.__stdout__


def load(path):
    angles, poses = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "NEW ITERATION" in line:  # one block per pass; keep the last
            angles, poses = [], []
            continue
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split(",") if x.strip()]
        if len(v) < 26:
            continue
        angles.append(v[0])
        poses.append(np.array(v[10:26]).reshape(4, 4))
    return angles, poses


def fit(path):
    angles, poses = load(path)
    with contextlib.redirect_stdout(io.StringIO()):
        res = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, robust=True)
    n = np.asarray(res["axis_opt"], float)
    n /= np.linalg.norm(n)
    c = np.asarray(res["c_opt"], float)  # mm
    # The same axis line in the marker frame (constant during the sweep): mean over samples.
    dirs, pts = [], []
    for T in poses:
        R, t = T[:3, :3], T[:3, 3] * 1000.0
        dirs.append(R.T @ n)
        p = R.T @ (c - t)
        d = R.T @ n
        pts.append(p - np.dot(p, d) * d)  # point of the line closest to the marker origin
    dirs, pts = np.array(dirs), np.array(pts)
    nm = dirs.mean(axis=0)
    nm /= np.linalg.norm(nm)
    return {"n": n, "c": c, "r": float(res["radius"]), "rmse": float(res["rmse"]),
            "nm": nm, "pm": pts.mean(axis=0), "nm_spread": float(np.degrees(np.std(dirs, axis=0)).max()),
            "pm_spread": float(pts.std(axis=0).max())}


def gap(f, g):
    cross = np.cross(f["n"], g["n"])
    return float(abs(np.dot(g["c"] - f["c"], cross)) / np.linalg.norm(cross))


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


fits = {}
for label, folder in SETS.items():
    for arm in ("right", "left"):
        fits[label, arm] = {ax: fit(folder / f"sweep_points_{arm}_marker_axis_{ax}.txt") for ax in (4, 5, 6)}

print("== A. Marker pose (J5 = 90 deg). Angles between neighbouring wrist axes [deg] and line gaps [mm] ==", file=out)
print("set          arm    J4^J5   J5^J6   J4^J6 | gap J4-J5  J5-J6  J4-J6 | r4      r5      r6", file=out)
for (label, arm), f in fits.items():
    a45 = angle(f[4]["n"], f[5]["n"]); a56 = angle(f[5]["n"], f[6]["n"]); a46 = angle(f[4]["n"], f[6]["n"])
    a45, a56, a46 = (min(a, 180 - a) for a in (a45, a56, a46))
    print(f"{label}  {arm:5s}  {a45:6.2f}  {a56:6.2f}  {a46:6.2f} |   {gap(f[4], f[5]):5.2f}   {gap(f[5], f[6]):5.2f}  {gap(f[4], f[6]):5.2f} |"
          f" {f[4]['r']:6.2f}  {f[5]['r']:6.2f}  {f[6]['r']:6.2f}", file=out)

print(file=out)
print("== B. J6 axis seen from the marker (depends only on flange + bracket + marker plate) ==", file=out)
print("set          arm    axis direction in marker frame        nearest point to marker origin [mm]    spread dir/pt", file=out)
for (label, arm), f in fits.items():
    g = f[6]
    s = np.sign(g["nm"][np.argmax(np.abs(g["nm"]))])
    print(f"{label}  {arm:5s}  [{s*g['nm'][0]:+.4f} {s*g['nm'][1]:+.4f} {s*g['nm'][2]:+.4f}]   "
          f"[{g['pm'][0]:+7.2f} {g['pm'][1]:+7.2f} {g['pm'][2]:+7.2f}]   {g['nm_spread']:.2f} deg / {g['pm_spread']:.2f} mm", file=out)

print(file=out)
print("== C. Change against 09-29 15:00, per arm (axis seen from the marker) ==", file=out)
ref = "09-29 15:00"
for label in ("09-29 12:08", "10-01 run2 "):
    for arm in ("right", "left"):
        f, g = fits[label, arm], fits[ref, arm]
        row = []
        for ax in (6, 5, 4):
            a = angle(f[ax]["nm"], g[ax]["nm"]); a = min(a, 180 - a)
            row.append(f"J{ax}: {a:5.2f} deg, {np.linalg.norm(f[ax]['pm'] - g[ax]['pm']):5.2f} mm")
        print(f"{label} vs {ref}  {arm:5s}  " + " | ".join(row), file=out)

print(file=out)
print("== D. Run 2, wrist-pitch pose (first iteration, J5 not yet corrected) and elbow pose ==", file=out)
folder = SETS["10-01 run2 "]
for arm in ("right", "left"):
    for name, a_file, b_file in (("wrist J4|J6", "joint_A_axis_4", "joint_B_axis_6"), ("elbow J2|J4", "joint_A_axis_2", "joint_B_axis_4")):
        A = fit(folder / f"sweep_points_{arm}_{a_file}.txt")
        B = fit(folder / f"sweep_points_{arm}_{b_file}.txt")
        d = B["c"] - A["c"]
        perp = d - np.dot(d, A["n"]) * A["n"]
        ang = angle(A["n"], B["n"]); ang = min(ang, 180 - ang)
        print(f"{arm:5s} {name}: angle {ang:.2f} deg | centre offset {np.linalg.norm(perp):.2f} mm | rA {A['r']:.2f} rB {B['r']:.2f} (diff {A['r'] - B['r']:+.2f}) mm", file=out)
