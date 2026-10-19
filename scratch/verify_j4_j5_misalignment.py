"""Offline check: did the right wrist's J4-J5 geometry change in the 2026-10-01 fall?

Reads saved Step 1 marker-pose sweeps (one joint turned at a time: J4, J5, J6) and nothing else.
No robot, no camera, nothing written except the optional figure passed as argv[1].

Per sweep, the turned joint's axis direction in the camera frame is estimated twice, independently:
  * position : normal of the plane the marker positions lie in,
  * rotation : axis of the marker's own rotation (orientation only), regressed on the joint angle.
The angle between two neighbouring axes is fixed by the link between them; a joint zero offset
cannot change it. So J4^J5 and J5^J6 are compared before and after the fall, right against left.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = [  # (label, folder, after the fall?, same setup as the run after the fall?)
    # 09-15 / 09-21: D435 and an older marker ready pose; kept as extra "before" samples only.
    ("09-15", ROOT / "result/backup_20260915_0923_real_run/result_txt", False, False),
    ("09-21", ROOT / "result/result_txt_before_j6_pose_change", False, False),
    # 09-29 onwards: D405, same intrinsics file (2026-09-29 11:35), same ready pose.
    ("09-29 12:08", ROOT / "archive/runs/20260929_d405/sweeps_1208", False, True),
    ("09-29 15:00", ROOT / "archive/runs/20260929_d405/sweeps_1500", False, True),
    ("10-01 18:17", ROOT / "archive/runs/20261001_after_fall_run2", True, True),
]
# Step 1 J5 sweep at the wrist-pitch pose (J5 = 0): |J4 circle radius - J6 circle radius| [mm] per
# iteration once J5 is within tolerance. Copied from joint_calib_debug_<arm>_wrist_pitch.txt
# ("Circle Size Error"); the 17:3x raw files were overwritten by the 18:1x run, the log values are
# in docs/progress.md.
STEP1_SIZE_ERROR_MM = {
    "right": {"09-29 15:00 (before)": [0.054, 0.055], "10-01 17:3x (after)": [1.557, 1.497],
              "10-01 18:1x (after, bracket refitted)": [1.211, 1.516]},
    "left": {"09-29 15:00 (before)": [0.131, 0.080], "10-01 17:3x (after)": [0.085, 0.139],
             "10-01 18:1x (after, bracket refitted)": [0.089, 0.009]},
}
ARMS = ("right", "left")
AXES = (4, 5, 6)
N_BOOT = 300
RNG = np.random.default_rng(0)


def load_blocks(path):
    """[(theta_deg (N,), T_cam2marker (N,4,4))] per sweep block ("NEW ITERATION" separates passes)."""
    blocks, theta, poses = [], [], []

    def close():
        if len(theta) >= 50:
            blocks.append((np.array(theta), np.array(poses)))

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "NEW ITERATION" in line:
            close()
            theta, poses = [], []
            continue
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.split(",") if x.strip()]
        if len(values) < 26:
            continue
        theta.append(values[0])
        poses.append(np.array(values[10:26]).reshape(4, 4))
    close()
    return blocks


def rotation_vector(R):
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def axis_from_rotation(theta_deg, poses):
    """Axis (unit, camera frame, right-handed for increasing joint angle) from orientations only."""
    ref = len(theta_deg) // 2
    d_theta = np.radians(theta_deg - theta_deg[ref])
    w = np.array([rotation_vector(T[:3, :3] @ poses[ref][:3, :3].T) for T in poses])
    keep = np.abs(d_theta) > np.radians(2.0)
    slope = (w[keep] * d_theta[keep, None]).sum(axis=0) / (d_theta[keep] ** 2).sum()
    return slope / np.linalg.norm(slope), float(np.linalg.norm(slope))  # |slope| = 1 for an ideal sweep


def plane_and_circle(points_mm):
    """Plane normal, circle centre and radius of the marker path (3-sigma out-of-plane rejection)."""
    pts = points_mm
    for _ in range(3):
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2]
        out = (pts - centroid) @ normal
        keep = np.abs(out) < 3.0 * max(out.std(), 1e-6)
        if keep.all():
            break
        pts = pts[keep]
    ex, ey = vh[0], vh[1]
    xy = np.column_stack([(pts - centroid) @ ex, (pts - centroid) @ ey])
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
    centre = centroid + sol[0] * ex + sol[1] * ey
    radius = float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
    return normal, centre, radius, float(out.std())


def fit_sweep(theta_deg, poses):
    points = poses[:, :3, 3] * 1000.0
    n_rot, slope = axis_from_rotation(theta_deg, poses)
    n_pos, centre, radius, out_rms = plane_and_circle(points)
    if np.dot(n_pos, n_rot) < 0:
        n_pos = -n_pos
    start = int(np.argmin(np.abs(theta_deg)))
    return {"n_pos": n_pos, "n_rot": n_rot, "c": centre, "r": radius, "out_rms": out_rms, "slope": slope,
            "p0": points[start], "span": float(theta_deg.max() - theta_deg.min()), "n": len(theta_deg)}


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def bootstrap_angle(block_a, block_b, key):
    """Std of the angle between two sweep axes under frame resampling (optimistic: frames are correlated)."""
    values = []
    for _ in range(N_BOOT):
        fits = []
        for theta, poses in (block_a, block_b):
            idx = np.sort(RNG.integers(0, len(theta), len(theta)))
            fits.append(fit_sweep(theta[idx], poses[idx]))
        values.append(angle_deg(fits[0][key], fits[1][key]))
    return float(np.std(values))


def rodrigues(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def rotate_about_line(point, centre, axis, angle_rad):
    return centre + rodrigues(axis, angle_rad) @ (point - centre)


def distance_to_line(point, centre, axis):
    d = point - centre
    return float(np.linalg.norm(d - np.dot(d, axis) * axis))


def predicted_wrist_pose_mismatch(f, j6_ready_deg):
    """From the marker-pose axes (J5 = 90 deg), where would the marker be at the wrist-pitch pose
    (J5 = 0, J6 = 0)? Returns the best J5 correction and the J4/J6 circle radius difference there,
    i.e. what the Step 1 J5 sweep reports as 'Circle Size Error' once J5 has converged."""
    n4, n5, n6 = f[4]["n_pos"], f[5]["n_pos"], f[6]["n_pos"]
    c4, c5, c6 = f[4]["c"], f[5]["c"], f[6]["c"]
    p = rotate_about_line(f[6]["p0"], c6, n6, np.radians(-j6_ready_deg))  # J6: ready -> 0
    best = None
    for delta in np.radians(np.linspace(-3.0, 3.0, 601)):
        turn = np.radians(-90.0) + delta  # J5: 90 -> 0 (+ a zero correction)
        p2 = rotate_about_line(p, c5, n5, turn)
        c6_2 = rotate_about_line(c6, c5, n5, turn)
        n6_2 = rodrigues(n5, turn) @ n6
        r4, r6 = distance_to_line(p2, c4, n4), distance_to_line(p2, c6_2, n6_2)
        foot4 = c4 + np.dot(p2 - c4, n4) * n4
        foot6 = c6_2 + np.dot(p2 - c6_2, n6_2) * n6_2
        d = foot6 - foot4
        centre_dist = float(np.linalg.norm(d - np.dot(d, n4) * n4))
        tilt = min(angle_deg(n4, n6_2), 180.0 - angle_deg(n4, n6_2))
        if best is None or centre_dist < best["centre_dist"]:
            best = {"delta_deg": float(np.degrees(delta)), "r4": r4, "r6": r6, "centre_dist": centre_dist, "tilt": tilt}
    return best


def main():
    rows = []
    for label, folder, after, same_setup in SETS:
        for arm in ARMS:
            blocks = {ax: load_blocks(folder / f"sweep_points_{arm}_marker_axis_{ax}.txt") for ax in AXES}
            for k in range(min(len(blocks[ax]) for ax in AXES)):
                f = {ax: fit_sweep(*blocks[ax][k]) for ax in AXES}
                row = {"label": label if k == 0 else f"{label} #{k + 1}", "arm": arm, "after": after,
                       "same_setup": same_setup, "fit": f}
                for name, (a, b) in (("45", (4, 5)), ("56", (5, 6)), ("46", (4, 6))):
                    for key in ("n_pos", "n_rot"):
                        row[name, key] = angle_deg(f[a][key], f[b][key])
                    row[name, "boot"] = bootstrap_angle(blocks[a][k], blocks[b][k], "n_pos")
                rows.append(row)

    print("== 1. Angle between neighbouring wrist axes [deg] (pos = marker path plane, rot = marker rotation axis) ==")
    print(f"{'set':15s} {'arm':5s} | J4^J5 pos (boot sd)   rot   | J5^J6 pos (boot sd)   rot   | J4^J6 pos    rot   | frames 4/5/6")
    for r in rows:
        f = r["fit"]
        print(f"{r['label']:15s} {r['arm']:5s} | {r['45', 'n_pos']:7.2f} ({r['45', 'boot']:.2f})   {r['45', 'n_rot']:7.2f} |"
              f" {r['56', 'n_pos']:7.2f} ({r['56', 'boot']:.2f})   {r['56', 'n_rot']:7.2f} |"
              f" {r['46', 'n_pos']:7.2f} {r['46', 'n_rot']:7.2f} | {f[4]['n']}/{f[5]['n']}/{f[6]['n']}"
              + ("   <- after the fall" if r["after"] else ""))

    print()
    print("== 2. J4^J5 and J5^J6 (position method), before vs after the fall ==")
    for scope, pick in (("same setup (D405, 09-29 vs 10-01)", lambda r: r["same_setup"]), ("all runs", lambda r: True)):
        print(f"-- {scope}")
        for arm in ARMS:
            for name in ("45", "56"):
                before = np.array([r[name, "n_pos"] for r in rows if r["arm"] == arm and not r["after"] and pick(r)])
                after = np.array([r[name, "n_pos"] for r in rows if r["arm"] == arm and r["after"]])
                sd = before.std(ddof=1)
                change = after.mean() - before.mean()
                outside = "outside" if (after.max() < before.min() or after.min() > before.max()) else "inside"
                print(f"   {arm:5s} J{name[0]}^J{name[1]}: before {before.mean():6.2f} (sd {sd:.2f}, n={len(before)}, "
                      f"{before.min():.2f}..{before.max():.2f}) | after {', '.join(f'{v:.2f}' for v in after)} | "
                      f"change {change:+.2f} deg ({abs(change) / max(sd, 1e-9):.1f} sd, {outside} the earlier range)")

    print()
    print("== 3. What Step 1 itself measured at the wrist-pitch pose (J5 = 0): |r(J4 circle) - r(J6 circle)| [mm] ==")
    for arm in ARMS:
        print(f"   {arm:5s}: " + " | ".join(f"{k}: {', '.join(f'{v:.2f}' for v in vals)}" for k, vals in STEP1_SIZE_ERROR_MM[arm].items()))

    print()
    print("== 4. Radii (marker to each axis) [mm]: a shifted (not tilted) part would change these ==")
    for arm in ARMS:
        for name, pick in (("before (09-29)", lambda r: r["same_setup"] and not r["after"]), ("after", lambda r: r["after"])):
            sel = [r["fit"] for r in rows if r["arm"] == arm and pick(r)]
            print(f"   {arm:5s} {name:15s}: " + "  ".join(
                f"r{ax} {np.mean([f[ax]['r'] for f in sel]):6.2f} (sd {np.std([f[ax]['r'] for f in sel], ddof=1):.2f})" for ax in AXES))

    print()
    print("== 5. Does the J4^J5 change have the right size and direction for the Step 1 result? ==")
    print("   Take each right-arm run before the fall, tilt everything from the J5 axis outwards against the J4 axis")
    print("   by the measured J4^J5 change, and recompute the J4/J6 circle radius difference at the wrist-pitch pose.")
    before_45 = np.mean([r["45", "n_pos"] for r in rows if r["arm"] == "right" and r["same_setup"] and not r["after"]])
    after_45 = np.mean([r["45", "n_pos"] for r in rows if r["arm"] == "right" and r["after"]])
    alpha = np.radians(before_45 - after_45)  # > 0: the J5 axis moved towards the J4 axis direction
    changes = []
    for r in rows:
        if not (r["arm"] == "right" and r["same_setup"] and not r["after"]):
            continue
        f = r["fit"]
        base = predicted_wrist_pose_mismatch(f, -10.0)
        n4, n5, c4, c5 = f[4]["n_pos"], f[5]["n_pos"], f[4]["c"], f[5]["c"]
        k = np.cross(n5, n4)
        k /= np.linalg.norm(k)
        pivot = c5 + np.dot(c4 - c5, n5) * n5  # wrist centre (on the J5 line, nearest the J4 line)
        R = rodrigues(k, alpha)
        g = {4: f[4]}
        for ax in (5, 6):
            g[ax] = dict(f[ax], n_pos=R @ f[ax]["n_pos"], c=pivot + R @ (f[ax]["c"] - pivot), p0=pivot + R @ (f[ax]["p0"] - pivot))
        tilted = predicted_wrist_pose_mismatch(g, -10.0)
        change = (tilted["r4"] - tilted["r6"]) - (base["r4"] - base["r6"])
        changes.append(change)
        print(f"   {r['label']:15s}: J4^J5 {angle_deg(g[4]['n_pos'], g[5]['n_pos']):.2f} deg after the tilt -> r4 - r6 changes by {change:+.2f} mm"
              f" (J5 zero would move by {tilted['delta_deg'] - base['delta_deg']:+.2f} deg)")
    lever = np.mean([np.sqrt(max(r["fit"][5]["r"] ** 2 - r["fit"][6]["r"] ** 2, 0.0)) for r in rows if r["same_setup"] and r["arm"] == "right"])
    print(f"   tilt {np.degrees(alpha):.2f} deg x lever {lever:.0f} mm (wrist centre to marker along the hand axis) = {alpha * lever:.2f} mm")
    print(f"   predicted change {np.mean(changes):+.2f} mm | Step 1 measured: r4 - r6 went from about 0 (|0.05|) to -1.2..-1.7 mm (r4 52.5 vs r6 54.2 in the 18:1x run)")
    print("   Note: the absolute radius difference reconstructed from the marker-pose sweeps alone is off by 3-5 mm for")
    print("   every run and both arms (axis directions carry a common 1-2 deg bias), so only this change is meaningful.")

    if len(sys.argv) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, (name, title) in zip(axes, (("45", "J4 axis ^ J5 axis"), ("56", "J5 axis ^ J6 axis"))):
            for arm, colour in (("right", "tab:red"), ("left", "tab:blue")):
                sel = [r for r in rows if r["arm"] == arm and r["same_setup"]]
                x = np.arange(len(sel))
                ax.errorbar(x, [r[name, "n_pos"] for r in sel], yerr=[r[name, "boot"] for r in sel], fmt="o-",
                            color=colour, capsize=3, label=f"{arm} arm")
                ax.set_xticks(x)
                ax.set_xticklabels([r["label"] for r in sel], rotation=30, ha="right")
            first_after = next(i for i, r in enumerate([r for r in rows if r["arm"] == "right" and r["same_setup"]]) if r["after"])
            ax.axvline(first_after - 0.5, color="gray", linestyle="--")
            ax.text(first_after - 0.45, ax.get_ylim()[0], " fall", color="gray", va="bottom")
            ax.axhline(90.0, color="black", linewidth=0.6)
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("angle between axes [deg] (design: 90)")
        axes[0].legend()
        fig.suptitle("Wrist axis angles from Step 1 marker sweeps (D405 runs, error bar = frame-resampling sd)")
        fig.tight_layout()
        fig.savefig(sys.argv[1], dpi=130)
        print(f"\nfigure saved: {sys.argv[1]}")


if __name__ == "__main__":
    main()
