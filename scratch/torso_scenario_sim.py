"""Torso calibration scenario check with the simulator's kinematic model (sim 127.0.0.1 only, no motion).

A 100 mm marker is fixed on a wall in front of the robot. For each proposed motion, compute the marker
seen from the head camera, then inject a 0.5 deg zero offset into one torso joint at a time and report how
much each motion's feature moves. Features are chosen so they do not depend on the (unknown) wall marker
pose: rotation axis lines and the camera's own translation path, both in the camera frame.
usage: python scratch/torso_scenario_sim.py <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby

OUT = Path(sys.argv[1]) / ("hip_roll" if "--hip-roll" in sys.argv else ""); OUT.mkdir(parents=True, exist_ok=True)
D = np.deg2rad
L = 0.35                       # torso_1 -> torso_2 -> torso_3 link length (sim model)
DELTA = D(0.5)
THETA0 = D(30.0)               # bent "ready" posture: q1 = t, q2 = -2t, q3 = t

if "--urdf" in sys.argv:          # offline: same model "m" kinematics from a saved URDF, no sim needed
    names = list(rby.Model_M().robot_joint_names)
    urdf = Path(sys.argv[sys.argv.index("--urdf") + 1]).read_text(encoding="utf-8")
    dyn = getattr(rby.dynamics, f"Robot_{len(names)}")(rby.dynamics.load_robot_from_urdf_data(urdf, "base"))
else:
    robot = rby.create_robot("127.0.0.1:50051", "m")
    assert robot.connect(), "sim not reachable"
    names = list(robot.model().robot_joint_names)
    dyn = robot.get_dynamics()
state = dyn.make_state(["base", "link_head_2"], names)
TORSO = [names.index(f"torso_{i}") for i in range(6)]
# camera optical frame on link_head_2: z forward (head x), x = -head y, y = -head z (upper body assumed calibrated)
R_H2C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def cam_pose(torso, delta=np.zeros(6)):
    q = np.zeros(len(names))
    for k, idx in enumerate(TORSO):
        q[idx] = torso[k] + delta[k]
    state.set_q(q); dyn.compute_forward_kinematics(state)
    T = np.array(dyn.compute_transformation(state, 0, 1))
    T[:3, :3] = T[:3, :3] @ R_H2C
    return T


def bent(t, dx=0.0):
    """Torso angles for squat parameter t (upright upper body), optionally shifted forward by dx at constant height."""
    if dx == 0.0:
        return np.array([0, t, -2 * t, t, 0, 0.0])
    z = 2 * L * np.cos(t); x = dx
    r = np.hypot(x, z); a2 = -2 * np.arccos(r / (2 * L)); a1 = np.arctan2(x, z) - a2 / 2
    return np.array([0, a1, a2, -(a1 + a2), 0, 0.0])


T_cam0 = cam_pose(bent(THETA0))
p_marker = T_cam0[:3, 3] + np.array([1.5, 0, 0])            # 1.5 m straight ahead at camera height
R_marker = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float).T   # tag z toward robot, y down (plumb)
T_marker = np.eye(4); T_marker[:3, :3] = R_marker; T_marker[:3, 3] = p_marker


def observe(torso, delta):
    return np.linalg.inv(cam_pose(torso, delta)) @ T_marker


def motions():
    s = np.linspace(-1, 1, 21)
    m = {}
    m["M1 허리 pitch (torso_3)"] = [bent(THETA0) + np.array([0, 0, 0, D(15) * v, 0, 0]) for v in s]
    m["M2 허리 roll (torso_4)"] = [bent(THETA0) + np.array([0, 0, 0, 0, D(15) * v, 0]) for v in s]
    m["M3 허리 yaw (torso_5)"] = [bent(THETA0) + np.array([0, 0, 0, 0, 0, D(20) * v]) for v in s]
    m["M4 수직 이동 (torso_1·2·3)"] = [bent(THETA0 + D(15) * v) for v in s]
    m["M5 전후 이동 (torso_1·2·3)"] = [bent(THETA0, dx=0.08 * v) for v in s]
    if "--hip-roll" in sys.argv:   # optional extra motion: sweep torso_0 (hip roll) too
        m["M6 엉덩이 roll (torso_0)"] = [bent(THETA0) + np.array([D(10) * v, 0, 0, 0, 0, 0]) for v in s]
    return m


def fit_axis(points):
    """Rotation axis line of points on a circle: normal by SVD, centre by algebraic circle fit in the plane."""
    c0 = points.mean(0); _, _, vt = np.linalg.svd(points - c0); n = vt[2]
    u = vt[0]; v = vt[1]
    P = np.c_[(points - c0) @ u, (points - c0) @ v]
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    cx, cy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    return n, c0 + cx * u + cy * v


def angle(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def features(name, poses, delta):
    obs = [observe(q, delta) for q in poses]
    if name.startswith(("M1", "M2", "M3")):
        pts = np.array([T[:3, 3] for T in obs])
        n, c = fit_axis(pts)
        return {"kind": "axis", "n": n, "c": c, "pts": pts, "g": obs[10][:3, 1]}
    # translation of the camera seen in the camera frame = marker displacement (marker is fixed)
    pts = np.array([T[:3, 3] for T in obs])
    d = pts[-1] - pts[0]
    line = pts[0] + np.outer((pts - pts[0]) @ d / (d @ d), d)
    rot = max(np.degrees(np.linalg.norm(np.array(rby_log(obs[0][:3, :3].T @ T[:3, :3])))) for T in obs)
    return {"kind": "path", "d": d, "dev": np.abs(pts - line).max() * 1000, "pts": pts, "rot": rot, "g": obs[10][:3, 1]}


def rby_log(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def axis_offset_mm(n, c, n0, c0):
    """Distance between the two axis lines near the circle centre (mm)."""
    w = c - c0
    return np.linalg.norm(w - (w @ n0) * n0) * 1000


rows = {}
traj = {}
for name, poses in motions().items():
    base = features(name, poses, np.zeros(6))
    traj[name] = {"nominal": base["pts"].tolist()}
    row = {}
    for j in range(6):
        dl = np.zeros(6); dl[j] = DELTA
        f = features(name, poses, dl)
        if base["kind"] == "axis":
            row[f"torso_{j}"] = {
                "axis_dir_deg": angle(f["n"], base["n"]),
                "axis_pos_mm": axis_offset_mm(f["n"], f["c"], base["n"], base["c"]),
                "vs_plumb_deg": angle(f["n"], f["g"]) - angle(base["n"], base["g"]),
            }
        else:
            row[f"torso_{j}"] = {
                "path_dir_deg": angle(f["d"], base["d"]),
                "path_bend_mm": f["dev"] - base["dev"],
                "vs_plumb_deg": angle(f["d"], f["g"]) - angle(base["d"], base["g"]),
            }
        traj[name][f"torso_{j}"] = f["pts"].tolist()
    rows[name] = row
    print(name, "nominal path/axis in camera frame:",
          np.round(base["n"] if base["kind"] == "axis" else base["d"] / np.linalg.norm(base["d"]), 3))
    for j, v in row.items():
        print("   ", j, {k: round(float(x), 4) for k, x in v.items()})

# Identifiability over all motions: unknowns = 6 torso offsets + wall marker pose (6).
def stack(delta, dm):
    Tm = T_marker.copy()
    Tm[:3, 3] = p_marker + dm[:3]
    w = dm[3:]; th = np.linalg.norm(w)
    if th > 0:
        k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Tm[:3, :3] = (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R_marker
    out = []
    for k, q in enumerate(q for poses in motions().values() for q in poses):
        T = np.linalg.inv(cam_pose(q, delta)) @ Tm
        # rotation relative to the nominal observation: the raw tag pose is near a 180 deg turn where log is singular
        out += list(T[:3, 3] / 0.001) + list(np.degrees(rby_log(NOMINAL[k][:3, :3].T @ T[:3, :3])) / 0.1)   # 1 mm, 0.1 deg
    return np.array(out)


NOMINAL = [observe(q, np.zeros(6)) for poses in motions().values() for q in poses]


x0 = np.zeros(12); y0 = stack(x0[:6], x0[6:]); eps = 1e-5
J = np.zeros((len(y0), 12))
for i in range(12):
    x = x0.copy(); x[i] += eps
    J[:, i] = (stack(x[:6], x[6:]) - y0) / eps
labels = [f"torso_{j}" for j in range(6)] + ["m.x", "m.y", "m.z", "m.rx", "m.ry", "m.rz"]


def svd_report(Jm, cols, tag):
    _, s, vt = np.linalg.svd(Jm, full_matrices=False)
    print(f"\n[{tag}] singular values:", np.round(s, 3))
    weak = []
    for k in range(len(s)):
        if s[k] < s[0] * 1e-6:
            combo = {cols[i]: round(float(vt[k, i]), 3) for i in np.argsort(-abs(vt[k]))[:4] if abs(vt[k, i]) > 0.05}
            weak.append(combo); print("   unobservable:", combo)
    return s.tolist(), weak


s_free, weak_free = svd_report(J, labels, "marker pose free")
keep = [i for i in range(12) if labels[i] not in ("m.rx", "m.ry")]   # plumb marker: tilt about base x/y known
s_plumb, weak_plumb = svd_report(J[:, keep], [labels[i] for i in keep], "marker plumb (tilt known)")

# Expected precision with all motions combined, marker plumb: 2 mm / 0.5 deg per observation (100 mm tag at 1.5 m, assumed).
Jp = J[:, keep].copy()
scale = np.tile([0.5] * 3 + [0.2] * 3, len(Jp) // 6)[:, None]   # rows were in 1 mm / 0.1 deg units
cov = np.linalg.inv((Jp * scale).T @ (Jp * scale))
std_plumb = {labels[i]: float(np.degrees(np.sqrt(cov[k, k]))) for k, i in enumerate(keep) if i < 6}
print("expected 1-sigma (deg), plumb, 2 mm / 0.5 deg noise:", {k: round(v, 3) for k, v in std_plumb.items()})

# Marker image track (u, v) per motion, nominal and with a 2 deg offset in each joint, for the slides.
FX, CX, CY = 615.0, 320.0, 240.0
img = {}
for name, poses in motions().items():
    img[name] = {}
    for j in [None] + list(range(6)):
        dl = np.zeros(6)
        if j is not None:
            dl[j] = D(2.0)
        P = np.array([observe(q, dl)[:3, 3] for q in poses])
        img[name]["nominal" if j is None else f"torso_{j}"] = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY, P[:, 2]].tolist()
json.dump(img, open(OUT / "torso_image_tracks.json", "w", encoding="utf-8"))

json.dump({"std_plumb_deg": std_plumb, "features": rows, "svd_free": s_free, "weak_free": weak_free, "svd_plumb": s_plumb,
           "weak_plumb": weak_plumb, "marker_distance_m": 1.5, "delta_deg": 0.5},
          open(OUT / "torso_sensitivity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
json.dump(traj, open(OUT / "torso_traj.json", "w", encoding="utf-8"))


# Gauge fix instead of a plumb marker: the wall marker pose stays fully free (floor tilt and marker tilt are absorbed
# there), and the torso joints that share a null direction with the marker are held at their current zero.
# torso_0 is the first joint, so its offset is exactly a tilt of the whole robot about the hip roll axis.
fixed = sorted({l for combo in weak_free for l in combo if l.startswith("torso")})
keep_g = [i for i in range(12) if labels[i] not in fixed]
s_gauge, weak_gauge = svd_report(J[:, keep_g], [labels[i] for i in keep_g], f"marker free, held {fixed}")
Jg = J[:, keep_g] * np.tile([0.5] * 3 + [0.2] * 3, len(J) // 6)[:, None]
cov_g = np.linalg.inv(Jg.T @ Jg)
std_gauge = {labels[i]: float(np.degrees(np.sqrt(cov_g[k, k]))) for k, i in enumerate(keep_g) if i < 6}
print("expected 1-sigma (deg), gauge fixed, 2 mm / 0.5 deg noise:", {k: round(v, 3) for k, v in std_gauge.items()})


def rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# Nonlinear check: tilted floor + non-plumb marker + offsets in all six joints, solved from scratch.
ALL = [q for poses in motions().values() for q in poses]
FREE = [j for j in range(6) if f"torso_{j}" not in fixed]
DELTA_TRUE = D(np.array([0.5, -0.4, 0.3, -0.5, 0.4, -0.3]))
T_tilt = np.eye(4); T_tilt[:3, :3] = rot(D(np.array([1.5, -1.0, 0.0])))       # base tilted 1.5 deg roll, -1 deg pitch
T_wall = T_marker.copy(); T_wall[:3, :3] = rot(D(np.array([0.0, 0.0, 3.0]))) @ rot(D(np.array([2.0, 0, 0]))) @ R_marker
T_marker_true = np.linalg.inv(T_tilt) @ T_wall                                  # marker seen from the tilted base


def predict(delta, Tm):
    return [np.linalg.inv(cam_pose(q, delta)) @ Tm for q in ALL]


def solve(meas, free, fit_marker, Tm0, iters=8):
    def unpack(x):
        d = np.zeros(6); d[free] = x[:len(free)]
        Tm = Tm0.copy()
        if fit_marker:
            Tm[:3, 3] = Tm0[:3, 3] + x[-6:-3]; Tm[:3, :3] = rot(x[-3:]) @ Tm0[:3, :3]
        return d, Tm

    def resid(x):
        d, Tm = unpack(x)
        r = []
        for P, M in zip(predict(d, Tm), meas):
            r += list((P[:3, 3] - M[:3, 3]) / 0.002) + list(np.degrees(rby_log(M[:3, :3].T @ P[:3, :3])) / 0.5)
        return np.array(r)

    x = np.zeros(len(free) + (6 if fit_marker else 0))
    for _ in range(iters):
        r = resid(x); Jn = np.zeros((len(r), len(x)))
        for i in range(len(x)):
            xe = x.copy(); xe[i] += 1e-6; Jn[:, i] = (resid(xe) - r) / 1e-6
        x = x - np.linalg.lstsq(Jn, r, rcond=None)[0]
    d, _ = unpack(x)
    return d, float(np.sqrt(np.mean(resid(x) ** 2)))


def noisy(meas, rng):
    out = []
    for M in meas:
        N = M.copy(); N[:3, 3] += rng.normal(0, 0.002, 3); N[:3, :3] = M[:3, :3] @ rot(D(rng.normal(0, 0.5, 3)))
        out.append(N)
    return out


meas_true = predict(DELTA_TRUE, T_marker_true)
Tm_ref = cam_pose(ALL[0]) @ meas_true[0]          # marker taken from the first view with the current (uncorrected) zeros
fmt = lambda d: {f"torso_{j}": round(float(np.degrees(d[j])), 4) for j in range(6)}
print("\n[nonlinear] true offsets (deg):", fmt(DELTA_TRUE), " floor tilt roll 1.5 / pitch -1.0, marker tilt 2.0, yaw 3.0")
d_g, rms_g = solve(meas_true, FREE, True, Tm_ref)
print("  gauge fixed, marker free, noise-free :", fmt(d_g), "rms", round(rms_g, 5))
d_f, rms_f = solve(meas_true, FREE, False, Tm_ref)
print("  marker frozen at first view          :", fmt(d_f), "rms", round(rms_f, 3))
rng = np.random.default_rng(0)
mc = np.array([solve(noisy(meas_true, rng), FREE, True, Tm_ref)[0] for _ in range(100)])
mc_err = {f"torso_{j}": {"mean": float(np.degrees(mc[:, j].mean() - DELTA_TRUE[j])), "std": float(np.degrees(mc[:, j].std()))}
          for j in FREE}
print("  gauge fixed, 100 noisy runs (err mean / std deg):", {k: (round(v["mean"], 3), round(v["std"], 3)) for k, v in mc_err.items()})

json.dump({"held": fixed, "svd_gauge": s_gauge, "weak_gauge": weak_gauge, "std_gauge_deg": std_gauge,
           "true_deg": fmt(DELTA_TRUE), "floor_tilt_deg": [1.5, -1.0], "marker_tilt_deg": 2.0, "marker_yaw_deg": 3.0,
           "gauge_noise_free_deg": fmt(d_g), "gauge_rms": rms_g, "frozen_marker_deg": fmt(d_f), "frozen_rms": rms_f,
           "gauge_mc100": mc_err},
          open(OUT / "torso_gauge.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
