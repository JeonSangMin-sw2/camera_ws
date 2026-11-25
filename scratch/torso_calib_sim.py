"""Torso zero calibration run on the SIMULATOR (127.0.0.1 only).

The sim only moves the robot and reports encoder angles. The wall marker view is built from those angles by a virtual
sensor (true torso offsets, tilted floor, tilted marker), like core/calibration/simulation.py does for the arms; no
image detection. Solved with the marker pose free and torso_0 held at its current zero (floor tilt absorbed).
usage: python scratch/torso_calib_sim.py <out dir> [--from <record json>]   (--from re-solves without the sim)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
D = np.deg2rad
URDF = ROOT / "archive" / "reports" / "v13_sweep_clearance" / "model_m_v1.3.urdf"   # same kinematics as the v1.3 sim
L = 0.35
THETA0 = D(30.0)
ARM_R = D(np.array([0, -15, 0, -20, 0, 0, 0])); ARM_L = D(np.array([0, 15, 0, -20, 0, 0, 0]))

names = list(rby.Model_M().robot_joint_names)
dyn = getattr(rby.dynamics, f"Robot_{len(names)}")(rby.dynamics.load_robot_from_urdf_data(URDF.read_text(encoding="utf-8"), "base"))
state = dyn.make_state(["base", "link_head_2"], names)
TORSO = [names.index(f"torso_{i}") for i in range(6)]
R_H2C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
K = np.array([[651.83, 0, 632.69], [0, 651.23, 360.67], [0, 0, 1]])   # config/camera_intrinsics.yaml (D405)
W, H = 1280, 720
TAG = 0.10

DELTA_TRUE = D(np.array([0.5, -0.4, 0.3, -0.5, 0.4, -0.3]))
FLOOR_TILT_DEG = np.array([1.5, -1.0, 0.0])
MARKER_TILT_DEG, MARKER_YAW_DEG = 2.0, 3.0
NOISE = {"upper_sim 0.1mm/0deg": (0.0001, 0.0), "upper_sim 0.1mm/0.37deg": (0.0001, 0.37), "assumed 2mm/0.5deg": (0.002, 0.5)}


def rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; S = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * S + (1 - np.cos(th)) * S @ S


def log3(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def cam_pose(q_full, delta=np.zeros(6)):
    q = np.array(q_full, float).copy()
    q[TORSO] += delta
    state.set_q(q); dyn.compute_forward_kinematics(state)
    T = np.array(dyn.compute_transformation(state, 0, 1))
    T[:3, :3] = T[:3, :3] @ R_H2C
    return T


def bent(t, dx=0.0):
    if dx == 0.0:
        return np.array([0, t, -2 * t, t, 0, 0.0])
    z = 2 * L * np.cos(t)
    r = np.hypot(dx, z); a2 = -2 * np.arccos(r / (2 * L)); a1 = np.arctan2(dx, z) - a2 / 2
    return np.array([0, a1, a2, -(a1 + a2), 0, 0.0])


def motions():
    s = np.linspace(-1, 1, 21)
    return {
        "M1 waist pitch": [bent(THETA0) + np.array([0, 0, 0, D(15) * v, 0, 0]) for v in s],
        "M2 waist roll": [bent(THETA0) + np.array([0, 0, 0, 0, D(15) * v, 0]) for v in s],
        "M3 waist yaw": [bent(THETA0) + np.array([0, 0, 0, 0, 0, D(20) * v]) for v in s],
        "M4 vertical": [bent(THETA0 + D(15) * v) for v in s],
        "M5 fore-aft": [bent(THETA0, dx=0.08 * v) for v in s],
        "M6 hip roll": [bent(THETA0) + np.array([D(10) * v, 0, 0, 0, 0, 0]) for v in s],
    }


def full_q(torso):
    q = np.zeros(len(names)); q[TORSO] = torso
    return q


def record_on_sim():
    from core.robot.robot_core import connect_robot_session
    lo = np.array(dyn.get_limit_q_lower(state))[TORSO]; hi = np.array(dyn.get_limit_q_upper(state))[TORSO]
    for name, poses in motions().items():
        for p in poses:
            assert np.all(p > lo + D(2)) and np.all(p < hi - D(2)), f"{name} pose outside limits: {np.degrees(p)}"
    robot = connect_robot_session("127.0.0.1:50051", "m", include_head=True, simulated=True, log_callback=print)

    def move(torso, t):
        body = (rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(torso))
                .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(ARM_R))
                .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(ARM_L)))
        cmd = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body)
                                                    .set_head_command(rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.zeros(2))))
        return robot.send_command(cmd, 10).get().finish_code

    rec = []; t0 = time.time()
    for name, poses in motions().items():
        for k, p in enumerate(poses):
            fc = move(p, 3.0 if k == 0 else 0.6)
            time.sleep(0.3)
            q = np.array(robot.get_state().position)
            rec.append({"motion": name, "cmd": p.tolist(), "q": q.tolist(), "finish": str(fc), "t": time.time() - t0})
            if fc != rby.RobotCommandFeedback.FinishCode.Ok:
                print("[WARN]", name, k, fc)
        print(f"{name}: done at {time.time() - t0:.0f} s, tracking max {np.degrees(np.max([np.abs(np.array(r['q'])[TORSO] - r['cmd']) for r in rec if r['motion'] == name])):.3f} deg")
    move(np.zeros(6), 4.0)
    return rec


rec = json.load(open(sys.argv[sys.argv.index("--from") + 1], encoding="utf-8")) if "--from" in sys.argv else record_on_sim()
json.dump(rec, open(OUT / "torso_calib_record.json", "w", encoding="utf-8"))

# Wall marker: 1.5 m ahead of the nominal start camera, plumb in the world, then seen from a tilted base.
T_cam0 = cam_pose(full_q(bent(THETA0)))
T_wall = np.eye(4); T_wall[:3, 3] = T_cam0[:3, 3] + np.array([1.5, 0, 0])
T_wall[:3, :3] = rot(D(np.array([0, 0, MARKER_YAW_DEG]))) @ rot(D(np.array([MARKER_TILT_DEG, 0, 0]))) \
    @ np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float).T
T_tilt = np.eye(4); T_tilt[:3, :3] = rot(D(FLOOR_TILT_DEG))
T_marker_true = np.linalg.inv(T_tilt) @ T_wall

Q = [np.array(r["q"]) for r in rec]
meas_true = [np.linalg.inv(cam_pose(q, DELTA_TRUE)) @ T_marker_true for q in Q]
corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * TAG / 2


def margin_px(M):
    P = (M[:3, :3] @ corners.T).T + M[:3, 3]
    uv = (K @ P.T).T; uv = uv[:, :2] / uv[:, 2:]
    return float(min(uv[:, 0].min(), W - uv[:, 0].max(), uv[:, 1].min(), H - uv[:, 1].max()))


margins = np.array([margin_px(M) for M in meas_true])
seen = [i for i in range(len(Q)) if margins[i] > 20]
print(f"\nposes {len(Q)}, marker fully in view (20 px margin) {len(seen)}, worst margin {margins.min():.0f} px")
for name in motions():
    idx = [i for i, r in enumerate(rec) if r["motion"] == name]
    print(f"  {name}: in view {sum(margins[i] > 20 for i in idx)}/{len(idx)}, min margin {margins[idx].min():.0f} px")

FREE = [1, 2, 3, 4, 5]       # torso_0 held: its offset is exactly a tilt of the whole robot


def solve(meas, idx, sig_m, sig_deg, iters=8):
    sig_deg = max(sig_deg, 0.01)
    Tm0 = cam_pose(Q[idx[0]]) @ meas[idx[0]]     # initial marker from the first view with the current zeros

    def unpack(x):
        d = np.zeros(6); d[FREE] = x[:5]
        Tm = Tm0.copy(); Tm[:3, 3] += x[5:8]; Tm[:3, :3] = rot(x[8:]) @ Tm0[:3, :3]
        return d, Tm

    def resid(x):
        d, Tm = unpack(x)
        r = []
        for i in idx:
            P = np.linalg.inv(cam_pose(Q[i], d)) @ Tm; M = meas[i]
            r += list((P[:3, 3] - M[:3, 3]) / sig_m) + list(np.degrees(log3(M[:3, :3].T @ P[:3, :3])) / sig_deg)
        return np.array(r)

    x = np.zeros(11)
    for _ in range(iters):
        r = resid(x); Jn = np.zeros((len(r), 11))
        for i in range(11):
            xe = x.copy(); xe[i] += 1e-6; Jn[:, i] = (resid(xe) - r) / 1e-6
        x = x - np.linalg.lstsq(Jn, r, rcond=None)[0]
    return unpack(x)[0]


def add_noise(meas, sig_m, sig_deg, rng):
    out = []
    for M in meas:
        N = M.copy(); N[:3, 3] += rng.normal(0, sig_m, 3)
        if sig_deg > 0:
            N[:3, :3] = M[:3, :3] @ rot(D(rng.normal(0, sig_deg, 3)))
        out.append(N)
    return out


err0 = np.degrees(solve(meas_true, seen, 0.0001, 0.01) - DELTA_TRUE)
print("noise-free error (deg):", {f"torso_{j}": round(float(err0[j]), 5) for j in FREE})
result = {"poses": len(Q), "in_view": len(seen), "worst_margin_px": float(margins.min()), "noise_free_err_deg": err0.tolist(),
          "true_deg": np.degrees(DELTA_TRUE).tolist(), "floor_tilt_deg": FLOOR_TILT_DEG.tolist(),
          "marker_tilt_deg": MARKER_TILT_DEG, "marker_yaw_deg": MARKER_YAW_DEG, "duration_s": rec[-1]["t"], "noise": {}}
rng = np.random.default_rng(20261119)
for label, (sm, sd) in NOISE.items():
    E = np.degrees(np.array([solve(add_noise(meas_true, sm, sd, rng), seen, sm, sd) - DELTA_TRUE for _ in range(100)]))[:, FREE]
    run_max = np.abs(E).max(1)
    result["noise"][label] = {"std_deg": dict(zip([f"torso_{j}" for j in FREE], E.std(0).tolist())),
                              "mean_deg": dict(zip([f"torso_{j}" for j in FREE], E.mean(0).tolist())),
                              "run_max_median_deg": float(np.median(run_max)), "run_max_p95_deg": float(np.percentile(run_max, 95))}
    print(f"[{label}] std (deg):", {f"torso_{j}": round(float(v), 3) for j, v in zip(FREE, E.std(0))},
          f" per-run max |err| median {np.median(run_max):.3f}, p95 {np.percentile(run_max, 95):.3f}")
json.dump(result, open(OUT / "torso_calib_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
