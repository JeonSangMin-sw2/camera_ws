"""Search the v1.3 J5-stage ready pose (J4, J5 at ready) for a larger J4 radius that stays in view.

The v1.3 J5 estimate is the angle between the fitted J6-sweep and J4-sweep circle axes; the J4 circle
is small (67 mm) at the current ready pose (J5 = 0), which makes its axis the noisier half. Bending J5
grows it (J5 -30: 120 mm) but turned the plate away from the camera in the simulator (out of the
image by up to 173 px). This scans J4 and J5 at the ready pose and reports, per arm, the J4 radius and
the worst image margin / plate tilt over both sweeps (J6 +-a6, J4 +-a4 about the ready pose).

Camera: torso->camera fitted (rigid, least squares) to a recorded simulator sweep that has the marker
in both the torso frame and the camera frame (the head did not move during it). Kinematics: v1.3 URDF.
Plate: 80 mm, corners projected with the D405 intrinsics. Offline. Approximate (nominal bracket).
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_j5_ready_search.py <scatter run folder> [a6 a4]
"""
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration.calibration_optimizer import make_transform  # noqa: E402

RUN = Path(sys.argv[1]) / "result" / "result_txt"
A6, A4 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (30.0, 30.0)
MARGIN_MIN, TILT_MAX = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (60.0, 40.0)
URDF = ROOT / "archive" / "reports" / "v13_sweep_clearance" / "model_m_v1.3.urdf"
K = np.array(yaml.safe_load((ROOT / "config" / "camera_intrinsics_d405.yaml").read_text(encoding="utf-8"))["camera_matrix"])
HALF = 0.040
CORNERS = np.array([[x, y, 0.0, 1.0] for x in (-HALF, HALF) for y in (-HALF, HALF)])
BRACKET = make_transform([0.067, 0.0, 0.0, 90.0, 0.0, -90.0])

model = rby.Model_M()
names = list(model.robot_joint_names)
dyn = getattr(rby.dynamics, f"Robot_{len(names)}")(rby.dynamics.load_robot_from_urdf_data(URDF.read_text(encoding="utf-8"), "base"))
LINKS = ["link_torso_5", "link_right_arm_4", "link_left_arm_4", "ee_right", "ee_left"]
state = dyn.make_state(LINKS, names)
ready = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v1.3"]["joint"]["wrist_pitch"]


def fit_camera(side):
    """torso_5 -> camera from one recorded sweep (Kabsch on the marker positions)."""
    torso, cam = [], []
    for line in (RUN / f"sweep_points_{side}_wrist_pitch_v13_joint_B_axis_4.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "NEW" not in line:
            v = [float(x) for x in line.split(",") if x.strip()]
            if len(v) >= 26:
                cam.append(v[1:4]); torso.append(v[4:7])
    P, Q = np.array(cam) / 1000.0, np.array(torso) / 1000.0
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = qc - R @ pc
    residual = np.linalg.norm((P @ R.T + T[:3, 3]) - Q, axis=1).max() * 1000.0
    return T, residual


def marker_in_camera(q, side, T_torso_cam):
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    marker = dyn.compute_transformation(state, 0, LINKS.index(f"ee_{side}")) @ BRACKET
    return np.linalg.inv(T_torso_cam) @ marker


def view(T):
    p = (K @ (T @ CORNERS.T)[:3]).T
    p = p[:, :2] / p[:, 2:3]
    margin = min(p[:, 0].min(), 1280 - p[:, 0].max(), p[:, 1].min(), 720 - p[:, 1].max())
    ray = T[:3, 3] / np.linalg.norm(T[:3, 3])
    return margin, float(np.degrees(np.arccos(abs(T[:3, 2] @ ray))))


def j4_radius(q, side):
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    T4 = dyn.compute_transformation(state, 0, LINKS.index(f"link_{side}_arm_4"))
    axis = T4[:3, :3] @ np.array([0.0, 0.0, 1.0])
    d = (dyn.compute_transformation(state, 0, LINKS.index(f"ee_{side}")) @ BRACKET)[:3, 3] - T4[:3, 3]
    return float(np.linalg.norm(d - (d @ axis) * axis)) * 1000.0


rows = []
for side in ("right", "left"):
    T_cam, residual = fit_camera(side)
    print(f"{side}: camera fitted from the recorded sweep, worst residual {residual:.2f} mm")
    idx = list(getattr(model, f"{side}_arm_idx"))
    sign = 1.0 if side == "right" else -1.0  # J4 mirrors between the arms (check_calib pose)
    for j4 in np.arange(-90.0, 90.01, 15.0):
        for j5 in np.arange(-45.0, 45.01, 7.5):
            q = np.zeros(len(names))
            q[idx] = np.radians(ready[f"{side}_arm"])
            q[idx[4]] = np.radians(sign * j4)
            q[idx[5]] = np.radians(j5)
            worst_margin, worst_tilt = np.inf, 0.0
            for joint, arc in ((6, A6), (4, A4)):
                for d in np.linspace(-arc, arc, 13):
                    qs = q.copy(); qs[idx[joint]] += np.radians(d)
                    m, t = view(marker_in_camera(qs, side, T_cam))
                    worst_margin, worst_tilt = min(worst_margin, m), max(worst_tilt, t)
            rows.append((side, j4, j5, j4_radius(q, side), worst_margin, worst_tilt))

print(f"\nJ6 +-{A6:g}, J4 +-{A4:g}. Ready J4/J5 (deg, J4 given for the right arm, mirrored on the left) "
      f"where both arms keep margin >= {MARGIN_MIN:g} px and tilt <= {TILT_MAX:g} deg, best J4 radius first:")
ok = {}
for side, j4, j5, r, m, t in rows:
    ok.setdefault((j4, j5), []).append((side, r, m, t))
good = [(min(v[1] for v in vals), key, vals) for key, vals in ok.items()
        if len(vals) == 2 and all(v[2] >= MARGIN_MIN and v[3] <= TILT_MAX for v in vals)]
for r, (j4, j5), vals in sorted(good, reverse=True)[:12]:
    print(f"  J4 {j4:+6.1f} J5 {j5:+6.1f}: J4 radius {r:5.1f} mm | " +
          " | ".join(f"{s} margin {m:4.0f} px tilt {t:4.1f}" for s, _, m, t in vals))
current = ok.get((0.0, 0.0))
print("  current ready (J4 0, J5 0): " + " | ".join(f"{s} r {r:5.1f} margin {m:4.0f} tilt {t:4.1f}" for s, r, m, t in current))

# Check against the recorded sweep: model image position of the plate along the J6 sweep at the
# current ready pose (compare with v13_sweep_view_margin.py on the same run).
for side in ("right", "left"):
    T_cam, _ = fit_camera(side)
    idx = list(getattr(model, f"{side}_arm_idx"))
    q = np.zeros(len(names)); q[idx] = np.radians(ready[f"{side}_arm"])
    for d in (-22.5, 0.0, 22.5):
        qs = q.copy(); qs[idx[6]] += np.radians(d)
        T = marker_in_camera(qs, side, T_cam)
        p = (K @ (T @ CORNERS.T)[:3]).T; p = p[:, :2] / p[:, 2:3]
        print(f"  check {side} J6 {d:+5.1f}: margin {view(T)[0]:5.0f} px, plate centre u {p[:, 0].mean():4.0f} v {p[:, 1].mean():4.0f}")

# --j6-shift DEG: also report the current ready pose with J6 turned by DEG (right; mirrored left),
# i.e. the J6 sweep centred off the image's bottom edge.
if "--j6-shift" in sys.argv:
    shift = float(sys.argv[sys.argv.index("--j6-shift") + 1])
    for side in ("right", "left"):
        T_cam, _ = fit_camera(side)
        idx = list(getattr(model, f"{side}_arm_idx"))
        q = np.zeros(len(names)); q[idx] = np.radians(ready[f"{side}_arm"])
        q[idx[6]] += np.radians(shift if side == "right" else -shift)
        worst = {}
        for joint, arc in ((6, A6), (4, A4)):
            vals = []
            for d in np.linspace(-arc, arc, 19):
                qs = q.copy(); qs[idx[joint]] += np.radians(d)
                vals.append(view(marker_in_camera(qs, side, T_cam)))
            worst[joint] = (min(v[0] for v in vals), max(v[1] for v in vals))
        print(f"  J6 shifted {shift:+.1f} ({side}): J4 radius {j4_radius(q, side):5.1f} mm | J6 sweep margin {worst[6][0]:4.0f} px "
              f"tilt {worst[6][1]:4.1f} | J4 sweep margin {worst[4][0]:4.0f} px tilt {worst[4][1]:4.1f}")
