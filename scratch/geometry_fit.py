"""Would arm geometry (link lengths, joint axis tilts) that differs from the URDF explain why Step 2 and the photos
outside its pose range cannot be fitted together? (2026-11-26, user)

Re-solves a real Step 2 dataset as run (scratch/step2_offline_variants.py, BR_A_as_run). The arm chain from
link_torso_5 to the hand is then rebuilt from the simulator's URDF segments (fixed transform x rotation about the
joint axis) so that, per arm, these can move too: upper arm and forearm length, the two elbow offsets (+-31 mm), and
the J0-J3 axes tilted about their two perpendicular directions. Linearised at the run's solution, the Step 2 photos
and the photos of scratch/check_path_views.py (all seen poses except "ready") with weight w are fitted with the
parameters Step 2 solves plus those groups. It reports the Step 2 residual RMS, the photos' RMS and each photo's x/z
gap, and the geometry it would need. With zero geometry change the rebuilt chain must equal the SDK's (checked).
Simulator model (127.0.0.1, model m, v1.2); nothing in result/ is written.
usage: python scratch/geometry_fit.py <result json> <dataset npz> <step1.5 json> <check_path_views json> <out dir>
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration import observability  # noqa: E402
from core.calibration.calibration_optimizer import prepare_q_full, se3_exp, se3_log  # noqa: E402
from core.calibration.sequences import step2 as step2_module  # noqa: E402

RESULT, DATASET, STEP15, VIEWS, OUT = sys.argv[1:6]
BASE_FREE = {"J0", "J1", "J2", "J4", "J6", "head_pan", "camX", "camY", "camZ"}
WEIGHTS = (0.0, 1.0, 10.0, 100.0)
captured = {}
original = step2_module.step2_weak_directions


def capture(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms, **kw):
    captured.update(opt=opt, q_arm_list=q_arm_list, q_head_list=q_head_list, T_meas_list=T_meas_list,
                    q_arm_offset=np.asarray(q_arm_offset, float), q_head_offset=np.asarray(q_head_offset, float),
                    xi_cam=np.zeros(6) if xi_cam is None else np.asarray(xi_cam, float), active_arms=active_arms)
    return original(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms, **kw)


step2_module.step2_weak_directions = capture
sys.argv = ["step2_offline_variants.py", str(ROOT), RESULT, DATASET, STEP15, OUT, "127.0.0.1:50051", "BR_A_as_run"]
runpy.run_path(str(ROOT / "scratch" / "step2_offline_variants.py"), run_name="__main__")

c = captured
opt = c["opt"]
dyn, model = opt.dyn_model, opt.model
names = list(model.robot_joint_names)
SIDES = ("right", "left")
ARM_IDX = {"right": list(model.right_arm_idx[:7]), "left": list(model.left_arm_idx[:7])}


def rot(axis, angle):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()
    return T


# URDF segments per arm: fixed transform at q = 0 and the joint axis in the child frame.
SEG = {}
for side in SIDES:
    chain = ["link_torso_5"] + [f"link_{side}_arm_{k}" for k in range(7)] + [f"ee_{side}"]
    st = dyn.make_state(chain, names)
    q0 = np.zeros(model.robot_dof)

    def seg(i, q):
        st.set_q(q)
        dyn.compute_forward_kinematics(st)
        return np.asarray(dyn.compute_transformation(st, i, i + 1))
    fixed, axes = [], []
    for i in range(7):
        T0 = seg(i, q0)
        q1 = q0.copy()
        q1[ARM_IDX[side][i]] = 1e-5
        R_rel = np.linalg.inv(T0) @ seg(i, q1)
        a = Rotation.from_matrix(R_rel[:3, :3]).as_rotvec() / 1e-5
        fixed.append(T0), axes.append(a / np.linalg.norm(a))
    SEG[side] = dict(fixed=fixed, axes=axes, ee=seg(7, q0))


def perp(a):
    """Two unit vectors perpendicular to axis a."""
    e = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, e)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


# Geometry parameters per arm: (segment, kind, vector). "len": translation of the fixed transform (mm) in the
# parent frame; "tilt": rotation (deg) inserted before the joint, about a direction perpendicular to its axis.
GEO = []
for side in SIDES:
    s = side[0].upper()
    GEO += [(side, 3, "len", np.array([0, 0, 1.0]), f"{s}upper_len"), (side, 3, "len", np.array([1.0, 0, 0]), f"{s}elbow_x"),
            (side, 4, "len", np.array([0, 0, 1.0]), f"{s}fore_len"), (side, 4, "len", np.array([1.0, 0, 0]), f"{s}fore_x")]
    for j in range(4):
        u, v = perp(SEG[side]["axes"][j])
        GEO += [(side, j, "tilt", u, f"{s}J{j}_tilt_a"), (side, j, "tilt", v, f"{s}J{j}_tilt_b")]


def arm_fk(side, q_arm7, geo):
    """link_torso_5 -> ee for one arm, joint angles in rad, geometry deltas geo (len in mm, tilt in deg)."""
    T = np.eye(4)
    for i in range(7):
        F = SEG[side]["fixed"][i].copy()
        for (sd, seg_i, kind, vec, _), value in zip(GEO, geo):
            if sd == side and seg_i == i and value != 0.0:
                if kind == "len":
                    F[:3, 3] += vec * value * 1e-3
                else:
                    F = F @ rot(vec, np.radians(value))
        T = T @ F @ rot(SEG[side]["axes"][i], q_arm7[i])
    return T @ SEG[side]["ee"]


st_base = dyn.make_state([opt.base_link, "link_torso_5"], names)
weights = observability._noise_weights(opt)


def model_marker(q_arm, q_head, side, qa, qh, xi, geo):
    q_full = prepare_q_full(q_nominal=opt.q_nominal, arm_idx=opt.arm_idx, q_cmd=q_arm, q_offset=qa,
                            head_idx=opt.head_idx, q_head=q_head, q_head_offset=qh)
    st_base.set_q(q_full)
    dyn.compute_forward_kinematics(st_base)
    T_base_t5 = np.asarray(dyn.compute_transformation(st_base, 0, 1))
    T_fk = T_base_t5 @ arm_fk(side, q_full[ARM_IDX[side]], geo)
    T_cam = opt.get_nominal_mount_to_cam() @ se3_exp(xi)   # as QPCalibrationOptimizer.evaluate_sample
    B = opt.get_nominal_ee_to_marker(side)
    return T_base_t5, T_cam, B, np.linalg.inv(T_cam) @ T_fk @ B


labels, scale, slots = observability._parameter_layout(opt, c["q_arm_offset"], c["q_head_offset"], c["active_arms"])
labels, scale, slots = list(labels), list(scale), list(slots)
for k, g in enumerate(GEO):
    labels.append(g[4]), scale.append(1.0), slots.append(("geo", k))
scale = np.array(scale)


def unpack(delta):
    qa, qh, xi = c["q_arm_offset"].copy(), c["q_head_offset"].copy(), c["xi_cam"].copy()
    geo = np.zeros(len(GEO))
    for (kind, k), d, s in zip(slots, delta, scale):
        if kind == "arm":
            qa[k] += d * s
        elif kind == "head":
            qh[k] += d * s
        elif kind == "cam":
            xi[k] += d * s
        else:
            geo[k] += d * s
    return qa, qh, xi, geo


def residuals(q_list, h_list, m_list, delta):
    qa, qh, xi, geo = unpack(delta)
    out = []
    for q_arm, q_head, T_pair in zip(q_list, h_list, m_list):
        for i, side in enumerate(SIDES):
            _, _, _, T_model = model_marker(q_arm, q_head, side, qa, qh, xi, geo)
            out.append(se3_log(np.linalg.inv(T_model) @ np.asarray(T_pair)[i]) * weights)
    return np.concatenate(out)


views, vnames = [], []
for k, path in enumerate(VIEWS.split(",")):      # several check_path_views.json files, comma-separated
    for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]:
        if r.get("seen") and r["pose"] != "ready":
            views.append(r), vnames.append(f"{k}:{r['pose']}")
v_q = [np.asarray(r["q_arm"], float) for r in views]
v_h = [np.asarray(r["q_head"], float) for r in views]
v_m = [np.array([r["markers"]["right"], r["markers"]["left"]], float) for r in views]
step2_res = lambda d: residuals(c["q_arm_list"], c["q_head_list"], c["T_meas_list"], d)  # noqa: E731
view_res = lambda d: residuals(v_q, v_h, v_m, d)  # noqa: E731


def gaps(delta):
    qa, qh, xi, geo = unpack(delta)
    out = []
    for q_arm, q_head, markers in zip(v_q, v_h, v_m):
        e = {}
        for i, side in enumerate(SIDES):
            T_base_t5, T_cam, B, T_model = model_marker(q_arm, q_head, side, qa, qh, xi, geo)
            T_t5 = np.linalg.inv(T_base_t5)
            e[side] = ((T_t5 @ T_cam @ markers[i] @ np.linalg.inv(B))[:3, 3]
                       - (T_t5 @ T_cam @ T_model @ np.linalg.inv(B))[:3, 3]) * 1000.0
        out.append(np.array([e["left"][0], -e["left"][1], e["left"][2]]) - e["right"])
    return np.concatenate(out)


# The rebuilt chain has to reproduce the SDK model exactly before anything else is read from it.
ref = observability._residuals(opt, c["q_arm_list"], c["q_head_list"], c["T_meas_list"], c["q_arm_offset"],
                               c["q_head_offset"], c["xi_cam"], c["active_arms"])
mine = step2_res(np.zeros(len(slots)))
print(f"rebuilt chain vs SDK: max residual difference {np.max(np.abs(ref - mine)):.2e} sigma over {len(ref)}")
# 1e-4 sigma is far below anything read here (the joint axes come from a 1e-5 rad finite difference: ~5e-6 sigma).
assert np.max(np.abs(ref - mine)) < 1e-4, "rebuilt arm chain does not match the SDK model"


def jac(fun, h=1e-4):
    f0 = fun(np.zeros(len(slots)))
    J = np.zeros((len(f0), len(slots)))
    for j in range(len(slots)):
        d = np.zeros(len(slots))
        d[j] = h
        J[:, j] = (fun(d) - f0) / h
    return f0, J


r0, J = jac(step2_res)
rc, Jc = jac(view_res)
g0, G = jac(gaps)
rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
xz = lambda g: [round(float(np.hypot(g[3 * k], g[3 * k + 2])), 2) for k in range(len(vnames))]  # noqa: E731
print(f"photos: {vnames}")
print(f"as solved: Step 2 RMS {rms(r0):.3f} sigma, photos RMS {rms(rc):.3f} sigma, x/z gaps {xz(g0)} mm")
base = {lab for lab in labels if lab[1:] in BASE_FREE or lab in BASE_FREE}
lens = {g[4] for g in GEO if g[2] == "len"}
tilts = {g[4] for g in GEO if g[2] == "tilt"}
SETS = [("as Step 2", base), ("+ link lengths", base | lens), ("+ axis tilts J0-J3", base | tilts),
        ("+ lengths + tilts", base | lens | tilts)]
rows = []
for title, sel in SETS:
    free = [j for j, lab in enumerate(labels) if lab in sel]
    sv = np.linalg.svd(J[:, free], compute_uv=False)
    print(f"\n== {title}: {len(free)} parameters, Step 2 alone condition {sv[0] / sv[-1]:.0f}")
    for w in WEIGHTS:
        # Ridge 0.05 per unit (deg / mm) keeps directions neither data set sees near zero instead of anywhere.
        A = np.vstack([J[:, free], np.sqrt(w) * Jc[:, free], 0.05 * np.eye(len(free))])
        b = np.concatenate([-r0, -np.sqrt(w) * rc, np.zeros(len(free))])
        d_free = np.linalg.lstsq(A, b, rcond=None)[0]
        d = np.zeros(len(slots))
        d[free] = d_free
        r_new, rc_new, g_new = r0 + J @ d, rc + Jc @ d, g0 + G @ d
        top = np.argsort(-np.abs(d))[:10]
        moves = {labels[j]: round(float(d[j]), 3) for j in top if abs(d[j]) > 1e-3}
        print(f"[{title}] w {w:5.0f}: Step 2 RMS {rms(r0):.3f} -> {rms(r_new):.3f}, photos {rms(rc):.3f} -> {rms(rc_new):.3f} "
              f"sigma, x/z gaps {xz(g_new)} mm")
        print("   largest (deg / mm): " + ", ".join(f"{k} {v:+.2f}" for k, v in moves.items()))
        rows.append({"set": title, "w": w, "step2_rms": round(rms(r_new), 3), "photos_rms": round(rms(rc_new), 3),
                     "gaps_xz": xz(g_new), "moves": moves})
Path(OUT, "geometry_fit.json").write_text(json.dumps({"photos": vnames, "rows": rows}, indent=1), encoding="utf-8")

# Nonlinear check of the link-length set: same parameters, no linearisation, the camera position kept inside its
# design range (link_head_2 axes; the camera frame is that mount rotated by rpy -90/0/-90, so mount x = cam z,
# mount y = -cam x, mount z = -cam y), and a weak prior of 5 mm per length so unseen lengths stay near the URDF.
from scipy.optimize import least_squares  # noqa: E402

cons = json.loads(Path(RESULT).read_text(encoding="utf-8"))["camera_mount_constraint"]
off_mm = np.asarray(cons["camera_offset_from_design_mm"], float)
rng_mm = np.asarray(cons["position_range_m"], float) * 1000.0
cam_bounds = {"camZ": (rng_mm[0][0] - off_mm[0], rng_mm[0][1] - off_mm[0]),
              "camX": (off_mm[1] - rng_mm[1][1], off_mm[1] - rng_mm[1][0]),
              "camY": (off_mm[2] - rng_mm[2][1], off_mm[2] - rng_mm[2][0])}
LEN_PRIOR_MM = 5.0
free = [j for j, lab in enumerate(labels) if lab in (base | lens)]
lb = np.full(len(free), -np.inf)
ub = np.full(len(free), np.inf)
for i, j in enumerate(free):
    if labels[j] in cam_bounds:
        lb[i], ub[i] = cam_bounds[labels[j]]
is_len = np.array([labels[j] in lens for j in free])
print(f"\n== nonlinear, link lengths, camera within its design range (cam deltas allowed {cam_bounds})")
for w in (1.0, 10.0):
    def fun(x):
        d = np.zeros(len(slots))
        d[free] = x
        return np.concatenate([step2_res(d), np.sqrt(w) * view_res(d), x[is_len] / LEN_PRIOR_MM])
    sol = least_squares(fun, np.zeros(len(free)), bounds=(lb, ub), x_scale=1.0, diff_step=1e-4, max_nfev=60)
    d = np.zeros(len(slots))
    d[free] = sol.x
    r_new, rc_new, g_new = step2_res(d), view_res(d), gaps(d)
    lens_out = {labels[j]: round(float(sol.x[i]), 2) for i, j in enumerate(free) if is_len[i]}
    cams = {labels[j]: round(float(sol.x[i]), 2) for i, j in enumerate(free) if labels[j] in cam_bounds}
    print(f"[nonlinear lengths] w {w:4.0f}: Step 2 RMS {rms(r0):.3f} -> {rms(r_new):.3f}, photos {rms(rc):.3f} -> "
          f"{rms(rc_new):.3f} sigma, x/z gaps {xz(g_new)} mm ({sol.nfev} evaluations)")
    print(f"   lengths (mm, + = segment shorter than the URDF along -z for *_len): {lens_out}")
    print(f"   camera deltas (mm, camera frame): {cams}")
