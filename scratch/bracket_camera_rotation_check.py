"""Would freeing the marker-bracket rotation (and position) or the camera rotation in Step 2 explain the check-pose
gap? 2026-11-26 (user): maybe the brackets / camera are off by more than allowed and the joints absorb it.

Re-solves a real Step 2 dataset as run (scratch/step2_offline_variants.py, variant BR_A_as_run), then, linearised at
that solution, lets extra parameters move together with what Step 2 already solves (J0/J1/J2/J4/J6, head pan,
camera position): bracket roll/pitch per arm, bracket position per arm, camera rotation rx/ry/rz. Bracket yaw stays
fixed (degenerate with J6) and head tilt stays anchored (degenerate with camera rx). For each set and check-view
weight w it reports the Step 2 residual RMS, the check-view residual and the x/z gap the solution would leave.
Simulator model (127.0.0.1, model m, v1.2); nothing in result/ is written.
usage: python scratch/bracket_camera_rotation_check.py <result json> <dataset npz> <step1.5 json> <out dir>
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration import observability  # noqa: E402
from core.calibration.calibration_optimizer import prepare_q_full  # noqa: E402
from core.calibration.sequences import step2 as step2_module  # noqa: E402

RESULT, DATASET, STEP15, OUT = sys.argv[1:5]
BASE_FREE = {"J0", "J1", "J2", "J4", "J6", "head_pan", "camX", "camY", "camZ"}
WEIGHTS = (0.0, 1.0, 10.0)
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
labels, scale, slots = observability._parameter_layout(opt, c["q_arm_offset"], c["q_head_offset"], c["active_arms"])
labels, scale, slots = list(labels), list(scale), list(slots)
# Extra parameters: camera rx/ry (rz is already in the layout), bracket roll/pitch and x/y/z per arm.
for k, name in ((0, "camRx"), (1, "camRy")):
    if ("cam", k) not in slots:
        labels.append(name), scale.append(np.pi / 180.0), slots.append(("cam", k))
for side in ("right", "left"):
    for k, name in ((3, "roll"), (4, "pitch")):
        labels.append(f"{side[0].upper()}br_{name}"), scale.append(1.0), slots.append(("bracket", (side, k)))
    for k, name in ((0, "x"), (1, "y"), (2, "z")):
        labels.append(f"{side[0].upper()}br_{name}"), scale.append(1e-3), slots.append(("bracket", (side, k)))
scale = np.array(scale)
bracket0 = {side: list(opt.ee_to_marker_nom[side]) for side in ("right", "left")}
print("brackets as run:", bracket0)


def unpack(delta):
    qa, qh, xi = c["q_arm_offset"].copy(), c["q_head_offset"].copy(), c["xi_cam"].copy()
    br = {side: list(v) for side, v in bracket0.items()}
    for (kind, k), d, s in zip(slots, delta, scale):
        if kind == "arm":
            qa[k] += d * s
        elif kind == "head":
            qh[k] += d * s
        elif kind == "cam":
            xi[k] += d * s
        else:
            side, idx = k
            br[side][idx] += d * s
    return qa, qh, xi, br


def with_brackets(br, fn):
    opt.ee_to_marker_nom.update(br)
    try:
        return fn()
    finally:
        opt.ee_to_marker_nom.update(bracket0)


view = json.loads(Path(RESULT).read_text(encoding="utf-8"))["check_pose_view"]
v_q, v_h = [np.asarray(view["q_arm"], float)], [np.asarray(view["q_head"], float)]
v_m = [np.array([view["markers"]["right"], view["markers"]["left"]], float)]


def step2_res(delta):
    qa, qh, xi, br = unpack(delta)
    return with_brackets(br, lambda: observability._residuals(opt, c["q_arm_list"], c["q_head_list"], c["T_meas_list"],
                                                              qa, qh, xi, c["active_arms"]))


def view_res(delta):
    qa, qh, xi, br = unpack(delta)
    return with_brackets(br, lambda: observability._residuals(opt, v_q, v_h, v_m, qa, qh, xi, c["active_arms"]))


def gaps(delta):
    """Left minus mirrored right of (seen flange - model flange) in link_torso_5, mm, at the check view."""
    qa, qh, xi, br = unpack(delta)

    def run():
        q_full = prepare_q_full(q_nominal=opt.q_nominal, arm_idx=opt.arm_idx, q_cmd=v_q[0], q_offset=qa,
                                head_idx=opt.head_idx, q_head=v_h[0], q_head_offset=qh)
        state = opt.dyn_model.make_state(["link_torso_5", opt.base_link], opt.model.robot_joint_names)
        state.set_q(q_full)
        opt.dyn_model.compute_forward_kinematics(state)
        T_t5 = np.asarray(opt.dyn_model.compute_transformation(state, 0, 1))
        e = {}
        for i, side in enumerate(("right", "left")):
            _, T_cam, B, T_model = opt.evaluate_sample(v_q[0], v_h[0], side, qa, qh, xi)
            seen = T_t5 @ T_cam @ v_m[0][i] @ np.linalg.inv(B)
            model = T_t5 @ T_cam @ T_model @ np.linalg.inv(B)
            e[side] = (seen[:3, 3] - model[:3, 3]) * 1000.0
        return np.array([e["left"][0], -e["left"][1], e["left"][2]]) - e["right"]
    return with_brackets(br, run)


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
print(f"\nStep 2: {len(r0)} weighted residuals, RMS {rms(r0):.3f} sigma; check view weighted RMS {rms(rc):.3f} sigma; "
      f"gap as solved x {g0[0]:+.2f} y {g0[1]:+.2f} z {g0[2]:+.2f} mm -> x/z {np.hypot(g0[0], g0[2]):.2f} mm")

base = {lab for lab in labels if lab[1:] in BASE_FREE or lab in BASE_FREE}
br_rot = {lab for lab in labels if "br_roll" in lab or "br_pitch" in lab}
br_pos = {lab for lab in labels if lab[2:] in ("br_x", "br_y", "br_z")}
cam_rot = {"camRx", "camRy", "camRz"}
SETS = [("as Step 2", base), ("+ bracket roll/pitch", base | br_rot), ("+ camera rotation", base | cam_rot),
        ("+ bracket rot + camera rot", base | br_rot | cam_rot),
        ("+ bracket rot/pos + camera rot", base | br_rot | br_pos | cam_rot)]
rows = []
for title, names in SETS:
    free = [j for j, lab in enumerate(labels) if lab in names]
    for w in WEIGHTS:
        A = np.vstack([J[:, free], np.sqrt(w) * Jc[:, free], 1e-3 * np.eye(len(free))])
        b = np.concatenate([-r0, -np.sqrt(w) * rc, np.zeros(len(free))])
        d_free = np.linalg.lstsq(A, b, rcond=None)[0]
        d = np.zeros(len(slots))
        d[free] = d_free
        r_new, rc_new, g_new = r0 + J @ d, rc + Jc @ d, g0 + G @ d
        # The same camera / head / brackets but the run's arm offsets: what this model says the robot standing at the
        # check pose now looks like. The operator sees about x -2, z -3 mm (right hand forward and up) there.
        d_now = d.copy()
        d_now[[j for j, (kind, _) in enumerate(slots) if kind == "arm"]] = 0.0
        g_now = g0 + G @ d_now
        top = np.argsort(-np.abs(d))[:8]
        moves = ", ".join(f"{labels[j]} {d[j]:+.2f}" for j in top if abs(d[j]) > 1e-3)
        print(f"\n[{title}] w {w:4.0f}: Step 2 RMS {rms(r0):.3f} -> {rms(r_new):.3f} sigma, check view {rms(rc):.3f} -> "
              f"{rms(rc_new):.3f} sigma, gap x {g_new[0]:+.2f} z {g_new[2]:+.2f} -> {np.hypot(g_new[0], g_new[2]):.2f} mm")
        print(f"   this model's view of the robot as calibrated now: x {g_now[0]:+.2f} z {g_now[2]:+.2f} -> "
              f"{np.hypot(g_now[0], g_now[2]):.2f} mm")
        print(f"   moves (deg / mm): {moves}")
        rows.append({"set": title, "w": w, "step2_rms": round(rms(r_new), 3), "view_rms": round(rms(rc_new), 3),
                     "gap_x": round(float(g_new[0]), 2), "gap_z": round(float(g_new[2]), 2),
                     "gap_xz": round(float(np.hypot(g_new[0], g_new[2])), 2),
                     "now_x": round(float(g_now[0]), 2), "now_z": round(float(g_now[2]), 2),
                     "moves": {labels[j]: round(float(d[j]), 3) for j in top if abs(d[j]) > 1e-3}})
Path(OUT, "bracket_camera_rotation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

# How well Step 2's own data pins each parameter of a set: 1-sigma (deg / mm) from (J^T J)^-1 of the weighted
# residuals, and the strongest correlation with another parameter. |corr| near 1 means the two trade off freely.
print("\nconditioning (Step 2 data only):")
for title, names in SETS[:4]:
    free = [j for j, lab in enumerate(labels) if lab in names]
    A = J[:, free]
    sv = np.linalg.svd(A, compute_uv=False)
    cov = np.linalg.pinv(A.T @ A)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    print(f"[{title}] condition {sv[0] / sv[-1]:.0f}, smallest singular value {sv[-1]:.2f}")
    for i, j in enumerate(free):
        if labels[j] in names - base or title == "as Step 2":
            k = int(np.argmax(np.abs(corr[i])))
            print(f"   {labels[j]:10s} 1-sigma {std[i]:.3f}   strongest corr {corr[i, k]:+.2f} with {labels[free[k]]}")
