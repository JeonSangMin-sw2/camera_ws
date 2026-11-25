"""Can Step 2's weak offset combinations explain the saved check-pose views? Re-solves a real Step 2 dataset with
scratch/step2_offline_variants.py (as run), then, linearised at that solution, adds the check-view photos
(result JSON check_pose_view: first view, other head angles, facing view) to the Step 2 residuals with weight w
and reports, per w: the Step 2 residual RMS over the original poses, the check-view residual, the left/right
flange gap the new solution would leave at each view, and which offsets move. Only what Step 2 solves freely
moves: J0/J1/J2/J4/J6, head pan, camera position (J3/J5 and head tilt are anchored, camera angles fixed).
Simulator model (127.0.0.1, model m, v1.2); nothing in result/ is written.
usage: python scratch/weak_direction_check_views.py <result json> <dataset npz> <step1.5 json> <out dir>
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
FREE = {"J0", "J1", "J2", "J4", "J6", "head_pan", "camX", "camY", "camZ"}
WEIGHTS = (0.0, 1.0, 10.0, 100.0)
captured = {}
original = step2_module.step2_weak_directions


def capture(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms, **kw):
    captured.update(opt=opt, q_arm_list=q_arm_list, q_head_list=q_head_list, T_meas_list=T_meas_list,
                    q_arm_offset=np.asarray(q_arm_offset, float), q_head_offset=np.asarray(q_head_offset, float),
                    xi_cam=np.zeros(6) if xi_cam is None else np.asarray(xi_cam, float), active_arms=active_arms)
    return original(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms, **kw)


step2_module.step2_weak_directions = capture
sys.argv = ["step2_offline_variants.py", str(ROOT), RESULT, DATASET, STEP15, OUT, "127.0.0.1:50051", "W_as_run_2e5"]
runpy.run_path(str(ROOT / "scratch" / "step2_offline_variants.py"), run_name="__main__")

c = captured
opt = c["opt"]
labels, scale, slots = observability._parameter_layout(opt, c["q_arm_offset"], c["q_head_offset"], c["active_arms"])
free = [j for j, name in enumerate(labels) if name[1:] in FREE or name in FREE]


def unpack(delta):
    qa, qh, xi = c["q_arm_offset"].copy(), c["q_head_offset"].copy(), c["xi_cam"].copy()
    for (kind, k), d, s in zip(slots, delta, scale):
        if kind == "arm":
            qa[k] += d * s
        elif kind == "head":
            qh[k] += d * s
        else:
            xi[k] += d * s
    return qa, qh, xi


view = json.loads(Path(RESULT).read_text(encoding="utf-8"))["check_pose_view"]
views = [("check head 58", view)] + [(f"check head {v['head_deg'][1]:.0f}", v) for v in view.get("other_views") or []]
if view.get("facing_view"):
    views.append(("facing head 35", view["facing_view"]))
views = [(t, v) for t, v in views if v.get("seen")]
v_q = [np.asarray(v["q_arm"], float) for _, v in views]
v_h = [np.asarray(v["q_head"], float) for _, v in views]
v_m = [np.array([v["markers"]["right"], v["markers"]["left"]], float) for _, v in views]


def step2_res(delta):
    return observability._residuals(opt, c["q_arm_list"], c["q_head_list"], c["T_meas_list"], *unpack(delta),
                                    c["active_arms"])


def view_res(delta):
    return observability._residuals(opt, v_q, v_h, v_m, *unpack(delta), c["active_arms"])


def gaps(delta):
    """Per view, left minus mirrored right of (seen flange - model flange) in link_torso_5, mm: the gap the
    robot would show re-commanded with this solution (the model flanges are made symmetric by the command)."""
    qa, qh, xi = unpack(delta)
    out = []
    for q_arm, q_head, markers in zip(v_q, v_h, v_m):
        q_full = prepare_q_full(q_nominal=opt.q_nominal, arm_idx=opt.arm_idx, q_cmd=q_arm, q_offset=qa,
                                head_idx=opt.head_idx, q_head=q_head, q_head_offset=qh)
        state = opt.dyn_model.make_state(["link_torso_5", opt.base_link], opt.model.robot_joint_names)
        state.set_q(q_full)
        opt.dyn_model.compute_forward_kinematics(state)
        T_t5 = np.asarray(opt.dyn_model.compute_transformation(state, 0, 1))
        e = {}
        for i, side in enumerate(("right", "left")):
            _, T_cam, B, T_model = opt.evaluate_sample(q_arm, q_head, side, qa, qh, xi)
            seen = T_t5 @ T_cam @ markers[i] @ np.linalg.inv(B)
            model = T_t5 @ T_cam @ T_model @ np.linalg.inv(B)
            e[side] = (seen[:3, 3] - model[:3, 3]) * 1000.0
        out.append(np.array([e["left"][0], -e["left"][1], e["left"][2]]) - e["right"])
    return np.concatenate(out)


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
print(f"\nStep 2: {len(r0)} weighted residuals, RMS {np.sqrt(np.mean(r0 ** 2)):.3f} sigma; check views "
      f"{len(views)} ({', '.join(t for t, _ in views)}), weighted RMS {np.sqrt(np.mean(rc ** 2)):.3f} sigma")
for k, (tag, _) in enumerate(views):
    print(f"  {tag:15s} gap as solved x {g0[3*k]:+.2f} y {g0[3*k+1]:+.2f} z {g0[3*k+2]:+.2f} mm")
print("free parameters:", [labels[j] for j in free])
for w in WEIGHTS:
    A = np.vstack([J[:, free], np.sqrt(w) * Jc[:, free]])
    b = -np.concatenate([r0, np.sqrt(w) * rc])
    # A tiny ridge keeps exactly unobservable combinations at zero instead of anywhere.
    A = np.vstack([A, 1e-3 * np.eye(len(free))])
    b = np.concatenate([b, np.zeros(len(free))])
    d_free = np.linalg.lstsq(A, b, rcond=None)[0]
    d = np.zeros(len(slots))
    d[free] = d_free
    r_new, rc_new, g_new = r0 + J @ d, rc + Jc @ d, g0 + G @ d
    top = np.argsort(-np.abs(d))[:7]
    print(f"\nw {w:5.0f}: Step 2 RMS {np.sqrt(np.mean(r0 ** 2)):.3f} -> {np.sqrt(np.mean(r_new ** 2)):.3f} sigma, "
          f"check views {np.sqrt(np.mean(rc ** 2)):.3f} -> {np.sqrt(np.mean(rc_new ** 2)):.3f} sigma")
    print("   moves (deg / mm): " + ", ".join(f"{labels[j]} {d[j]:+.2f}" for j in top))
    for k, (tag, _) in enumerate(views):
        gk = g_new[3 * k:3 * k + 3]
        print(f"   {tag:15s} gap x {gk[0]:+.2f} y {gk[1]:+.2f} z {gk[2]:+.2f} -> {np.linalg.norm(gk):.2f} mm "
              f"(as solved {np.linalg.norm(g0[3*k:3*k+3]):.2f})")
