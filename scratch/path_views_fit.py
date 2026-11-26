"""Can joint offsets fit Step 2 and the photos taken outside its pose range together? (2026-11-26, user)

Re-solves a real Step 2 dataset as run (scratch/step2_offline_variants.py, variant BR_A_as_run), then, linearised at
that solution, adds the photos of scratch/check_path_views.py (every seen pose except "ready") to the Step 2
residuals with weight w. Parameter sets: what Step 2 solves (J0/J1/J2/J4/J6, head pan, camera position), plus J3/J5
(held at Step 1 in v1.2), plus the bracket roll/pitch per arm, plus both. For each set and w it reports the Step 2
residual RMS, the photos' residual RMS and the x/z gap each photo would keep. If a set reaches small gaps for a
small Step 2 cost, the Step 2 poses just did not see that direction; if the Step 2 cost grows a lot, joint offsets
cannot make both regions right (something outside the model).
Simulator model (127.0.0.1, model m, v1.2); nothing in result/ is written.
usage: python scratch/path_views_fit.py <result json> <dataset npz> <step1.5 json> <check_path_views json> <out dir>
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
labels, scale, slots = observability._parameter_layout(opt, c["q_arm_offset"], c["q_head_offset"], c["active_arms"])
labels, scale, slots = list(labels), list(scale), list(slots)
for side in ("right", "left"):
    for k, name in ((3, "roll"), (4, "pitch")):
        labels.append(f"{side[0].upper()}br_{name}"), scale.append(1.0), slots.append(("bracket", (side, k)))
scale = np.array(scale)
bracket0 = {side: list(opt.ee_to_marker_nom[side]) for side in ("right", "left")}

rows_in = json.loads(Path(VIEWS).read_text(encoding="utf-8"))["rows"]
views = [r for r in rows_in if r.get("seen") and r["pose"] != "ready"]
names = [r["pose"] for r in views]
v_q = [np.asarray(r["q_arm"], float) for r in views]
v_h = [np.asarray(r["q_head"], float) for r in views]
v_m = [np.array([r["markers"]["right"], r["markers"]["left"]], float) for r in views]


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
            br[k[0]][k[1]] += d * s
    return qa, qh, xi, br


def with_brackets(br, fn):
    opt.ee_to_marker_nom.update(br)
    try:
        return fn()
    finally:
        opt.ee_to_marker_nom.update(bracket0)


def step2_res(delta):
    qa, qh, xi, br = unpack(delta)
    return with_brackets(br, lambda: observability._residuals(opt, c["q_arm_list"], c["q_head_list"], c["T_meas_list"],
                                                              qa, qh, xi, c["active_arms"]))


def view_res(delta):
    qa, qh, xi, br = unpack(delta)
    return with_brackets(br, lambda: observability._residuals(opt, v_q, v_h, v_m, qa, qh, xi, c["active_arms"]))


def gaps(delta):
    """Per photo, left minus mirrored right of (seen flange - model flange) in link_torso_5, mm."""
    qa, qh, xi, br = unpack(delta)

    def run():
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
xz = lambda g: [round(float(np.hypot(g[3 * k], g[3 * k + 2])), 2) for k in range(len(names))]  # noqa: E731
print(f"\nphotos: {names}")
print(f"as solved: Step 2 RMS {rms(r0):.3f} sigma, photos RMS {rms(rc):.3f} sigma, x/z gaps {xz(g0)} mm")
base = {lab for lab in labels if lab[1:] in BASE_FREE or lab in BASE_FREE}
j35 = {lab for lab in labels if lab[1:] in ("J3", "J5")}
br_rot = {lab for lab in labels if "br_" in lab}
SETS = [("as Step 2", base), ("+ J3/J5", base | j35), ("+ bracket roll/pitch", base | br_rot),
        ("+ J3/J5 + bracket roll/pitch", base | j35 | br_rot)]
rows = []
for title, sel in SETS:
    free = [j for j, lab in enumerate(labels) if lab in sel]
    for w in WEIGHTS:
        A = np.vstack([J[:, free], np.sqrt(w) * Jc[:, free], 1e-3 * np.eye(len(free))])
        b = np.concatenate([-r0, -np.sqrt(w) * rc, np.zeros(len(free))])
        d_free = np.linalg.lstsq(A, b, rcond=None)[0]
        d = np.zeros(len(slots))
        d[free] = d_free
        r_new, rc_new, g_new = r0 + J @ d, rc + Jc @ d, g0 + G @ d
        top = np.argsort(-np.abs(d))[:8]
        moves = {labels[j]: round(float(d[j]), 3) for j in top if abs(d[j]) > 1e-3}
        print(f"[{title}] w {w:5.0f}: Step 2 RMS {rms(r0):.3f} -> {rms(r_new):.3f}, photos {rms(rc):.3f} -> {rms(rc_new):.3f} "
              f"sigma, x/z gaps {xz(g_new)} mm")
        print("   moves (deg / mm): " + ", ".join(f"{k} {v:+.2f}" for k, v in moves.items()))
        rows.append({"set": title, "w": w, "step2_rms": round(rms(r_new), 3), "photos_rms": round(rms(rc_new), 3),
                     "gaps_xz": xz(g_new), "gaps": np.round(g_new, 2).tolist(), "moves": moves})
Path(OUT, "path_views_fit.json").write_text(json.dumps({"photos": names, "rows": rows}, indent=1), encoding="utf-8")
