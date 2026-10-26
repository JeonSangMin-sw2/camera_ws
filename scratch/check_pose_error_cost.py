"""How cheaply can a saved Step 2 run hide a hand error at the check pose?

2026-10-19 (v1.2 robot, applied): at the symmetric check pose the right hand sat right in x but rolled
about the robot x axis, about 2-3 mm off in z, and the right J0 looked off. This re-solves the run's Step 2
(scratch/step2_rerun.py, as run) and keeps the fitted optimizer, then finds the offset change
that produces a given right-hand displacement at the check pose (torso frame) while raising the
noise-weighted Step 2 residual the least:  min |J d|^2  s.t.  C d = t  (d in deg / mm; head and camera
free to absorb). If that rise is a small fraction of a sigma, the 64 poses cannot see such an error.
Offline, nothing is applied; outputs in archive/reports/step2_rerun_<date_time>/ (the rerun's).
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\check_pose_error_cost.py <run folder> <timestamp>
"""
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
m = runpy.run_path(str(ROOT / "scratch" / "step2_rerun.py"), run_name="rerun")
import core.calibration.observability as obs  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

captured = {}
original = step2_mod.step2_weak_directions


def keep(*args, **kwargs):
    captured["args"], captured["kwargs"] = args, kwargs
    return original(*args, **kwargs)


with patch.object(step2_mod, "step2_weak_directions", keep):
    m["solve"]("as_run_cost")
opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms = captured["args"]
reference = captured["kwargs"]["reference_q_arm"]
q_arm_offset = np.asarray(q_arm_offset, float)
q_head_offset = None if q_head_offset is None else np.asarray(q_head_offset, float)
xi_cam = np.zeros(6) if xi_cam is None else np.asarray(xi_cam, float)
labels, scale, slots = obs._parameter_layout(opt, q_arm_offset, q_head_offset, active_arms)


def unpack(delta):
    qa, qh, xi = q_arm_offset.copy(), None if q_head_offset is None else q_head_offset.copy(), xi_cam.copy()
    for (kind, k), d in zip(slots, delta):
        if kind == "arm":
            qa[k] += d
        elif kind == "head":
            qh[k] += d
        else:
            xi[k] += d
    return qa, qh, xi


args = (opt, q_arm_list, q_head_list, T_meas_list)
r0 = obs._residuals(*args, q_arm_offset, q_head_offset, xi_cam, active_arms)
J = np.zeros((len(r0), len(slots)))
for j, (kind, _) in enumerate(slots):
    h = obs.FD_STEP_M if (kind == "cam" and scale[j] == 1e-3) else obs.FD_STEP_RAD
    d = np.zeros(len(slots))
    d[j] = h
    J[:, j] = (obs._residuals(*args, *unpack(d), active_arms) - r0) / h * scale[j]


def hand_poses(qa):
    """[right x y z (mm), right rotvec (deg), left x y z, left rotvec] in the torso frame at the check pose."""
    q_full = obs.prepare_q_full(q_nominal=opt.q_nominal, arm_idx=opt.arm_idx, q_cmd=reference, q_offset=qa)
    out = []
    for side in obs.ARM_SIDES:
        state = opt.dyn_model.make_state([obs.TORSO_LINK, opt.ee_links[side]], opt.model.robot_joint_names)
        state.set_q(q_full)
        opt.dyn_model.compute_forward_kinematics(state)
        T = np.asarray(opt.dyn_model.compute_transformation(state, 0, 1))
        out.append((T[:3, 3] * 1000.0, T[:3, :3]))
    return out


base = hand_poses(q_arm_offset)
G = np.zeros((12, len(slots)))
for j, (kind, k) in enumerate(slots):
    if kind != "arm":
        continue
    qa = q_arm_offset.copy()
    qa[k] += obs.FD_STEP_RAD
    moved = hand_poses(qa)
    for s in range(2):
        G[6 * s:6 * s + 3, j] = (moved[s][0] - base[s][0]) / obs.FD_STEP_RAD * scale[j]
        G[6 * s + 3:6 * s + 6, j] = np.degrees(Rotation.from_matrix(moved[s][1] @ base[s][1].T).as_rotvec()) \
            / obs.FD_STEP_RAD * scale[j]

n = len(r0)
print(f"\nStep 2 residual RMS {np.sqrt(np.mean(r0 ** 2)):.3f} sigma over {n} residuals")
step1_locked = {f"{s}J{j}" for s in "RL" for j in (3, 5)}
rows = {"R x": 0, "R y": 1, "R z": 2, "L x": 6, "L y": 7, "L z": 8}
cases = [
    ("right z +2.5, right x,y 0", ["R x", "R y", "R z"], [0, 0, 2.5]),
    ("right z +2.5, right x,y 0, left hand 0", ["R x", "R y", "R z", "L x", "L y", "L z"], [0, 0, 2.5, 0, 0, 0]),
]
for with_locked in (False, True):
    free = [i for i, name in enumerate(labels) if with_locked or name not in step1_locked]
    for title, names, target in cases:
        C = G[[rows[x] for x in names]][:, free]
        Jf = J[:, free]
        M = Jf.T @ Jf + 1e-9 * np.eye(len(free))
        Mi = np.linalg.inv(M)
        d_free = Mi @ C.T @ np.linalg.solve(C @ Mi @ C.T, np.asarray(target, float))
        d = np.zeros(len(slots))
        d[free] = d_free
        rise = np.sqrt(np.sum((J @ d) ** 2) / n)
        hand = G @ d
        top = np.argsort(-np.abs(d))[:7]
        print(f"\n[{title}] {'J3/J5 free too' if with_locked else 'J3/J5 held (Step 1)'}: residual RMS "
              f"{np.sqrt(np.mean(r0 ** 2)):.3f} -> {np.sqrt(np.mean(r0 ** 2) + rise ** 2):.3f} sigma (rise term {rise:.3f})")
        print("   offsets: " + ", ".join(f"{labels[i]} {d[i]:+.2f}" for i in top))
        print(f"   right hand: move {np.round(hand[0:3], 2)} mm, rotation about torso x/y/z {np.round(hand[3:6], 2)} deg")
        print(f"   left hand : move {np.round(hand[6:9], 2)} mm, rotation {np.round(hand[9:12], 2)} deg")
