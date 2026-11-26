"""Can a stiffness difference between the arms explain the first robot's check-pose gap? (2026-11-26, offline)

The camera gap at each photo (left minus mirrored right, link_torso_5, x and z) is fitted with
    gap(p) = sum_j c_j * dgap/dq_j(p) * (tau_j(p) - mean tau_j over the Step 2 dataset)
c_j = joint compliance of the right arm relative to the left (rad per Nm; the sign takes either arm). The Step 2 offsets
already absorb the mean sag at the Step 2 poses, hence the subtraction. Gravity torque from the simulator model
(127.0.0.1, model m, v1.2; torso [0,30,-60,30,0,0], no payload added: marker brackets and missing grippers are not in
the model), at physical angles = encoder + result offsets. Nothing moves; leave-one-photo-out RMS guards overfitting.
usage: python scratch/compliance_fit.py <result json> <dataset npz> <check_path_views.json> [more views json ...]
"""
import itertools
import json
import sys

import numpy as np
import rby1_sdk as rby

result = json.load(open(sys.argv[1], encoding="utf-8"))
data = np.load(sys.argv[2], allow_pickle=True)
rows = [r for f in sys.argv[3:] for r in json.load(open(f, encoding="utf-8"))["rows"] if r.get("seen")]
off = np.radians(np.concatenate([result["right_arm_joint_offset_deg"], result["left_arm_joint_offset_deg"]]))

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
r_idx, l_idx = list(model.right_arm_idx), list(model.left_arm_idx)
st = dyn.make_state(["link_torso_5", "ee_right", "ee_left"], list(model.robot_joint_names))
st.set_gravity(np.array([0, 0, 0, 0, 0, -9.81]))
q0 = np.zeros(model.robot_dof)
q0[list(model.torso_idx)] = np.radians([0, 30, -60, 30, 0, 0])


def full(q_arm):
    q = q0.copy()
    q[r_idx + l_idx] = np.asarray(q_arm, dtype=float) + off
    return q


def gap(q):
    st.set_q(q)
    dyn.compute_forward_kinematics(st)
    r, l = (np.asarray(dyn.compute_transformation(st, 0, i))[:3, 3] * 1000 for i in (1, 2))
    return np.array([l[0] - r[0], l[2] - r[2]])


def tau(q):
    st.set_q(q)
    st.set_qdot(np.zeros_like(q))
    st.set_qddot(np.zeros_like(q))
    dyn.compute_forward_kinematics(st)
    dyn.compute_diff_forward_kinematics(st)
    dyn.compute_2nd_diff_forward_kinematics(st)
    dyn.compute_inverse_dynamics(st)
    return np.asarray(st.get_tau())


tau_ref = np.mean([tau(full(q))[r_idx] for q in data["q_arm"]], axis=0)
names, G, F, TR = [], [], [], []
for row in rows:
    q = full(row["q_arm"])
    t = tau(q)
    J = np.zeros((2, 7))
    for j, k in enumerate(r_idx):
        dq = q.copy()
        dq[k] += 1e-4
        J[:, j] = (gap(dq) - gap(q)) / 1e-4
    names.append(row["pose"])
    G.append([row["gap_x"], row["gap_z"]])
    F.append(J * (t[r_idx] - tau_ref))      # 2 x 7, mm per (rad/Nm)
    TR.append(t[r_idx])
G, F, TR = np.array(G), np.array(F), np.array(TR)
print(f"{len(rows)} photos; Step 2 mean right-arm gravity torque (Nm): {np.round(tau_ref, 1).tolist()}")
for n, t, g in zip(names, TR, G):
    print(f"   {n:12s} tau {np.round(t, 1).tolist()}  gap x {g[0]:+.2f} z {g[1]:+.2f}")
print(f"left/right torque mirror check at the last photo: {np.round(tau(full(rows[-1]['q_arm']))[l_idx], 1).tolist()}")


def fit(cols, rows_used):
    A = F[rows_used][:, :, cols].reshape(-1, len(cols))
    c, *_ = np.linalg.lstsq(A, G[rows_used].reshape(-1), rcond=None)
    return c


def evaluate(cols):
    n = len(G)
    c = fit(cols, np.arange(n))
    pred = np.einsum("pij,j->pi", F[:, :, cols], c)
    loo = np.array([np.einsum("ij,j->i", F[k][:, cols], fit(cols, np.delete(np.arange(n), k))) for k in range(n)])
    return c, pred, np.sqrt(np.mean((G - pred) ** 2)), np.sqrt(np.mean((G - loo) ** 2))


print(f"\nno model: RMS {np.sqrt(np.mean(G ** 2)):.2f} mm (x and z over all photos)")
results = []
for k in (1, 2, 3):
    for cols in itertools.combinations(range(7), k):
        c, pred, rms, loo = evaluate(list(cols))
        results.append((loo, rms, cols, c, pred))
results.sort(key=lambda x: x[0])
for loo, rms, cols, c, _ in results[:8]:
    stiff = ", ".join(f"J{j} {1 / v:+.0f}" if abs(v) > 1e-12 else f"J{j} inf" for j, v in zip(cols, c))
    print(f"joints {cols}: fit RMS {rms:.2f}, leave-one-out {loo:.2f} mm; relative stiffness (Nm/rad) {stiff}")
c, pred, rms, loo = evaluate(list(range(7)))
print(f"all 7 joints: fit RMS {rms:.2f}, leave-one-out {loo:.2f} mm")
best = results[0]
print(f"\nbest {best[2]} per photo (measured -> fitted, mm):")
for n, g, p in zip(names, G, best[4]):
    print(f"   {n:12s} x {g[0]:+.2f} -> {p[0]:+.2f}   z {g[1]:+.2f} -> {p[1]:+.2f}")
