"""Print the simulator's torso joints and their axes at the zero pose (sim 127.0.0.1 only, no motion)."""
import numpy as np
import rby1_sdk as rby

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "sim not reachable"
model = robot.model()
dyn = robot.get_dynamics()
names = list(model.robot_joint_names)
print("joints:", names)
print("torso_idx:", list(model.torso_idx), "head_idx:", list(model.head_idx))
print("dyn methods:", [m for m in dir(dyn) if not m.startswith("_")])
links = ["base"] + [f"link_torso_{i}" for i in range(6)] + ["link_head_0", "link_head_2"]
state = dyn.make_state(links, names)
state.set_q(np.zeros(len(names)))
dyn.compute_forward_kinematics(state)
for i, link in enumerate(links[1:], start=1):
    T = dyn.compute_transformation(state, 0, i)
    print(f"{link:14s} p={np.round(T[:3, 3], 4)}\n  R=\n{np.round(T[:3, :3], 3)}")

for j in range(6):
    q = np.zeros(len(names)); q[4 + j] = 0.1
    state.set_q(q); dyn.compute_forward_kinematics(state)
    T1 = dyn.compute_transformation(state, 0, 1 + j)
    state.set_q(np.zeros(len(names))); dyn.compute_forward_kinematics(state)
    T0 = dyn.compute_transformation(state, 0, 1 + j)
    dR = T1[:3, :3] @ T0[:3, :3].T
    w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
    print(f"torso_{j} axis (base frame) = {np.round(w / np.linalg.norm(w), 3)}  limits deg [{np.degrees(dyn.get_limit_q_lower(state)[4+j]):.0f}, {np.degrees(dyn.get_limit_q_upper(state)[4+j]):.0f}]")
