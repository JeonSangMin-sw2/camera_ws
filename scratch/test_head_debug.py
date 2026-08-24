import rby1_sdk
import numpy as np

robot = rby1_sdk.create_robot("127.0.0.1", "m")
dyn = robot.get_dynamics()
model = robot.model()

print("model.head_idx:", model.head_idx)
state = dyn.make_state(["link_head_2", "ee_right"], model.robot_joint_names)
q = np.zeros(len(model.robot_joint_names))
state.set_q(q)
dyn.compute_forward_kinematics(state)
dyn.compute_diff_forward_kinematics(state)

Jb = dyn.compute_body_jacobian(state, 0, 1)
print("Jb shape:", Jb.shape)
print("Head pan col (idx 24):", Jb[:, 24])
print("Head tilt col (idx 25):", Jb[:, 25])
