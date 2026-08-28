import numpy as np

import rby1_sdk
robot = rby1_sdk.create_robot("127.0.0.1", "m")
dyn = robot.get_dynamics()
model = robot.model()

q = np.zeros(26)
# right arm ready pose: [-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0]
r_arm = np.radians([-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0])
q[model.right_arm_idx] = r_arm

state = dyn.make_state(
    ["base", "link_right_arm_4", "link_right_arm_5", "link_right_arm_6", "ee_right"],
    model.robot_joint_names
)
state.set_q(q)
dyn.compute_forward_kinematics(state)

T_0_4 = dyn.compute_transformation(state, 0, 1) # link 4
T_0_5 = dyn.compute_transformation(state, 0, 2) # link 5
T_0_6 = dyn.compute_transformation(state, 0, 3) # link 6
T_0_ee = dyn.compute_transformation(state, 0, 4) # ee

print("=== Robot Kinematic Axes at Ready Pose (in Base Frame) ===")
# In URDF, what are the joint axes for J4, J5, J6?
print("T_0_4 Rotation Matrix:\n", np.round(T_0_4[:3, :3], 3))
print("T_0_5 Rotation Matrix:\n", np.round(T_0_5[:3, :3], 3))
print("T_0_6 Rotation Matrix:\n", np.round(T_0_6[:3, :3], 3))
print("T_0_ee Rotation Matrix:\n", np.round(T_0_ee[:3, :3], 3))
