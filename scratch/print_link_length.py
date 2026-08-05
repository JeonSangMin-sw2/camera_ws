import sys
import numpy as np
import rby1_sdk as rby

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    dyn_model = robot.get_dynamics()
    names = robot.model().robot_joint_names
    
    for arm_side in ["left", "right"]:
        state = dyn_model.make_state(
            [f"link_{arm_side}_arm_5", f"ee_{arm_side}"],
            names
        )
        state.set_q(robot.get_state().position)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        length = np.linalg.norm(T[:3, 3]) * 1000.0
        print(f"{arm_side} arm L_5_ee length: {length:.4f} mm")
        print(f"{arm_side} arm T matrix:\n{T}")
        
finally:
    robot.disconnect()
