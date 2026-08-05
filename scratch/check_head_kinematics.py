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
    
    # Make a state with head_0 and head_1 set to zero
    state = dyn_model.make_state(
        ["link_head_0", "link_head_2"],
        names
    )
    q = robot.get_state().position.copy()
    # Find head joint indices
    head_idx = [names.index("head_0"), names.index("head_1")]
    q[head_idx] = 0.0
    
    state.set_q(q)
    dyn_model.compute_forward_kinematics(state)
    T = dyn_model.compute_transformation(state, 0, 1)
    
    print("Transformation from link_head_0 to link_head_2 (head joints at 0.0):")
    print(T)
    print("Translation in mm:", T[:3, 3] * 1000.0)
    
finally:
    robot.disconnect()
