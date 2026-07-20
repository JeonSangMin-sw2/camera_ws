import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.robot_motion import build_incremental_motion_plan, AutoCollectionConfig

def test_plan():
    print("Testing build_incremental_motion_plan...")
    
    # Mock robot and dyn_model
    robot = MagicMock()
    dyn_model = MagicMock()
    
    # Mock state
    state = MagicMock()
    # position has to have length suitable for head and arm joints.
    # Torso: 6, Right Arm: 7, Left Arm: 7, Head: 2, etc.
    # Total joints at least 22. Let's make it 30.
    state.position = np.zeros(30)
    # Head joint indices (e.g. 20, 21)
    state.position[20] = np.radians(0.0) # head pan
    state.position[21] = np.radians(0.29) # head tilt
    
    robot.get_state.return_value = state
    
    # Mock model
    model = MagicMock()
    model.right_arm_idx = list(range(6, 13))
    model.left_arm_idx = list(range(13, 20))
    model.head_idx = [20, 21]
    robot.model.return_value = model
    
    # Mock compute_fk to return identity transformation and some position
    # compute_fk returns (state, T)
    T_right = np.eye(4)
    T_right[:3, 3] = [0.3, -0.15, 0.25]
    T_left = np.eye(4)
    T_left[:3, 3] = [0.3, 0.15, 0.25]
    
    # We want to mock compute_fk in robot_motion.py
    # Since compute_fk is imported or defined in core/robot_motion.py, let's look at its calls.
    # In core/robot_motion.py, compute_fk is defined as:
    # def compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5"):
    # So it uses state and dyn_model to compute kinematics.
    # Let's mock dyn_model's methods used by compute_fk:
    # make_state, compute_forward_kinematics, compute_transformation
    dyn_state = MagicMock()
    dyn_model.make_state.return_value = dyn_state
    
    # We want to make sure the T returned by compute_fk is realistic
    # Let's patch compute_fk in robot_motion or mock dyn_model.compute_transformation
    # to return T_right when ee_link == "ee_right" and T_left when ee_link == "ee_left".
    # And return T_head when ee_link == "link_head_2".
    T_head = np.eye(4)
    T_head[:3, 3] = [0.1, 0.0, 0.5] # neck position
    
    def mock_compute_transformation(state, idx0, idx1):
        return state.computed_T
        
    dyn_model.compute_transformation = mock_compute_transformation
    
    original_make_state = dyn_model.make_state
    def mock_make_state(links, joint_names):
        ee_link = links[1]
        s = MagicMock()
        if ee_link == "ee_right":
            s.computed_T = T_right
        elif ee_link == "ee_left":
            s.computed_T = T_left
        elif ee_link == "link_head_2":
            s.computed_T = T_head
        else:
            s.computed_T = np.eye(4)
        return s
        
    dyn_model.make_state = mock_make_state

    config = AutoCollectionConfig()
    config.angle_step_deg = 5.0
    config.step_x_m = 0.03
    config.max_x = 0.4  # base X is 0.3. step is 0.03.
    # slices: 0.30, 0.33, 0.36, 0.39. Total 4 slices.
    
    plan = build_incremental_motion_plan(robot, dyn_model, config, active_arms=["right", "left"])
    
    print(f"Total steps in plan: {len(plan)}")
    
    # Group steps by cycle or print descriptions
    # Let's count steps with type="joint" or check their types
    step_types = [step.get("type", "cartesian") for step in plan]
    from collections import Counter
    print("Step type counts:", Counter(step_types))
    
    # Check if the head sweeps are present
    head_steps = [step for step in plan if "Head Pan" in step.get("desc", "") or "Head Tilt" in step.get("desc", "")]
    print(f"Number of independent head motion steps: {len(head_steps)}")
    for i, step in enumerate(head_steps[:8]):
        print(f"  Step: {step['desc']} | head_q: {np.degrees(step['head_q'])}")
        
    # Check if total steps = 4 cycles * 37 steps = 148 steps
    expected_steps = 4 * 37
    if len(plan) == expected_steps:
        print("SUCCESS: Plan step count matches expected (148 steps)!")
    else:
        print(f"WARNING: Expected {expected_steps} steps, but got {len(plan)}!")

if __name__ == "__main__":
    test_plan()
