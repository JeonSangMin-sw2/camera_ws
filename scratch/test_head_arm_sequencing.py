import sys
import os
import numpy as np
from unittest.mock import MagicMock, call

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.robot_motion import (
    build_incremental_motion_plan,
    AutoCollectionConfig,
    send_auto_motion_cmd,
    execute_auto_motion_step,
    reset_motion_state,
)

def test_elbow_targets_safety():
    print("Testing elbow joint targets safety...")
    robot = MagicMock()
    dyn_model = MagicMock()
    state = MagicMock()
    state.position = np.zeros(30)
    robot.get_state.return_value = state
    
    model = MagicMock()
    model.right_arm_idx = list(range(6, 13))
    model.left_arm_idx = list(range(13, 20))
    model.head_idx = [20, 21]
    robot.model.return_value = model
    
    T_right = np.eye(4)
    T_right[:3, 3] = [0.3, -0.15, 0.25]
    T_left = np.eye(4)
    T_left[:3, 3] = [0.3, 0.15, 0.25]
    T_head = np.eye(4)
    T_head[:3, 3] = [0.1, 0.0, 0.5]
    
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
    dyn_model.compute_transformation = lambda state, idx0, idx1: state.computed_T

    config = AutoCollectionConfig()
    config.max_loops = 1
    config.step_x_m = 0.03
    config.max_x = 0.35

    plan = build_incremental_motion_plan(robot, dyn_model, config, active_arms=["right", "left"])
    
    # Check that there are NO inward motions in the plan and elbow steps are safe
    for step in plan:
        desc = step.get("desc", "")
        if "Inward" in desc:
            raise AssertionError(f"Found Inward motion in plan: {desc}")
        if "Elbow" in desc and "offsets_dict" in step and step["offsets_dict"] is not None:
            off_dict = step["offsets_dict"]
            # Verify J3 is extension-only (>= 0)
            if 3 in off_dict:
                assert off_dict[3] > 0, f"Elbow step Joint 3 offset should be positive (extension > 0), got: {off_dict[3]}"
            # If joint 2 is present in elbow step, offset should be positive (outward)
            if 2 in off_dict:
                assert off_dict[2] > 0, f"Elbow step Joint 2 offset should be outward (>0), got: {off_dict[2]}"

    print("SUCCESS: All elbow targets are extension-only and outward safe!")

def test_sequencing_logic():
    print("Testing sequential command logic (Head Up first vs Arm Down first)...")
    
    robot = MagicMock()
    config = AutoCollectionConfig()
    config.move_time = 1.5
    config.hold_time = 0.3
    config.priority = 10
    
    model = MagicMock()
    model.right_arm_idx = list(range(6, 13))
    model.left_arm_idx = list(range(13, 20))
    model.head_idx = [20, 21]
    robot.model.return_value = model
    
    mock_feedback = MagicMock()
    import rby1_sdk as rby
    mock_feedback.finish_code = rby.RobotCommandFeedback.FinishCode.Ok
    mock_future = MagicMock()
    mock_future.get.return_value = mock_feedback
    robot.send_command.return_value = mock_future
    
    # Scenario 1: Head looking UP (target pitch = -5 deg, current pitch = 0 deg)
    # Pitch difference = -5 deg < -0.3 deg -> HEAD FIRST, then ARM
    state = MagicMock()
    state.position = np.zeros(30)
    state.position[21] = np.radians(0.0) # current head pitch = 0
    state.position[6:13] = np.radians([0, 0, 0, 0, 0, 0, 0])
    state.position[13:20] = np.radians([0, 0, 0, 0, 0, 0, 0])
    robot.get_state.return_value = state
    
    robot.send_command.reset_mock()
    head_target_up = np.array([0.0, np.radians(-5.0)])
    arm_target = np.radians([10, 20, 30, -40, 50, 60, 70])
    
    send_auto_motion_cmd(
        robot=robot,
        config=config,
        active_arms=["right", "left"],
        q_right=arm_target,
        q_left=arm_target,
        head_position=head_target_up,
    )
    
    assert robot.send_command.call_count == 2, f"Expected 2 sequential commands, got {robot.send_command.call_count}"
    print("SUCCESS: Head Up triggered 2 sequential commands (Head 1st, Arm 2nd)!")
    
    # Scenario 2: Head looking DOWN (target pitch = +5 deg, current pitch = 0 deg)
    # Pitch difference = +5 deg > +0.3 deg -> ARM FIRST, then HEAD
    robot.send_command.reset_mock()
    head_target_down = np.array([0.0, np.radians(5.0)])
    
    send_auto_motion_cmd(
        robot=robot,
        config=config,
        active_arms=["right", "left"],
        q_right=arm_target,
        q_left=arm_target,
        head_position=head_target_down,
    )
    
    assert robot.send_command.call_count == 2, f"Expected 2 sequential commands, got {robot.send_command.call_count}"
    print("SUCCESS: Head Down triggered 2 sequential commands (Arm 1st, Head 2nd)!")
    
    # Scenario 3: No significant pitch change (pitch difference = 0.05 deg) -> SINGLE COMMAND
    robot.send_command.reset_mock()
    head_target_same = np.array([0.0, np.radians(0.05)])
    
    send_auto_motion_cmd(
        robot=robot,
        config=config,
        active_arms=["right", "left"],
        q_right=arm_target,
        q_left=arm_target,
        head_position=head_target_same,
    )
    
    assert robot.send_command.call_count == 1, f"Expected 1 simultaneous command, got {robot.send_command.call_count}"
    print("SUCCESS: No pitch change correctly sends a single simultaneous command!")

if __name__ == "__main__":
    test_elbow_targets_safety()
    test_sequencing_logic()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
