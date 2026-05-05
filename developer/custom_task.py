import time
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to sys.path to allow importing calibration_core
sys.path.append(str(Path(__file__).resolve().parent.parent))

import rby1_sdk as rby
from calibration_core import (
    rot_x, rot_y, rot_z, make_T, compute_fk
)

# ---------------------------------------------------------
# Motion Logic (Pure Robot Motion, No Camera)
# ---------------------------------------------------------

def apply_cartesian_offset(T, dx=0, dy=0, dz=0, droll_deg=0, dpitch_deg=0, dyaw_deg=0):
    """Apply relative offsets to a 4x4 transform."""
    T_new = T.copy()
    T_new[0, 3] += dx
    T_new[1, 3] += dy
    T_new[2, 3] += dz
    
    R_off = rot_z(np.deg2rad(dyaw_deg)) @ rot_y(np.deg2rad(dpitch_deg)) @ rot_x(np.deg2rad(droll_deg))
    T_new[:3, :3] = R_off @ T_new[:3, :3]
    return T_new

def run_incremental_motion(robot, angle_step_deg=5.0, position_step_m=0.03, max_x=0.5):
    """
    Moves the robot in an incremental pattern:
    1. Base Pose
    2. R+5, R-5, P+5, P-5, Y+5, Y-5 (No return to base in between)
    3. Y+3, Y-3, Z+3, Z-3 (No return to base)
    4. Next Base (X+step)
    """
    model = robot.model()
    dyn_model = robot.get_dynamics()
    
    move_time = 1.2
    settle_time = 0.5
    priority = 10

    def send_move(T_r, T_l, desc=""):
        if desc: print(f"  [Move] {desc}")
        body = rby.BodyComponentBasedCommandBuilder()
        body.set_right_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_right", T_r, 0.2, 0.5, 0.3)
            .set_minimum_time(move_time)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(2.0))
        )
        body.set_left_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_left", T_l, 0.2, 0.5, 0.3)
            .set_minimum_time(move_time)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(2.0))
        )
        cmd = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body)
        )
        rv = robot.send_command(cmd, priority).get()
        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            print(f"    ! Error: Move failed with code {rv.finish_code}")
        time.sleep(settle_time)

    print(f"Starting Incremental Motion Pattern (Max X: {max_x}m)...")
    
    # Get initial base pose
    state = robot.get_state()
    q_full = state.position
    _, T_base_right = compute_fk(robot, dyn_model, q_full, "ee_right", "link_torso_5")
    _, T_base_left = compute_fk(robot, dyn_model, q_full, "ee_left", "link_torso_5")

    while True:
        curr_x = T_base_right[0, 3]
        if curr_x > max_x:
            print(f"Reached max X limit ({curr_x:.3f} > {max_x}). Finishing.")
            break
            
        print(f"\n--- Level: X = {curr_x:.3f} ---")

        # 1. RPY Offsets (+5, -5 only, no return to base)
        print("  Running RPY offsets...")
        rpy_targets = [
            (angle_step_deg, 0, 0), (-angle_step_deg, 0, 0),
            (0, angle_step_deg, 0), (0, -angle_step_deg, 0),
            (0, 0, angle_step_deg), (0, 0, -angle_step_deg)
        ]
        for dr, dp, dy in rpy_targets:
            tr = apply_cartesian_offset(T_base_right, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            tl = apply_cartesian_offset(T_base_left, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            send_move(tr, tl, f"RPY: ({dr}, {dp}, {dy})")

        # 2. YZ Offsets (+3cm, -3cm only, no return to base)
        print("  Running YZ offsets...")
        yz_targets = [
            (0, position_step_m, 0), (0, -position_step_m, 0),
            (0, 0, position_step_m), (0, 0, -position_step_m)
        ]
        for dx, dy, dz in yz_targets:
            tr = apply_cartesian_offset(T_base_right, dx=dx, dy=dy, dz=dz)
            tl = apply_cartesian_offset(T_base_left, dx=dx, dy=dy, dz=dz)
            send_move(tr, tl, f"Pos: ({dx}, {dy}, {dz})")

        # 3. Advance X reference
        T_base_right = apply_cartesian_offset(T_base_right, dx=position_step_m)
        T_base_left = apply_cartesian_offset(T_base_left, dx=position_step_m)

# ---------------------------------------------------------
# Initialization and Main
# ---------------------------------------------------------

def main():
    address = "127.0.0.1:50051"
    model_name = "a"
    
    print(f"Connecting to robot at {address}...")
    robot = rby.create_robot(address, model_name)
    if not robot.connect():
        print("Error: Could not connect to robot.")
        return

    print("Initializing Control Manager...")
    robot.power_on(".*")
    time.sleep(0.5)
    robot.servo_on(".*")
    time.sleep(0.5)
    robot.reset_fault_control_manager()
    time.sleep(0.5)
    
    # CRITICAL: Enable control manager and wait until it's actually ENABLED
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("Error: Failed to send enable command to Control Manager.")
        return
        
    print("Waiting for Control Manager to be ENABLED...")
    timeout = 10.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        state = robot.get_control_manager_state().state
        if state == rby.ControlManagerState.State.Enabled:
            print("Control Manager is ENABLED.")
            break
        time.sleep(0.2)
    else:
        print(f"Error: Control Manager failed to enable (Current state: {robot.get_control_manager_state().state})")
        return

    time.sleep(1)

    # Move to Ready Pose first (to avoid singularities and ensure robot is ready)
    print("Moving to Ready Pose...")
    D2R = np.pi / 180.0
    q_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
    q_right = np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R
    q_left = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R
    q_ready = np.concatenate([q_torso, q_right, q_left])
    
    cmd_ready = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.JointPositionCommandBuilder().set_position(q_ready).set_minimum_time(3.0)
        )
    )
    robot.send_command(cmd_ready, 10).get()
    time.sleep(0.5)

    # Starting Pose (from check_calibration_result.py)
    # Right: [0.3, -0.13, 0.0, 90, -90, 0]
    # Left:  [0.3, 0.13, 0.0, -90, -90, 0]
    T_r_start = make_T(rot_z(0) @ rot_y(np.deg2rad(-90)) @ rot_x(np.deg2rad(90)), [0.3, -0.13, 0.0])
    T_l_start = make_T(rot_z(0) @ rot_y(np.deg2rad(-90)) @ rot_x(np.deg2rad(-90)), [0.3, 0.13, 0.0])
    
    print("Moving to initial calibration checking pose...")
    body = rby.BodyComponentBasedCommandBuilder()
    body.set_right_arm_command(rby.CartesianCommandBuilder().add_target("link_torso_5", "ee_right", T_r_start, 0.2, 0.5, 0.3).set_minimum_time(3.0))
    body.set_left_arm_command(rby.CartesianCommandBuilder().add_target("link_torso_5", "ee_left", T_l_start, 0.2, 0.5, 0.3).set_minimum_time(3.0))
    cmd = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body))
    
    rv = robot.send_command(cmd, 10).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        print(f"Error: Initial move failed with code {rv.finish_code}")
        return

    # Run the pattern
    run_incremental_motion(robot, angle_step_deg=5.0, position_step_m=0.03, max_x=0.5)
    
    print("\nTask Completed Successfully.")

if __name__ == "__main__":
    main()
