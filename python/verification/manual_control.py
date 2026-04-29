import rby1_sdk as rby
import numpy as np
import sys
import time
import argparse
import cv2
import math

# ============================================================
# Global Variables / Configuration
# ============================================================
D2R = np.pi / 180
MINIMUM_TIME = 2.0 # 원본과 동일하게 설정
LINEAR_VELOCITY_LIMIT = 0.5
ANGULAR_VELOCITY_LIMIT = np.pi * 0.5
ACCELERATION_LIMIT = 0.5
STOP_ORIENTATION_TRACKING_ERROR = 1e-4
STOP_POSITION_TRACKING_ERROR = 1e-3

# Movement Offset (meters)
OFFSET = 0.01 

# Initial Poses (x, y, z, roll, pitch, yaw) 
# Original check_calibration_result.py parameters
INITIAL_POSE_RIGHT = [0.4, -0.15, 0.0, 90, -90, 0.0]
INITIAL_POSE_LEFT = [0.4, 0.15, 0.0, -90, -90, 0.0]

# State
selected_arm = "right"
current_target_right = list(INITIAL_POSE_RIGHT)
current_target_left = list(INITIAL_POSE_LEFT)

robot = None

# ============================================================
# Helper Functions (Exact copies from check_calibration_result.py)
# ============================================================

def make_transform(data):
    x, y, z = data[0], data[1], data[2]
    roll = data[3] * math.pi / 180
    pitch = data[4] * math.pi / 180
    yaw = data[5] * math.pi / 180
    
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = cy * cp
    m[0, 1] = sr * sp * cy - cr * sy
    m[0, 2] = cr * sp * cy + sr * sy
    m[0, 3] = x
    m[1, 0] = sy * cp
    m[1, 1] = sr * sp * sy + cr * cy
    m[1, 2] = cr * sp * sy - sr * cy
    m[1, 3] = y
    m[2, 0] = -sp
    m[2, 1] = cp * sr
    m[2, 2] = cp * cr
    m[2, 3] = z
    return m

def cb(rs):
    pass

def initialize_robot(address, model_name, power, servo):
    global robot
    robot = rby.create_robot(address, model_name)

    if not robot.connect():
        print(f"Error: Unable to establish connection to the robot at {address}")
        sys.exit(1)

    print("Successfully connected to the robot")
    robot.start_state_update(cb, 10)

    # 파라미터 설정을 원본 코드에 맞춰 더 상세히 시도
    robot.set_parameter("default.acceleration_limit_scaling", "1.0")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "20")
    robot.set_parameter("default.angular_acceleration_limit", "10")
    robot.set_parameter("manipulability_threshold", "1e4")

    if not robot.is_power_on(power):
        robot.power_on(power)

    if not robot.is_servo_on(servo):
        robot.servo_on(servo)

    control_manager_state = robot.get_control_manager_state()
    if control_manager_state.state in [rby.ControlManagerState.State.MinorFault, rby.ControlManagerState.State.MajorFault]:
        print("Resetting fault...")
        robot.reset_fault_control_manager()

    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("Error: Failed to enable Control Manager.")
        sys.exit(1)
    
    return robot

def go_to_ready_pose(robot, model_name):
    print("Moving to READY pose (Torso movement)...")
    q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
    q_joint_right_arm = np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R
    q_joint_left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R
    q = np.concatenate([q_joint_torso, q_joint_right_arm, q_joint_left_arm])

    rc = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyCommandBuilder().set_command(
                rby.JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(MINIMUM_TIME)
            )
        )
    )
    rv = robot.send_command(rc, 10).get()
    return rv.finish_code == rby.RobotCommandFeedback.FinishCode.Ok

def move_arms_cartesian(robot, target_right, target_left):
    T_right = make_transform(target_right)
    T_left = make_transform(target_left)

    rc = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(
                rby.CartesianCommandBuilder()
                .add_target(
                    "link_torso_5", "ee_right", T_right,
                    LINEAR_VELOCITY_LIMIT, ANGULAR_VELOCITY_LIMIT, ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_left_arm_command(
                rby.CartesianCommandBuilder()
                .add_target(
                    "link_torso_5", "ee_left", T_left,
                    LINEAR_VELOCITY_LIMIT, ANGULAR_VELOCITY_LIMIT, ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )
    robot.send_command(rc) 

def main(address, model_name, power, servo):
    global selected_arm, current_target_right, current_target_left, robot

    # 1. Initialize
    robot = initialize_robot(address, model_name, power, servo)

    # 2. Go to Ready Pose (Torso setup) - VERY IMPORTANT
    if not go_to_ready_pose(robot, model_name):
        print("Error: Failed to move to ready pose.")
        sys.exit(1)

    # 3. Initial Cartesian Move
    print("Moving both arms to initial Cartesian pose...")
    current_target_right = list(INITIAL_POSE_RIGHT)
    current_target_left = list(INITIAL_POSE_LEFT)
    move_arms_cartesian(robot, current_target_right, current_target_left)
    
    print("\nControls:")
    print("  'r'/'l' : Select Right/Left arm")
    print("  'a'/'z' : +/- X, 's'/'x' : +/- Y, 'd'/'c' : +/- Z")
    print("  'h'     : Reset Fault & Home (both arms)")
    print("  'q'     : Quit")

    cv2.namedWindow('Manual Control HUD')

    try:
        while True:
            try:
                state = robot.get_control_manager_state()
                state_text = str(state.state).split('.')[-1]
                state_color = (0, 255, 0) if state.state == rby.ControlManagerState.State.Enabled else (0, 0, 255)
            except RuntimeError as e:
                print(f"\n[Error] Connection lost: {e}")
                break

            # HUD
            hud = np.zeros((350, 600, 3), dtype=np.uint8)
            cv2.putText(hud, f"Status: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(hud, f"Selected: {selected_arm.upper()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(hud, f"Offset: {OFFSET}m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            curr = current_target_right if selected_arm == "right" else current_target_left
            cv2.putText(hud, f"Pos: [{curr[0]:.3f}, {curr[1]:.3f}, {curr[2]:.3f}]", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(hud, "r/l: switch, a/s/d: +, z/x/c: -, h: home, q: quit", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if state.state != rby.ControlManagerState.State.Enabled:
                 cv2.putText(hud, "FAULT! Press 'h' to Reset", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Manual Control HUD', hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): selected_arm = "right"; print("RIGHT arm selected")
            elif key == ord('l'): selected_arm = "left"; print("LEFT arm selected")
            elif key == ord('h'):
                print("Resetting and going Home...")
                if state.state != rby.ControlManagerState.State.Enabled:
                    robot.reset_fault_control_manager()
                    time.sleep(0.1)
                    robot.enable_control_manager(unlimited_mode_enabled=True)
                current_target_right = list(INITIAL_POSE_RIGHT)
                current_target_left = list(INITIAL_POSE_LEFT)
                move_arms_cartesian(robot, current_target_right, current_target_left)
            elif key in [ord('a'), ord('s'), ord('d'), ord('z'), ord('x'), ord('c')]:
                curr = current_target_right if selected_arm == "right" else current_target_left
                if key == ord('a'): curr[0] += OFFSET
                elif key == ord('s'): curr[1] += OFFSET
                elif key == ord('d'): curr[2] += OFFSET
                elif key == ord('z'): curr[0] -= OFFSET
                elif key == ord('x'): curr[1] -= OFFSET
                elif key == ord('c'): curr[2] -= OFFSET
                print(f"Moving {selected_arm} to {curr[:3]}")
                try:
                    move_arms_cartesian(robot, current_target_right, current_target_left)
                except RuntimeError as e:
                    print(f"Command error: {e}"); break
            
            time.sleep(0.1)

    finally:
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default='a')
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="^(?!.*head).*", )
    args = parser.parse_args()
    main(args.address, args.model, args.power, args.servo)
