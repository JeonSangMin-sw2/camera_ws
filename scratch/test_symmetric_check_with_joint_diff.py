import time
import numpy as np
import rby1_sdk as rby

ROBOT_IP = "127.0.0.1:50051"
MODEL_NAME = "m"
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# Given Diff (B-O) in degrees
DIFF_RIGHT_DEG = np.array([-0.0266, -0.975,  0.4921,  0.0746,  0.4687, 0.9339,  1.1469], dtype=np.float64)
DIFF_LEFT_DEG  = np.array([-0.9664,  1.646, -0.7741, -1.7296, -0.3687, 0.4048, -0.1936], dtype=np.float64)
DIFF_HEAD_DEG  = np.array([ 1.0026, -1.8257], dtype=np.float64)

def setup_robot(robot):
    print("\n--- 1. Robot Power & Servo Setup (Simulation: .*) ---")
    # Reset CM fault if needed
    cm = robot.get_control_manager_state()
    if cm.state != rby.ControlManagerState.State.Idle:
        print(f"Control Manager in {cm.state}. Resetting fault...")
        robot.reset_fault_control_manager()
        time.sleep(1.0)

    # Power ON .*
    if not robot.is_power_on(".*"):
        print("Powering on (.*)...")
        if not robot.power_on(".*"):
            raise RuntimeError("Failed to power on")
        time.sleep(1.0)
    print("Power state: ON")

    # Servo ON .*
    if not robot.is_servo_on(".*"):
        print("Turning servos on (.*)...")
        if not robot.servo_on(".*"):
            raise RuntimeError("Failed to turn servos on")
        time.sleep(1.0)
    print("Servo state: ON")

    # Enable CM
    cm = robot.get_control_manager_state()
    if cm.state != rby.ControlManagerState.State.Enabled:
        if cm.state != rby.ControlManagerState.State.Idle:
            robot.reset_fault_control_manager()
            time.sleep(0.5)
        print("Enabling Control Manager...")
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable Control Manager")
        time.sleep(0.5)
    print("Control Manager state: Enabled")

def move_to_symmetric_check_pose(robot, x=0.35, y=0.0, z=0.0, offset=0.1):
    print(f"\n--- 2. Moving to Step2 Symmetrical Check Pose (X={x}, Y={y}, Z={z}, Offset={offset}m) ---")
    
    # Step 2-1: Joint Ready Pose
    print("Moving to Joint Ready Pose...")
    q_torso = np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R
    q_right = np.array([-45, -30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    q_left  = np.array([-45,  30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    q_ready = np.concatenate([q_torso, q_right, q_left])

    cmd_ready = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.JointPositionCommandBuilder()
            .set_position(q_ready)
            .set_minimum_time(4.0)
        )
    )
    rv1 = robot.send_command(cmd_ready, 10).get()
    if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Failed to reach Ready Pose: {rv1.finish_code}")
    time.sleep(1.0)

    # Step 2-2: Cartesian Symmetrical Pose
    print(f"Moving to Cartesian Symmetrical Pose with Offset={offset}m...")
    import math
    roll_r = 90.0 * D2R
    pitch_r = -90.0 * D2R
    yaw_r = 0.0
    cr_r = math.cos(roll_r); sr_r = math.sin(roll_r)
    cp_r = math.cos(pitch_r); sp_r = math.sin(pitch_r)
    cy_r = math.cos(yaw_r); sy_r = math.sin(yaw_r)

    T_right = np.eye(4, dtype=np.float64)
    T_right[0, 0] = cy_r * cp_r
    T_right[0, 1] = sr_r * sp_r * cy_r - cr_r * sy_r
    T_right[0, 2] = cr_r * sp_r * cy_r + sr_r * sy_r
    T_right[0, 3] = x

    T_right[1, 0] = sy_r * cp_r
    T_right[1, 1] = sr_r * sp_r * sy_r + cr_r * cy_r
    T_right[1, 2] = cr_r * sp_r * sy_r - sr_r * cy_r
    T_right[1, 3] = y - offset

    T_right[2, 0] = -sp_r
    T_right[2, 1] = cp_r * sr_r
    T_right[2, 2] = cp_r * cr_r
    T_right[2, 3] = z

    roll_l = -90.0 * D2R
    pitch_l = -90.0 * D2R
    yaw_l = 0.0
    cr_l = math.cos(roll_l); sr_l = math.sin(roll_l)
    cp_l = math.cos(pitch_l); sp_l = math.sin(pitch_l)
    cy_l = math.cos(yaw_l); sy_l = math.sin(yaw_l)

    T_left = np.eye(4, dtype=np.float64)
    T_left[0, 0] = cy_l * cp_l
    T_left[0, 1] = sr_l * sp_l * cy_l - cr_l * sy_l
    T_left[0, 2] = cr_l * sp_l * cy_l + sr_l * sy_l
    T_left[0, 3] = x

    T_left[1, 0] = sy_l * cp_l
    T_left[1, 1] = sr_l * sp_l * sy_l + cr_l * cy_l
    T_left[1, 2] = cr_l * sp_l * sy_l - sr_l * cy_l
    T_left[1, 3] = y + offset

    T_left[2, 0] = -sp_l
    T_left[2, 1] = cp_l * sr_l
    T_left[2, 2] = cp_l * cr_l
    T_left[2, 3] = z

    body = rby.BodyComponentBasedCommandBuilder()
    right_cart = rby.CartesianCommandBuilder()
    right_cart.add_target("link_torso_5", "ee_right", T_right.astype(np.float32), 1.5, math.pi * 1.5, 1.0)
    right_cart.set_stop_position_tracking_error(1e-3)
    right_cart.set_stop_orientation_tracking_error(1e-4)
    right_cart.set_minimum_time(4.0)
    body.set_right_arm_command(right_cart)

    left_cart = rby.CartesianCommandBuilder()
    left_cart.add_target("link_torso_5", "ee_left", T_left.astype(np.float32), 1.5, math.pi * 1.5, 1.0)
    left_cart.set_stop_position_tracking_error(1e-3)
    left_cart.set_stop_orientation_tracking_error(1e-4)
    left_cart.set_minimum_time(4.0)
    body.set_left_arm_command(left_cart)

    cmd_cart = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )
    rv2 = robot.send_command(cmd_cart, 10).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Failed to reach Cartesian Pose: {rv2.finish_code}")
    time.sleep(1.0)
    print("Reached Cartesian Symmetrical Pose successfully.")

def apply_joint_diffs_and_move(robot):
    print("\n--- 3. Reading Current Joints and Applying Diff (B-O) ---")
    model = robot.model()
    state = robot.get_state()

    q_torso = state.position[list(model.torso_idx)]
    q_right = state.position[list(model.right_arm_idx)]
    q_left  = state.position[list(model.left_arm_idx)]
    q_head  = state.position[list(model.head_idx)]

    print(f"Current Right Arm (deg): {np.round(q_right * R2D, 4).tolist()}")
    print(f"Current Left Arm  (deg): {np.round(q_left * R2D, 4).tolist()}")
    print(f"Current Head      (deg): {np.round(q_head * R2D, 4).tolist()}")

    print(f"\nAdding Diff (B-O):")
    print(f"  Right Diff (deg): {DIFF_RIGHT_DEG.tolist()}")
    print(f"  Left Diff  (deg): {DIFF_LEFT_DEG.tolist()}")
    print(f"  Head Diff  (deg): {DIFF_HEAD_DEG.tolist()}")

    # Target joints (rad)
    q_right_target = q_right + (DIFF_RIGHT_DEG * D2R)
    q_left_target  = q_left  + (DIFF_LEFT_DEG * D2R)
    q_head_target  = q_head  + (DIFF_HEAD_DEG * D2R)
    q_torso_target = q_torso

    print(f"\nTarget Right Arm (deg): {np.round(q_right_target * R2D, 4).tolist()}")
    print(f"Target Left Arm  (deg): {np.round(q_left_target * R2D, 4).tolist()}")
    print(f"Target Head      (deg): {np.round(q_head_target * R2D, 4).tolist()}")

    # Command Robot to Target Joint Positions
    print("\n--- 4. Moving Robot to New Joint Target Positions ---")
    body = rby.BodyComponentBasedCommandBuilder()
    body.set_torso_command(
        rby.JointPositionCommandBuilder().set_position(q_torso_target).set_minimum_time(3.0)
    )
    body.set_right_arm_command(
        rby.JointPositionCommandBuilder().set_position(q_right_target).set_minimum_time(3.0)
    )
    body.set_left_arm_command(
        rby.JointPositionCommandBuilder().set_position(q_left_target).set_minimum_time(3.0)
    )

    comp = rby.ComponentBasedCommandBuilder().set_body_command(body)
    comp.set_head_command(
        rby.JointPositionCommandBuilder().set_position(q_head_target).set_minimum_time(3.0)
    )

    cmd = rby.RobotCommandBuilder().set_command(comp)
    rv = robot.send_command(cmd, 10).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Failed to move to target joints: {rv.finish_code}")

    time.sleep(1.0)
    final_state = robot.get_state()
    q_right_final = final_state.position[list(model.right_arm_idx)]
    q_left_final  = final_state.position[list(model.left_arm_idx)]
    q_head_final  = final_state.position[list(model.head_idx)]

    print("\n--- 5. Verification: Final Robot Joint Positions ---")
    print(f"Final Right Arm (deg): {np.round(q_right_final * R2D, 4).tolist()}")
    print(f"Final Left Arm  (deg): {np.round(q_left_final * R2D, 4).tolist()}")
    print(f"Final Head      (deg): {np.round(q_head_final * R2D, 4).tolist()}")

    err_r = np.max(np.abs((q_right_final - q_right_target) * R2D))
    err_l = np.max(np.abs((q_left_final - q_left_target) * R2D))
    err_h = np.max(np.abs((q_head_final - q_head_target) * R2D))
    print(f"\nMax Tracking Error (deg) -> Right: {err_r:.4f}°, Left: {err_l:.4f}°, Head: {err_h:.4f}°")
    print("SUCCESS: Check pose with joint diff offsets completed!")

def main():
    print(f"Connecting to robot at {ROBOT_IP} (model: {MODEL_NAME})...")
    robot = rby.create_robot(ROBOT_IP, MODEL_NAME)
    if not robot.connect():
        raise RuntimeError(f"Could not connect to robot at {ROBOT_IP}")

    setup_robot(robot)
    move_to_symmetric_check_pose(robot, x=0.35, y=0.0, z=0.0, offset=0.1)
    apply_joint_diffs_and_move(robot)

if __name__ == "__main__":
    main()
