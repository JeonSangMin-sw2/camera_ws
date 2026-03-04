import rby1_sdk as rby
import argparse
import numpy as np
import logging
import time
import json


def load_offset_from_json(filename="calibration_result.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    offset_deg = np.array(data["joint_offset_deg"])
    return np.deg2rad(offset_deg)


def movej(robot, torso=None, right_arm=None, left_arm=None, minimum_time=5):
    rc = rby.BodyComponentBasedCommandBuilder()

    if right_arm is not None:
        rc.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(right_arm)
        )

    if left_arm is not None:
        rc.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(left_arm)
        )
    rc.set_torso_command(
        rby.JointPositionCommandBuilder()
        .set_minimum_time(minimum_time)
        .set_position(np.zeros(6))
    )

    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(rc)
        ),
        1,
    ).get()

    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        logging.error("Failed to conduct movej.")
        return False

    return True


def initialize_robot(address, model, power=".*", servo=".*"):
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        exit(1)

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error("Power on failed")
            exit(1)

    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error("Servo on failed")
            exit(1)

    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        robot.reset_fault_control_manager()

    robot.enable_control_manager()

    return robot


def main(address, model_name, power, servo, arm):
    robot = initialize_robot(address, model_name, power, servo)

    model = robot.model()

    if arm == "right":
        arm_dof = len(model.right_arm_idx)
        zero_pose = np.zeros(arm_dof)
        arm_prefix = "right_arm"
    else:
        arm_dof = len(model.left_arm_idx)
        zero_pose = np.zeros(arm_dof)
        arm_prefix = "left_arm"

    # 1️⃣ zero pose
    print(f"Moving {arm} arm to zero pose...")
    # if arm == "right":
    #     movej(robot, right_arm=zero_pose, minimum_time=5)
    # else:
    #     movej(robot, left_arm=zero_pose, minimum_time=5)
    movej(robot, right_arm=zero_pose, left_arm=zero_pose, minimum_time=5)
    
    print("\nSelect offset mode:")
    print("u → User input")
    print("j → Load from JSON")
    mode = input("Mode (u/j) = ").strip().lower()

    # =========================
    # 1️⃣ 사용자 입력 모드
    # =========================
    if mode == "u":
        print(f"\nEnter {arm_dof} joint offsets in DEG (space separated):")
        print("Example: 1 0 0 0 0 0 0")

        user_input = input("Offset (deg) = ")

        try:
            offset_deg = np.array([float(x) for x in user_input.split()])
        except Exception:
            print("Invalid input.")
            return

        if len(offset_deg) != arm_dof:
            print(f"Need exactly {arm_dof} values.")
            return

        offset_rad = np.deg2rad(offset_deg)

    # =========================
    # 2️⃣ JSON 모드
    # =========================
    elif mode == "j":
        try:
            offset_rad = -load_offset_from_json("calibration_result.json")
            offset_deg = np.rad2deg(offset_rad)
        except Exception as e:
            print("Failed to load JSON:", e)
            return

        if len(offset_rad) != arm_dof:
            print("JSON offset size mismatch.")
            return

        print("Loaded offset from JSON (deg):")
        print(offset_deg)

    else:
        print("Invalid mode selected.")
        return

    # 3️⃣ zero + offset
    target_pose = zero_pose + offset_rad

    print(f"Moving {arm} arm with offset...")
    print("Offset (deg):", offset_deg)

    if arm == "right":
        movej(robot, right_arm=target_pose, minimum_time=10)
    else:
        movej(robot, left_arm=target_pose, minimum_time=10)

    time.sleep(1)

    all_success = True

    for i in range(arm_dof):
        joint_name = f"{arm_prefix}_{i}"
        success = robot.home_offset_reset(joint_name)

        if not success:
            print(f"Failed to reset {joint_name}")
            all_success = False

    if all_success:
        print(f"All {arm} arm joints reset successfully.")
        robot.disable_control_manager()
        time.sleep(2)

        print("power off")
        robot.power_off("48v")
        time.sleep(1)

        print("init")
        robot = initialize_robot(address, model_name, power=".*", servo="^(?!.*head).*")

        print("======================move_j======================")
        # if arm == "right":
        #     movej(robot, right_arm=zero_pose, minimum_time=5)
        # else:
        #     movej(robot, left_arm=zero_pose, minimum_time=5)
        movej(robot, right_arm=zero_pose, left_arm=zero_pose, minimum_time=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero + Offset Move")
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="a")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="^(?!.*head).*")
    parser.add_argument("--arm", type=str, required=True, choices=["right", "left"])
    args = parser.parse_args()

    main(
        address=args.address,
        model_name=args.model,
        power=args.power,
        servo=args.servo,
        arm=args.arm,
    )