import rby1_sdk as rby
import argparse
import numpy as np
import logging
import time


def movej(robot, torso=None, right_arm=None, left_arm=None, minimum_time=5):
    rc = rby.BodyComponentBasedCommandBuilder()

    if right_arm is not None:
        rc.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(right_arm)
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


def main(address, model, power, servo):
    robot = initialize_robot(address, model, power, servo)

    model = robot.model()
    right_arm_dof = len(model.right_arm_idx)

    # 1️⃣ zero pose
    zero_right = np.zeros(right_arm_dof)

    print("Moving to zero pose...")
    movej(robot, right_arm=zero_right, minimum_time=5)

    # 2️⃣ 사용자 입력 받기
    print(f"\nEnter {right_arm_dof} joint offsets in DEG (space separated):")
    print("Example: 1 0 0 0 0 0 0")

    user_input = input("Offset (deg) = ")

    try:
        offset_deg = np.array([float(x) for x in user_input.split()])
    except:
        print("Invalid input.")
        return

    if len(offset_deg) != right_arm_dof:
        print(f"Need exactly {right_arm_dof} values.")
        return

    offset_rad = np.deg2rad(offset_deg)

    # 3️⃣ zero + offset
    target_right = zero_right + offset_rad

    print("Moving with offset...")
    print("Offset (deg):", offset_deg)

    movej(robot, right_arm=target_right, minimum_time=10)
    time.sleep(1)
    
    if robot.home_offset_reset("right_arm_3") == True:
        time.sleep(2)
        robot.disable_control_manager()
        time.sleep(0.1)
        print("power off")
        robot.power_off("48v")
        time.sleep(1)
        print("init")
        initialize_robot(address, model, power=".*", servo="^(?!.*head).*")
        print("======================move_j======================")
        movej(robot, right_arm=zero_right, minimum_time=5)
    else :
        print("failed to reset homeoffset!!!!!")
    # for i in range(right_arm_dof):
    #     robot.home_offset_reset(f"right_arm_{i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero + Offset Move")
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="a")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="^(?!.*head).*")
    args = parser.parse_args()

    main(
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
    )
