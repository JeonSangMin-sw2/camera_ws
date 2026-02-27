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

    print("\nSelect offset mode:")
    print("u → User input")
    print("j → Load from JSON")
    mode = input("Mode (u/j) = ").strip().lower()
    
    
    # =========================
    # 1️⃣ 사용자 입력 모드
    # =========================
    if mode == "u":
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

        if len(offset_rad) != right_arm_dof:
            print("JSON offset size mismatch.")
            return

        print("Loaded offset from JSON (deg):")
        print(np.rad2deg(offset_rad))

    else:
        print("Invalid mode selected.")
        return


    # 3️⃣ zero + offset
    target_right = zero_right + offset_rad

    print("Moving with offset...")
    print("Offset (deg):", offset_deg)

    movej(robot, right_arm=target_right, minimum_time=10)
    time.sleep(1)
    
    all_success = True

    for i in range(right_arm_dof):
        joint_name = f"right_arm_{i}"
        success = robot.home_offset_reset(joint_name)

        if not success:
            print(f"Failed to reset {joint_name}")
            all_success = False

    if all_success:
        print("All right arm joints reset successfully.")
        robot.disable_control_manager()
        time.sleep(2)
        print("power off")
        robot.power_off("48v")
        time.sleep(1)

        print("init")
        robot = initialize_robot(address, model, power=".*", servo="^(?!.*head).*")
        print("======================move_j======================")
        movej(robot, right_arm=zero_right, minimum_time=5)


    
    

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
