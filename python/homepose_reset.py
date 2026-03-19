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

    robot.enable_control_manager(unlimited_mode_enabled=True)

    return robot

def main(address, model_name, power, servo):
    robot = initialize_robot(address, model_name, power, servo)

    model = robot.model()

    right_dof = len(model.right_arm_idx)
    left_dof = len(model.left_arm_idx)

    # Control manager enable
    robot.enable_control_manager()

    print("\n==============================")
    print("Manual Teaching Required")
    print("==============================")
    print("Please move BOTH arms close to the home position using direct teaching.")
    print("After positioning the robot, press ENTER to continue.")
    input("Press ENTER to perform home offset reset...")

    time.sleep(1)

    all_success = True

    # Right arm reset
    for i in range(right_dof):
        joint_name = f"right_arm_{i}"
        success = robot.home_offset_reset(joint_name)

        if not success:
            print(f"Failed to reset {joint_name}")
            all_success = False
        else:
            print(f"{joint_name} reset OK")

    # Left arm reset
    for i in range(left_dof):
        joint_name = f"left_arm_{i}"
        success = robot.home_offset_reset(joint_name)

        if not success:
            print(f"Failed to reset {joint_name}")
            all_success = False
        else:
            print(f"{joint_name} reset OK")

    if all_success:
        print("\nAll arm joints reset successfully.")

        robot.disable_control_manager()
        time.sleep(2)

        print("Power off")
        robot.power_off("48v")
        time.sleep(1)

        print("Reinitialize robot")
        robot = initialize_robot(address, model_name, power=".*", servo="^(?!.*head).*")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero + Offset Move")
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="a")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="^(?!.*head).*")
    args = parser.parse_args()

    main(
        address=args.address,
        model_name=args.model,
        power=args.power,
        servo=args.servo,
    )
                                                                                                