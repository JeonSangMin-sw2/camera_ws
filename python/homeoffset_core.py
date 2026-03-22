import json
import logging
import time

import numpy as np
import rby1_sdk as rby


def load_offset_from_json(filename="calibration_result.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    offset_deg = np.array(data["joint_offset_deg"], dtype=np.float64)
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


def initialize_robot(address, model, power=".*", servo="^(?!.*head).*"):
    robot = rby.create_robot(address, model)

    if not robot.connect():
        raise RuntimeError(f"Failed to connect robot: {address}")

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            raise RuntimeError("Power on failed")

    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            raise RuntimeError("Servo on failed")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        robot.reset_fault_control_manager()

    robot.enable_control_manager()
    return robot


def apply_home_offset(
    address,
    model_name,
    arm,
    offset_rad,
    power=".*",
    servo="^(?!.*head).*",
):
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    if arm == "right":
        arm_dof = len(model.right_arm_idx)
        zero_pose = np.zeros(arm_dof)
        arm_prefix = "right_arm"
    elif arm == "left":
        arm_dof = len(model.left_arm_idx)
        zero_pose = np.zeros(arm_dof)
        arm_prefix = "left_arm"
    else:
        raise ValueError("arm must be 'right' or 'left'")

    offset_rad = np.array(offset_rad, dtype=np.float64).reshape(-1)

    if len(offset_rad) != arm_dof:
        raise RuntimeError(
            f"Offset size mismatch: expected {arm_dof}, got {len(offset_rad)}"
        )

    # 기존 CLI 로직 유지
    offset_to_apply = -offset_rad
    offset_deg = np.rad2deg(offset_to_apply)

    # 1) zero pose 이동
    ok = movej(robot, right_arm=zero_pose, left_arm=zero_pose, minimum_time=5)
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")

    time.sleep(2)
    # 2) offset pose 이동
    target_pose = zero_pose + offset_to_apply
    if arm == "right":
        ok = movej(robot, right_arm=target_pose, minimum_time=10)
    else:
        ok = movej(robot, left_arm=target_pose, minimum_time=10)

    if not ok:
        raise RuntimeError("Failed to move robot with offset pose")

    time.sleep(2)

    # 3) joint별 home offset reset
    failed_joints = []
    for i in range(arm_dof):
        joint_name = f"{arm_prefix}_{i}"
        success = robot.home_offset_reset(joint_name)
        if not success:
            failed_joints.append(joint_name)

    if failed_joints:
        raise RuntimeError(f"Failed to reset joints: {failed_joints}")

    # 4) 재초기화
    robot.disable_control_manager()
    time.sleep(2)

    robot.power_off("48v")
    time.sleep(2)

    robot = initialize_robot(address, model_name, power=".*", servo="^(?!.*head).*")

    # 5) 다시 zero pose 이동
    ok = movej(robot, right_arm=zero_pose, left_arm=zero_pose, minimum_time=5)
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose after reset")

    return {
        "status": "success",
        "arm": arm,
        "offset_deg": offset_deg.tolist(),
    }


def apply_home_offset_from_json(
    address,
    model_name,
    arm="right",
    json_path="calibration_result.json",
    power=".*",
    servo="^(?!.*head).*",
):
    offset_rad = load_offset_from_json(json_path)

    result = apply_home_offset(
        address=address,
        model_name=model_name,
        arm=arm,
        offset_rad=offset_rad,
        power=power,
        servo=servo,
    )

    result["source"] = "json"
    result["json_path"] = json_path
    return result