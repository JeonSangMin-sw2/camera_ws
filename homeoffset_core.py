import json
import logging
import time

import numpy as np
import rby1_sdk as rby


def load_offset_from_json(filename="calibration_result.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    if "joint_offset_deg" in data:
        arm_offset_deg = np.array(data["joint_offset_deg"], dtype=np.float64)
    elif (
        data.get("right_arm_joint_offset_deg") is not None
        and data.get("left_arm_joint_offset_deg") is not None
    ):
        arm_offset_deg = np.concatenate([
            np.array(data["right_arm_joint_offset_deg"], dtype=np.float64),
            np.array(data["left_arm_joint_offset_deg"], dtype=np.float64),
        ])
    else:
        raise KeyError("joint_offset_deg is required in calibration result JSON")

    head_offset_deg = data.get("head_joint_offset_deg")
    head_offset_rad = None
    if head_offset_deg is not None:
        head_offset_rad = np.deg2rad(np.array(head_offset_deg, dtype=np.float64))
    return np.deg2rad(arm_offset_deg), head_offset_rad


def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=5):
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

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(rc)
    if head is not None:
        cmd.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(head)
        )

    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(cmd),
        1,
    ).get()

    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        logging.error("Failed to conduct movej.")
        return False

    return True


def initialize_robot(address, model, power=".*", servo=".*"):
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

def move_robot_to_zero_pose(address, model_name, arm, power=".*", servo=".*", include_head=True):
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx))

    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose if include_head else None,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")

    return {
        "status": "success",
        "arm": arm,
        "message": "Robot moved to zero pose. Please compare it with the reference image.",
    }

def apply_home_offset(
    address,
    model_name,
    arm,
    offset_rad,
    head_offset_rad=None,
    power=".*",
    servo=".*",
    include_head=True,
):
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    right_arm_dof = len(model.right_arm_idx)
    left_arm_dof = len(model.left_arm_idx)

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    offset_rad = np.array(offset_rad, dtype=np.float64).reshape(-1)
    right_zero_pose = np.zeros(right_arm_dof)
    left_zero_pose = np.zeros(left_arm_dof)
    head_zero_pose = np.zeros(len(model.head_idx))
    if not include_head:
        head_offset_rad = None
    if head_offset_rad is not None:
        head_offset_rad = np.array(head_offset_rad, dtype=np.float64).reshape(-1)
        if len(head_offset_rad) != len(model.head_idx):
            raise RuntimeError(
                f"Head offset size mismatch: expected {len(model.head_idx)}, got {len(head_offset_rad)}"
            )

    if len(offset_rad) == right_arm_dof + left_arm_dof:
        apply_mode = "both"
        right_offset_rad = offset_rad[:right_arm_dof]
        left_offset_rad = offset_rad[right_arm_dof:]
    elif arm == "right" and len(offset_rad) == right_arm_dof:
        apply_mode = "right"
        right_offset_rad = offset_rad
        left_offset_rad = np.zeros(left_arm_dof)
    elif arm == "left" and len(offset_rad) == left_arm_dof:
        apply_mode = "left"
        right_offset_rad = np.zeros(right_arm_dof)
        left_offset_rad = offset_rad
    else:
        expected = f"{right_arm_dof + left_arm_dof} (both arms)"
        if arm == "right":
            expected += f" or {right_arm_dof} (right arm)"
        elif arm == "left":
            expected += f" or {left_arm_dof} (left arm)"
        raise RuntimeError(
            f"Offset size mismatch: expected {expected}, got {len(offset_rad)}"
        )

    # 기존 CLI 로직 유지
    right_offset_to_apply = -right_offset_rad
    left_offset_to_apply = -left_offset_rad
    offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
    offset_deg = np.rad2deg(offset_to_apply)
    right_offset_deg = np.rad2deg(right_offset_to_apply)
    left_offset_deg = np.rad2deg(left_offset_to_apply)
    head_offset_to_apply = None if head_offset_rad is None else -head_offset_rad
    head_offset_deg = None if head_offset_to_apply is None else np.rad2deg(head_offset_to_apply)

    # 1) zero pose 이동
    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")

    time.sleep(2)
    # 2) offset pose 이동
    head_target_pose = None
    if include_head:
        head_target_pose = head_zero_pose if head_offset_to_apply is None else (head_zero_pose + head_offset_to_apply)
    ok = movej(
        robot,
        right_arm=right_zero_pose + right_offset_to_apply,
        left_arm=left_zero_pose + left_offset_to_apply,
        head=head_target_pose if include_head else None,
        minimum_time=10,
    )

    if not ok:
        raise RuntimeError("Failed to move robot with offset pose")

    time.sleep(2)

    # 3) joint별 home offset reset
    failed_joints = []
    if apply_mode in ("right", "both"):
        for i in range(right_arm_dof):
            joint_name = f"right_arm_{i}"
            success = robot.home_offset_reset(joint_name)
            if not success:
                failed_joints.append(joint_name)

    if apply_mode in ("left", "both"):
        for i in range(left_arm_dof):
            joint_name = f"left_arm_{i}"
            success = robot.home_offset_reset(joint_name)
            if not success:
                failed_joints.append(joint_name)

    if head_offset_to_apply is not None:
        for i in range(len(model.head_idx)):
            joint_name = f"head_{i}"
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

    robot = initialize_robot(address, model_name, power=".*", servo=".*")

    # 5) 다시 zero pose 이동
    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose if include_head else None,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose after reset")

    return {
        "status": "success",
        "arm": apply_mode,
        "offset_deg": offset_deg.tolist(),
        "right_offset_deg": right_offset_deg.tolist(),
        "left_offset_deg": left_offset_deg.tolist(),
        "head_offset_deg": None if head_offset_deg is None else head_offset_deg.tolist(),
    }


def apply_home_offset_from_json(
    address,
    model_name,
    arm="right",
    json_path="calibration_result.json",
    power=".*",
    servo=".*",
    include_head=True,
):
    offset_rad, head_offset_rad = load_offset_from_json(json_path)

    result = apply_home_offset(
        address=address,
        model_name=model_name,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        power=power,
        servo=servo,
        include_head=include_head,
    )

    result["source"] = "json"
    result["json_path"] = json_path
    return result
