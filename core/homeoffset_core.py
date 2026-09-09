import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rby1_sdk as rby


def compare_home_offset_files(result_path, baseline_path):
    def read(path):
        if path is None or not Path(path).exists():
            return {}
        try:
            with open(path) as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return {}
    optimized, baseline = read(result_path), read(baseline_path)
    comparison = {}
    for side, key in (('right', 'right_arm_joint_offset_deg'),
                      ('left', 'left_arm_joint_offset_deg'), ('head', 'head_joint_offset_deg')):
        opt, base = optimized.get(key), baseline.get(key)
        difference = None
        if opt is not None and base is not None and len(opt) == len(base) and len(opt):
            difference = np.asarray(base) - np.asarray(opt)
        comparison[side] = dict(optimized=opt, baseline=base, difference=difference)
    return comparison


def move_connected_robot_to_zero(robot, model, include_head):
    head = np.zeros(len(model.head_idx)) if include_head and len(getattr(model, 'head_idx', [])) >= 2 else None
    if not movej(robot, right_arm=np.zeros(len(model.right_arm_idx)),
                 left_arm=np.zeros(len(model.left_arm_idx)), head=head, minimum_time=5):
        raise RuntimeError('Failed to move robot to zero pose')


def run_home_offset_operation(task_type, robot, model, *, json_path=None, label='',
                              arm='both', include_head=False, skip_init_pose=False,
                              setting_path=None, log_callback=None):
    if robot is None or model is None:
        raise RuntimeError('Robot is not connected.')
    if task_type == 'move_zero':
        return move_to_offset_candidate_from_json(robot=robot, model=model, arm=arm,
            json_path=str(json_path), include_head=include_head, minimum_time=10,
            move_zero_first=True)
    if task_type == 'move_check':
        return move_to_check_position_candidate_path(robot, model, json_path, label,
            arm, include_head, skip_init_pose, log_callback=log_callback)
    if task_type == 'apply':
        return apply_current_pose_home_offset(robot, model, arm, include_head,
            json_path=json_path, setting_path=setting_path, log_callback=log_callback)
    raise ValueError(f'Unknown home-offset operation: {task_type}')


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
    if (data.get('head_tilt_independent') is False
            or data.get('diagnostics', {}).get('head_tilt_mode') in
            ('effective_zero_gauge', 'camera_forward_gauge')):
        # Effective camera/tilt gauges must not be written to physical head homes.
        # Keep arm correction available; head requires an independent reference.
        logging.warning('Head offsets excluded: camera/tilt solution uses an effective gauge.')
        head_offset_deg = None
    diagnostics = data.get('diagnostics', {})
    if diagnostics and (not diagnostics.get('converged') or not diagnostics.get('observable')):
        raise ValueError('Cannot apply an unconverged or unobservable calibration result')
    head_offset_rad = None
    if head_offset_deg is not None:
        head_offset_rad = np.deg2rad(np.array(head_offset_deg, dtype=np.float64))
    return np.deg2rad(arm_offset_deg), head_offset_rad


def _split_arm_offset(model, arm, offset_rad):
    right_arm_dof = len(model.right_arm_idx)
    left_arm_dof = len(model.left_arm_idx)

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    offset_rad = np.array(offset_rad, dtype=np.float64).reshape(-1)
    if len(offset_rad) == right_arm_dof + left_arm_dof:
        return (
            "both",
            offset_rad[:right_arm_dof],
            offset_rad[right_arm_dof:],
        )
    if arm == "right" and len(offset_rad) == right_arm_dof:
        return "right", offset_rad, np.zeros(left_arm_dof)
    if arm == "left" and len(offset_rad) == left_arm_dof:
        return "left", np.zeros(right_arm_dof), offset_rad

    expected = f"{right_arm_dof + left_arm_dof} (both arms)"
    if arm == "right":
        expected += f" or {right_arm_dof} (right arm)"
    elif arm == "left":
        expected += f" or {left_arm_dof} (left arm)"
    raise RuntimeError(
        f"Offset size mismatch: expected {expected}, got {len(offset_rad)}"
    )


def _normalize_head_offset(model, head_offset_rad, include_head):
    head_dof = len(model.head_idx)
    if not include_head or head_offset_rad is None or head_dof == 0:
        return None, 0

    head_offset_rad = np.array(head_offset_rad, dtype=np.float64).reshape(-1)
    if len(head_offset_rad) > head_dof:
        raise RuntimeError(
            f"Head offset size mismatch: expected up to {head_dof}, got {len(head_offset_rad)}"
        )

    head_offset_full = np.zeros(head_dof, dtype=np.float64)
    head_offset_full[:len(head_offset_rad)] = head_offset_rad
    return head_offset_full, len(head_offset_rad)


def build_home_reset_baseline_data(robot, model, model_name=None, include_head=True):
    state = robot.get_state()
    q_full = np.array(state.position, dtype=np.float64).reshape(-1)

    right_offset_rad = q_full[model.right_arm_idx].copy()
    left_offset_rad = q_full[model.left_arm_idx].copy()
    head_offset_rad = None
    if include_head and len(model.head_idx) > 0:
        head_offset_rad = np.array([float(q_full[i]) for i in list(model.head_idx)], dtype=np.float64)

    data = {
        "joint_offset_deg": np.rad2deg(
            np.concatenate([right_offset_rad, left_offset_rad])
        ).tolist(),
        "right_arm_joint_offset_deg": np.rad2deg(right_offset_rad).tolist(),
        "left_arm_joint_offset_deg": np.rad2deg(left_offset_rad).tolist(),
        "head_joint_offset_deg": (
            None if head_offset_rad is None else np.rad2deg(head_offset_rad).tolist()
        ),
        "metadata": {
            "type": "home_reset_baseline",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "include_head": bool(include_head),
        },
    }
    return data


def save_home_reset_baseline_json(
    robot,
    model,
    output_dir,
    model_name=None,
    include_head=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "home_reset_baseline.json"

    data = build_home_reset_baseline_data(
        robot,
        model,
        model_name=model_name,
        include_head=include_head,
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path, data


def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=5):
    from core.robot_motion import move_joints_checked
    # Home-offset operations historically command a zero torso, regardless of the
    # optional torso argument; keep that convention separate from calibration motion.
    return move_joints_checked(robot, torso=np.zeros(6), right_arm=right_arm,
        left_arm=left_arm, head=head, minimum_time=minimum_time, priority=1,
        include_head=head is not None)


def initialize_robot(address, model, power="48v", servo=None, *, include_head=False):
    from core.robot_motion import initialize_robot_connection
    return initialize_robot_connection(address, model, power=power, servo=servo,
                                       include_head=include_head, unlimited_mode_enabled=False)

def move_robot_to_zero_pose(address, model_name, arm, power="48v", servo=None, include_head=True):
    if servo is None:
        servo = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
    robot = initialize_robot(address, model_name, power, servo, include_head=include_head)
    model = robot.model()

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if (include_head and hasattr(model, 'head_idx') and model.head_idx is not None and len(model.head_idx) >= 2) else None

    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")

    return {
        "status": "success",
        "arm": arm,
        "message": "Robot moved to zero pose. Please compare it with the reference image.",
    }


def move_to_offset_candidate(
    robot,
    model,
    arm,
    offset_rad,
    head_offset_rad=None,
    include_head=True,
    minimum_time=10,
    move_zero_first=True,
    settle_time=2.0,
):
    apply_mode, right_offset_rad, left_offset_rad = _split_arm_offset(
        model,
        arm,
        offset_rad,
    )

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    head_offset_full, head_offset_size = _normalize_head_offset(
        model,
        head_offset_rad,
        include_head,
    )

    right_offset_to_apply = -right_offset_rad
    left_offset_to_apply = -left_offset_rad
    head_offset_to_apply = None if head_offset_full is None else -head_offset_full

    if move_zero_first:
        ok = movej(
            robot,
            right_arm=right_zero_pose,
            left_arm=left_zero_pose,
            head=head_zero_pose,
            minimum_time=5,
        )
        if not ok:
            raise RuntimeError("Failed to move robot to zero pose")
        time.sleep(settle_time)

    head_target_pose = None
    if include_head:
        head_target_pose = (
            head_zero_pose
            if head_offset_to_apply is None
            else head_zero_pose + head_offset_to_apply
        )

    ok = movej(
        robot,
        right_arm=right_zero_pose + right_offset_to_apply,
        left_arm=left_zero_pose + left_offset_to_apply,
        head=head_target_pose if include_head else None,
        minimum_time=minimum_time,
    )
    if not ok:
        raise RuntimeError("Failed to move robot with offset pose")

    time.sleep(settle_time)

    offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
    head_offset_deg = None
    if head_offset_to_apply is not None:
        head_offset_deg = np.rad2deg(head_offset_to_apply[:head_offset_size]).tolist()

    return {
        "status": "success",
        "arm": apply_mode,
        "offset_deg": np.rad2deg(offset_to_apply).tolist(),
        "right_offset_deg": np.rad2deg(right_offset_to_apply).tolist(),
        "left_offset_deg": np.rad2deg(left_offset_to_apply).tolist(),
        "head_offset_deg": head_offset_deg,
    }


def move_to_offset_candidate_from_json(
    robot,
    model,
    arm,
    json_path,
    include_head=True,
    minimum_time=10,
    move_zero_first=True,
):
    offset_rad, head_offset_rad = load_offset_from_json(json_path)
    result = move_to_offset_candidate(
        robot=robot,
        model=model,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        include_head=include_head,
        minimum_time=minimum_time,
        move_zero_first=move_zero_first,
    )
    result["source"] = "json"
    result["json_path"] = str(json_path)
    return result


def validate_home_offset_joint_limits(robot, model, arm="both", include_head=True, log_cb=None):
    """
    Validates that setting home offset from the current pose will NOT cause any joint
    to exceed its allowed physical joint limits when commanded to ready poses.
    Returns (is_valid, error_message).
    """
    if robot is None or model is None:
        return True, ""

    try:
        state = robot.get_state()
        q_current = np.array(state.position, dtype=np.float64).reshape(-1)

        if hasattr(robot, "get_dynamics"):
            dyn_model = robot.get_dynamics()
        elif hasattr(model, "get_dynamics_model"):
            dyn_model = model.get_dynamics_model()
        else:
            dyn_model = None

        if dyn_model is not None:
            q_lower = np.array(dyn_model.get_limit_q_lower(state), dtype=np.float64).reshape(-1)
            q_upper = np.array(dyn_model.get_limit_q_upper(state), dtype=np.float64).reshape(-1)
        else:
            q_lower = np.full(len(q_current), -np.pi)
            q_upper = np.full(len(q_current), np.pi)

        # 1. Posture Direction Sign Validation for Home Offset Reset
        # Right Arm: J1 (Shoulder Roll) <= 0, J3 (Elbow) <= 0
        # Left Arm:  J1 (Shoulder Roll) >= 0, J3 (Elbow) <= 0
        tol = np.radians(5.0)  # 5° buffer
        invalid_reasons = []

        if arm in ("right", "both"):
            r_j1 = q_current[model.right_arm_idx[1]]
            r_j3 = q_current[model.right_arm_idx[3]]
            if r_j1 > tol:
                invalid_reasons.append(f"오른쪽 Shoulder Roll(J1)이 양수(+{np.degrees(r_j1):.1f}°)입니다. 음수(-) 방향(몸 안쪽)이어야 합니다.")
            if r_j3 > tol:
                invalid_reasons.append(f"오른쪽 Elbow(J3)가 양수(+{np.degrees(r_j3):.1f}°)입니다. 음수(-) 방향(뒤쪽)이어야 합니다.")

        if arm in ("left", "both"):
            l_j1 = q_current[model.left_arm_idx[1]]
            l_j3 = q_current[model.left_arm_idx[3]]
            if l_j1 < -tol:
                invalid_reasons.append(f"왼쪽 Shoulder Roll(J1)이 음수({np.degrees(l_j1):.1f}°)입니다. 양수(+) 방향(몸 안쪽)이어야 합니다.")
            if l_j3 > tol:
                invalid_reasons.append(f"왼쪽 Elbow(J3)가 양수(+{np.degrees(l_j3):.1f}°)입니다. 음수(-) 방향(뒤쪽)이어야 합니다.")

        if invalid_reasons:
            err_msg = (
                "⚠️ 유효하지 않은 홈 오프셋 리셋 자세입니다!\n\n"
                + "\n".join(f"- {reason}" for reason in invalid_reasons)
                + "\n\n안내 가이드라인(이미지 및 지침 사항)을 참고하여 각 관절을 올바른 방향으로 위치시킨 후 다시 리셋을 시도해 주세요."
            )
            if log_cb:
                log_cb(f"[ERROR] {err_msg}")
            return False, err_msg

        # Missing/malformed motion configuration must fail before a home write.
        from core.config_store import RobotConfig
        ready_poses_dict = RobotConfig.load().ready_poses

        # Collect ready poses to test (marker, joint, check_calib)
        poses_to_check = []

        for ver_key in ["v1.2", "v1.3"]:
            if ver_key in ready_poses_dict:
                ver_data = ready_poses_dict[ver_key]
                if "marker" in ver_data:
                    poses_to_check.append(("marker", ver_data["marker"]))
                if "check_calib" in ver_data:
                    poses_to_check.append(("check_calib", ver_data["check_calib"]))
                if "joint" in ver_data:
                    for mode_key, mode_data in ver_data["joint"].items():
                        poses_to_check.append((f"joint_{mode_key}", mode_data))

        # Check joint bounds for selected arms/head
        joint_indices_to_check = []
        if arm in ("right", "both"):
            for i, idx in enumerate(model.right_arm_idx):
                joint_indices_to_check.append((f"right_arm_{i}", idx, "right_arm", i))
        if arm in ("left", "both"):
            for i, idx in enumerate(model.left_arm_idx):
                joint_indices_to_check.append((f"left_arm_{i}", idx, "left_arm", i))
        if include_head and hasattr(model, 'head_idx'):
            for i, idx in enumerate(model.head_idx):
                joint_indices_to_check.append((f"head_{i}", idx, "head", i))

        for pose_label, pose_data in poses_to_check:
            for joint_name, idx, side, joint_i in joint_indices_to_check:
                if side in pose_data and len(pose_data[side]) > joint_i:
                    q_ready_val = np.radians(pose_data[side][joint_i])
                    q_offset_val = q_current[idx]
                    q_expected = q_ready_val + q_offset_val

                    min_lim = q_lower[idx]
                    max_lim = q_upper[idx]

                    if q_expected < min_lim or q_expected > max_lim:
                        exp_deg = np.degrees(q_expected)
                        min_deg = np.degrees(min_lim)
                        max_deg = np.degrees(max_lim)
                        off_deg = np.degrees(q_offset_val)

                        err_msg = (
                            f"Joint '{joint_name}' angle {exp_deg:+.1f}° (ready {pose_data[side][joint_i]:+.1f}° + offset {off_deg:+.1f}°) "
                            f"exceeds physical joint limits [{min_deg:+.1f}°, {max_deg:+.1f}°] under pose target '{pose_label}'!"
                        )
                        if log_cb:
                            log_cb(f"[ERROR] {err_msg}")
                        return False, err_msg

        return True, ""
    except Exception as e:
        if log_cb:
            log_cb(f"[WARN] Joint limit validation exception: {e}")
        return True, ""


def reset_current_pose_home_offsets(
    robot,
    model,
    arm="both",
    include_head=True,
    log_cb=None,
    power_cycle=True,
):
    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    if log_cb is not None:
        log_cb("Starting Home Offset Reset from current pose...")

    # Safety Check: Validate joint limits against ready poses before performing reset
    is_valid, err_msg = validate_home_offset_joint_limits(
        robot, model, arm=arm, include_head=include_head, log_cb=log_cb
    )
    if not is_valid:
        if log_cb is not None:
            log_cb(f"[ERROR] Home Offset Reset aborted due to joint limit violation: {err_msg}")
        return {
            "status": "failed",
            "success": False,
            "arm": arm,
            "include_head": bool(include_head),
            "failed_joints": [],
            "error": err_msg
        }

    robot.disable_control_manager()
    time.sleep(0.5)

    failed_joints = []

    def reset_joint_group(prefix, dof, label):
        for i in range(dof):
            joint_name = f"{prefix}_{i}"
            success = robot.home_offset_reset(joint_name)
            if not success:
                failed_joints.append(joint_name)
                if log_cb is not None:
                    log_cb(f"Failed to reset {label} joint: {joint_name}")
            elif log_cb is not None:
                log_cb(f"Reset {label} joint OK: {joint_name}")

    if arm in ("right", "both"):
        reset_joint_group("right_arm", len(model.right_arm_idx), "right arm")
    if arm in ("left", "both"):
        reset_joint_group("left_arm", len(model.left_arm_idx), "left arm")
    if include_head:
        reset_joint_group("head", len(model.head_idx), "head")

    all_success = len(failed_joints) == 0
    if log_cb is not None:
        if all_success:
            log_cb("All selected joints reset successfully!")
        else:
            log_cb("Some joints failed to reset. Proceeding with power cycle...")

    if power_cycle:
        if log_cb is not None:
            log_cb("Disabling control manager and waiting 2 seconds...")
        robot.disable_control_manager()
        time.sleep(2.0)

        if log_cb is not None:
            log_cb("Powering off overall power (.*)...")
        robot.power_off(".*")
        time.sleep(1.5)

    return {
        "status": "success" if all_success else "partial_failure",
        "success": all_success,
        "arm": arm,
        "include_head": bool(include_head),
        "failed_joints": failed_joints,
    }


def apply_home_offset(
    address,
    model_name,
    arm,
    offset_rad,
    head_offset_rad=None,
    power="48v",
    servo=None,
    include_head=True,
):
    if servo is None:
        servo = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
    robot = initialize_robot(address, model_name, power, servo, include_head=include_head)
    model = robot.model()

    move_result = move_to_offset_candidate(
        robot=robot,
        model=model,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        include_head=include_head,
        minimum_time=10,
        move_zero_first=True,
    )

    reset_result = reset_current_pose_home_offsets(
        robot=robot,
        model=model,
        arm=move_result["arm"],
        include_head=include_head and head_offset_rad is not None,
    )
    if not reset_result["success"]:
        raise RuntimeError(f"Failed to reset joints: {reset_result['failed_joints']}")

    robot = initialize_robot(address, model_name, power=power, servo=servo, include_head=include_head)

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose after reset")

    move_result["reset_result"] = reset_result
    return move_result


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


def reset_home_offsets(robot, model, log_cb=None):
    result = reset_current_pose_home_offsets(
        robot,
        model,
        arm="both",
        include_head=True,
        log_cb=log_cb,
    )
    return result["success"]


def move_to_check_position_candidate_path(robot, model, json_path, label, arm, include_head,
                                          skip_init_pose=False, log_callback=None):
    log = log_callback or (lambda message: None)

    # Load offsets from json
    from core.homeoffset_core import load_offset_from_json, _split_arm_offset, _normalize_head_offset, movej
    offset_rad, head_offset_rad = load_offset_from_json(str(json_path))

    apply_mode, right_offset_rad, left_offset_rad = _split_arm_offset(
        model,
        arm,
        offset_rad,
    )

    right_offset_to_apply = -right_offset_rad
    left_offset_to_apply = -left_offset_rad
    head_offset_full, head_offset_size = _normalize_head_offset(
        model,
        head_offset_rad,
        include_head,
    )
    head_offset_to_apply = None if head_offset_full is None else -head_offset_full

    # Determine version key
    version_key = "v1.2"
    if robot is not None:
        try:
            robot_info = robot.get_robot_info()
            raw_version = robot_info.robot_model_version
            if "1.3" in raw_version:
                version_key = "v1.3"
        except Exception:
            pass

    from core.config_store import RobotConfig
    check_calib_joints = RobotConfig.load().ready_poses[version_key]['check_calib']

    # 1. Move to 1st ready pose (like check_calibration_state)
    if not skip_init_pose:
        active_arms = []
        if arm in ("right", "both"):
            active_arms.append("right")
        if arm in ("left", "both"):
            active_arms.append("left")

        log(f"\n[Check Position] Step 1: Moving to Joint Ready Pose...")
        ok = movej(
            robot,
            torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
            right_arm=np.deg2rad([-45, -30, 0, -90, 0, 45, 0]) if "right" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
            left_arm=np.deg2rad([-45, 30, 0, -90, 0, 45, 0]) if "left" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
            minimum_time=5
        )
        if not ok:
            raise RuntimeError("Failed to move robot to Step 1 Ready Pose")
        time.sleep(2.0)
    else:
        log("[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)")

    # 2. Move to Check Position with offsets added
    log(f"[Check Position] Step 2: Moving to Check Pose with Offsets...")

    right_target = np.deg2rad(check_calib_joints["right_arm"]) + right_offset_to_apply
    left_target = np.deg2rad(check_calib_joints["left_arm"]) + left_offset_to_apply

    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    head_target_pose = None
    if include_head:
        head_target_pose = (
            head_zero_pose
            if head_offset_to_apply is None
            else head_zero_pose + head_offset_to_apply
        )

    ok = movej(
        robot,
        torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
        right_arm=right_target,
        left_arm=left_target,
        head=head_target_pose if include_head else None,
        minimum_time=10
    )
    if not ok:
        raise RuntimeError("Failed to move robot to Step 2 Check Pose")
    time.sleep(2.0)

    offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
    head_offset_deg = None
    if head_offset_to_apply is not None:
        head_offset_deg = np.rad2deg(head_offset_to_apply[:head_offset_size]).tolist()

    result = {
        "status": "success",
        "arm": apply_mode,
        "offset_deg": np.rad2deg(offset_to_apply).tolist(),
        "right_offset_deg": np.rad2deg(right_offset_to_apply).tolist(),
        "left_offset_deg": np.rad2deg(left_offset_to_apply).tolist(),
        "head_offset_deg": head_offset_deg,
    }

    log(f"\n===== HOME OFFSET PREVIEW: {label} =====")
    log(f"JSON: {json_path}")
    log(f"Arm: {result['arm']}")
    if result.get("right_offset_deg") is not None:
        log(f"Right move offset (deg): {result['right_offset_deg']}")
    if result.get("left_offset_deg") is not None:
        log(f"Left move offset (deg): {result['left_offset_deg']}")
    if result.get("head_offset_deg") is not None:
        log(f"Head move offset (deg): {result['head_offset_deg']}")
    log("Preview move complete. Inspect the robot pose before applying.")
    return result


def apply_current_pose_home_offset(robot, model, arm, include_head, json_path=None, *,
                                   setting_path=None, log_callback=None):
    log = log_callback or (lambda message: None)
    result = reset_current_pose_home_offsets(
        robot,
        model,
        arm=arm,
        include_head=include_head,
    )

    # If camera was optimized, save the new camera pose to setting.yaml
    if json_path is not None and os.path.exists(json_path):
        try:
            import json
            with open(json_path, "r") as f:
                data = json.load(f)

            mount_to_cam_new = data.get("mount_to_cam_new")
            head_base_to_cam_new = data.get("head_base_to_cam_new")

            if mount_to_cam_new or head_base_to_cam_new:
                from core.config_store import update_yaml
                key = "mount_to_cam" if include_head else "head_base_to_cam"
                values = mount_to_cam_new if include_head else head_base_to_cam_new
                if values is not None:
                    update_yaml(setting_path, {"camera": {key: values}})
                    result['camera_save_success'] = True
                    log(f"[APPLY] Saved optimized {key} to setting.yaml.")
        except Exception as e:
            result['camera_save_success'] = False
            log(f"[ERROR] Robot home reset already completed, but camera settings were NOT saved: {e}")

    # Robot reconnection should be done in the main thread to avoid GUI thread safety issues.
    # We will signal the caller to handle the reconnection.
    result['needs_reconnect'] = True
    return result
