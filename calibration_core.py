import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rby1_sdk as rby
import yaml

from marker_detection import Marker_Transform


np.set_printoptions(suppress=True, precision=6)

BASE_DIR = Path(__file__).resolve().parent
SETTING_PATH = BASE_DIR / "config" / "setting.yaml"
DEFAULT_LAMBDA_CAM = 1.0
ARM_SIDES = ("right", "left")
ARM_OPTIMIZATION_NDOF = {14, 16, 20, 22}
HEAD_OPTIMIZATION_NDOF = {2, 16, 22}
CAMERA_OPTIMIZATION_NDOF = {6, 20, 22}
D2R = np.pi / 180.0


# ============================================================
# Auto data-collection motion
# ============================================================

@dataclass
class AutoCollectionConfig:
    sampling_mode: str = "axis"
    pivot_x: float = 0.40
    pivot_y: float = 0.0
    arm_gap_y: float = 0.35
    arm_gap_z: float = 0.0
    head_height_z: float = 0.27
    tip_offset_x: float = 0.0
    tip_offset_y: float = 0.0
    tip_offset_z: float = 0.0
    max_roll_deg: float = 10.0
    max_pitch_deg: float = 10.0
    max_yaw_deg: float = 10.0
    init_roll_deg: float = 0.0
    init_pitch_deg: float = 90.0
    init_yaw_deg: float = -90.0
    roll_steps: int = 10
    pitch_steps: int = 10
    yaw_steps: int = 10
    move_time: float = 1.2
    settle_time: float = 0.6
    hold_time: float = 3.0
    priority: int = 10
    head_max_deg: float = 3.0


def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_T(R, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def compute_target_T(pivot, tip_offset, roll_deg, pitch_deg, yaw_deg):
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    p_ee = pivot - R @ tip_offset
    return make_T(R, p_ee)


def linspace_symmetric(max_abs_deg, steps):
    if steps <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-max_abs_deg, max_abs_deg, steps, dtype=np.float64)


def build_rpy_samples(mode, roll_vals, pitch_vals, yaw_vals):
    if mode == "grid":
        return [
            (float(r), float(p), float(y))
            for r, p, y in itertools.product(roll_vals, pitch_vals, yaw_vals)
        ]

    samples = [(0.0, 0.0, 0.0)]
    for v in roll_vals:
        if abs(v) < 1e-12:
            continue
        samples.append((float(v), 0.0, 0.0))
    for v in pitch_vals:
        if abs(v) < 1e-12:
            continue
        samples.append((0.0, float(v), 0.0))
    for v in yaw_vals:
        if abs(v) < 1e-12:
            continue
        samples.append((0.0, 0.0, float(v)))
    return samples


def build_auto_motion_plan(config):
    pivot_z = config.head_height_z
    half_gap = max(float(config.arm_gap_y), 0.0) * 0.5
    half_gap_z = max(float(config.arm_gap_z), 0.0) * 0.5
    center_y = config.pivot_y
    pivot_right = np.array(
        [config.pivot_x, center_y - half_gap, pivot_z - half_gap_z],
        dtype=np.float64,
    )
    pivot_left = np.array(
        [config.pivot_x, center_y + half_gap, pivot_z + half_gap_z],
        dtype=np.float64,
    )
    tip_offset = np.array(
        [config.tip_offset_x, config.tip_offset_y, config.tip_offset_z],
        dtype=np.float64,
    )
    roll_vals = linspace_symmetric(config.max_roll_deg, config.roll_steps)
    pitch_vals = linspace_symmetric(config.max_pitch_deg, config.pitch_steps)
    yaw_vals = linspace_symmetric(config.max_yaw_deg, config.yaw_steps)
    rpy_samples = build_rpy_samples(config.sampling_mode, roll_vals, pitch_vals, yaw_vals)
    return {
        "pivot_right": pivot_right,
        "pivot_left": pivot_left,
        "tip_offset": tip_offset,
        "rpy_samples": rpy_samples,
    }


def compute_auto_head_target(base_head_q, roll_delta_deg, pitch_delta_deg, yaw_delta_deg, head_max_deg):
    yaw_cmd_deg = float(np.clip(roll_delta_deg + yaw_delta_deg, -head_max_deg, head_max_deg))
    pitch_cmd_deg = float(np.clip(pitch_delta_deg, -head_max_deg, head_max_deg))
    return base_head_q + np.deg2rad(np.array([yaw_cmd_deg, pitch_cmd_deg], dtype=np.float64))


def make_dual_arm_head_cmd(T_right, T_left, head_position, min_time=1.2, hold_time=3.0):
    body = rby.BodyComponentBasedCommandBuilder()

    if T_right is not None:
        body.set_right_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_right", T_right, 0.2, 0.5, 0.3)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold_time))
            .set_minimum_time(min_time)
        )

    if T_left is not None:
        body.set_left_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_left", T_left, 0.2, 0.5, 0.3)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold_time))
            .set_minimum_time(min_time)
        )

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(body)
    if head_position is not None:
        cmd.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.asarray(head_position, dtype=np.float64))
            .set_minimum_time(min_time)
        )

    return rby.RobotCommandBuilder().set_command(cmd)


def execute_auto_motion_step(robot, config, motion_plan, base_head_q, rpy_delta_deg):
    roll_delta_deg, pitch_delta_deg, yaw_delta_deg = rpy_delta_deg
    roll_deg = config.init_roll_deg + roll_delta_deg
    pitch_deg = config.init_pitch_deg + pitch_delta_deg
    yaw_deg = config.init_yaw_deg + yaw_delta_deg

    T_right = compute_target_T(
        motion_plan["pivot_right"],
        motion_plan["tip_offset"],
        roll_deg,
        pitch_deg,
        yaw_deg,
    )
    T_left = compute_target_T(
        motion_plan["pivot_left"],
        motion_plan["tip_offset"],
        roll_deg,
        pitch_deg,
        -yaw_deg,
    )
    head_target = compute_auto_head_target(
        base_head_q=base_head_q,
        roll_delta_deg=roll_delta_deg,
        pitch_delta_deg=pitch_delta_deg,
        yaw_delta_deg=yaw_delta_deg,
        head_max_deg=config.head_max_deg,
    )

    cmd = make_dual_arm_head_cmd(
        T_right=T_right,
        T_left=T_left,
        head_position=head_target,
        min_time=config.move_time,
        hold_time=config.hold_time,
    )
    rv = robot.send_command(cmd, config.priority).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Auto motion command failed.")

    time.sleep(config.settle_time)
    return {
        "rpy_deg": (roll_deg, pitch_deg, yaw_deg),
        "rpy_delta_deg": (roll_delta_deg, pitch_delta_deg, yaw_delta_deg),
        "head_target_rad": np.asarray(head_target, dtype=np.float64),
    }


def move_to_auto_ready_pose(robot, minimum_time=3.0, priority=10):
    # Match the ready pose from developer/auto_data_collecting.py (go_ready).
    q_torso = np.array([0, 60, -120, 60, 0, 0], dtype=np.float64) * D2R
    q_right = np.deg2rad(
        np.array([-43.975, -21.385, -20.251, -104.030, 83.705, -62.694, 33.967], dtype=np.float64)
    )
    q_left = np.deg2rad(
        np.array([-43.975, 21.385, 20.251, -104.030, -83.705, -62.694, -33.967], dtype=np.float64)
    )
    q = np.concatenate([q_torso, q_right, q_left])

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(minimum_time)
            )
        )
    )
    rv = robot.send_command(cmd, priority).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to auto ready pose.")


# ============================================================
# Lie algebra utilities
# ============================================================

def adjoint(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = p_hat @ R
    return Ad


def make_transform(data):
    # data: [x, y, z, roll, pitch, yaw] (xyz: m, rpy: deg)
    x, y, z = data[:3]
    roll, pitch, yaw = np.deg2rad(data[3:])

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    T = np.eye(4, dtype=np.float64)
    T[0, 0] = cy * cp
    T[0, 1] = sr * sp * cy - cr * sy
    T[0, 2] = cr * sp * cy + sr * sy
    T[0, 3] = x

    T[1, 0] = sy * cp
    T[1, 1] = sr * sp * sy + cr * cy
    T[1, 2] = cr * sp * sy - sr * cy
    T[1, 3] = y

    T[2, 0] = -sp
    T[2, 1] = cp * sr
    T[2, 2] = cp * cr
    T[2, 3] = z
    return T


def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def se3_exp(xi):
    w = xi[:3]
    v = xi[3:]
    R = so3_exp(w)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]) / theta

        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([
        w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]
    ])


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = t
    else:
        w_hat = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]) / theta

        A = (
            np.eye(3)
            - 0.5 * w_hat
            + (1 / theta**2) * (1 - theta / (2 * np.tan(theta / 2))) * (w_hat @ w_hat)
        )
        v = A @ t

    return np.hstack([w, v])


def rot_to_euler_zyx(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


# ============================================================
# Config / helpers
# ============================================================

def load_npz_dataset(path):
    data = np.load(path)
    q_arm = data["q_arm"] if "q_arm" in data else data["q"]
    q_head = data["q_head"] if "q_head" in data else None
    return q_arm, q_head, data["marker"]


def validate_dataset_for_ndof(ndof, q_arm, q_head, T_meas):
    if len(q_arm) != len(T_meas):
        raise RuntimeError(
            f"Dataset size mismatch: q_arm={len(q_arm)}, marker={len(T_meas)}"
        )

    if q_head is not None and len(q_head) != len(q_arm):
        raise RuntimeError(
            f"Dataset size mismatch: q_head={len(q_head)}, q_arm={len(q_arm)}"
        )

    if q_head is None:
        raise RuntimeError(
            "Head-mounted camera calibration requires `q_head`, but the loaded npz does not contain it."
        )

    if q_arm.ndim != 2:
        raise RuntimeError(f"Expected q_arm to be a 2D array, got shape {q_arm.shape}")

    if q_arm.shape[1] != 14:
        raise RuntimeError(
            f"Unsupported q_arm width {q_arm.shape[1]}. Expected 14 (right+left arms)."
        )

    if T_meas.ndim != 4 or T_meas.shape[1:] != (2, 4, 4):
        raise RuntimeError(
            f"Expected marker measurements with shape (N, 2, 4, 4), got {T_meas.shape}"
        )


def split_arm_offsets(q_offset):
    q_offset = np.asarray(q_offset, dtype=np.float64).reshape(-1)
    if len(q_offset) == 14:
        return q_offset[:7], q_offset[7:]
    return q_offset, None


def save_npz_dataset(path, q_arm, T_meas, q_head=None):
    save_kwargs = {
        "q": q_arm,
        "q_arm": q_arm,
        "marker": T_meas,
    }
    if q_head is not None:
        save_kwargs["q_head"] = q_head
    np.savez_compressed(path, **save_kwargs)


def save_result(path, q_offset, xi_t5_cam, q_head_offset=None):
    right_arm_offset, left_arm_offset = split_arm_offsets(q_offset)
    result_dict = {
        "joint_offset_deg": np.rad2deg(q_offset).tolist(),
        "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist(),
        "left_arm_joint_offset_deg": (
            np.rad2deg(left_arm_offset).tolist()
            if left_arm_offset is not None else None
        ),
        "head_joint_offset_deg": (
            np.rad2deg(q_head_offset).tolist()
            if q_head_offset is not None else None
        ),
        "xi_t5_cam": np.asarray(xi_t5_cam).tolist(),
        "xi_cam": np.asarray(xi_t5_cam).tolist(),
    }
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)


def create_robot(ip, model_name="a"):
    robot = rby.create_robot(ip, model_name)
    robot.connect()
    time.sleep(1)
    robot.power_on(".*")
    time.sleep(1)
    robot.servo_on(".*")
    time.sleep(2) 
    robot.reset_fault_control_manager()
    robot.enable_control_manager(False)
    return robot


def load_camera_nominals():
    with open(SETTING_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    camera_cfg = config.get("camera", {})
    mount_to_cam_nom = camera_cfg.get("mount_to_cam", camera_cfg.get("T5_to_cam"))
    if mount_to_cam_nom is None:
        raise KeyError(
            "camera.mount_to_cam (or legacy camera.T5_to_cam) is required in setting.yaml"
        )

    return {
        "mount_to_cam_nom": mount_to_cam_nom,
        "camera_mount_link": camera_cfg.get("camera_mount_link", "link_head_2"),
        "ee_to_marker_left": camera_cfg["Tf_to_marker_left"],
        "ee_to_marker_right": camera_cfg["Tf_to_marker_right"],
    }


def get_arm_config(model, arm):
    camera_nominals = load_camera_nominals()

    if arm == "right":
        return {
            "arm_idx": model.right_arm_idx[:7],
            "ee_link": "ee_right",
            "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
            "ee_to_marker_nom": camera_nominals["ee_to_marker_right"],
        }

    return {
        "arm_idx": model.left_arm_idx[:7],
        "ee_link": "ee_left",
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "ee_to_marker_nom": camera_nominals["ee_to_marker_left"],
    }


def get_both_arm_config(model):
    camera_nominals = load_camera_nominals()
    return {
        "arm_idx": np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]]),
        "ee_links": {
            "right": "ee_right",
            "left": "ee_left",
        },
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "ee_to_marker_nom": {
            "right": camera_nominals["ee_to_marker_right"],
            "left": camera_nominals["ee_to_marker_left"],
        },
    }


def get_head_config(model):
    camera_nominals = load_camera_nominals()
    return {
        "head_idx": model.head_idx[:2],
        "camera_link": camera_nominals["camera_mount_link"],
    }


def compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5"):
    state = dyn_model.make_state(
        [base_link, ee_link],
        robot.model().robot_joint_names
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    return state, dyn_model.compute_transformation(state, 0, 1)


def prepare_q_full(q_nominal, arm_idx, q_cmd, q_offset=None, head_idx=None, q_head=None, q_head_offset=None):
    q_full = q_nominal.copy()
    q_full[arm_idx] = q_cmd if q_offset is None else (q_cmd + q_offset)
    if head_idx is not None and q_head is not None:
        q_full[head_idx] = q_head if q_head_offset is None else (q_head + q_head_offset)
    return q_full


# ============================================================
# Capture dataset
# ============================================================

def create_live_marker_transform():
    marker_transform = Marker_Transform(
        serial_number=None,
        monitoring=False
    )
    marker_transform.marker_detection.set_marker_type("plate")
    return marker_transform


def capture_one_sample(robot, arm_idx, marker_transform, sampling_time=2, side="all", head_idx=None):
    state = robot.get_state()
    q_full = state.position.copy()
    q_arm = q_full[arm_idx].copy()
    q_head = q_full[head_idx].copy() if head_idx is not None else None

    result = marker_transform.get_marker_transform(sampling_time=sampling_time, side=side)
    if result is None:
        return None, None, None

    # side="all" returns [right, left] where each entry is a flattened 4x4.
    # If either side is missing, skip this sample gracefully.
    if side == "all":
        if len(result) < 2 or result[0] is None or result[1] is None:
            return None, None, None

        def _to_tf(flat_tf):
            arr = np.asarray(flat_tf, dtype=np.float64).reshape(-1)
            if arr.size != 16:
                raise RuntimeError(
                    f"Expected one marker transform to contain 16 values, got shape {np.asarray(flat_tf).shape}"
                )
            return arr.reshape(4, 4)

        T_right = _to_tf(result[0])
        T_left = _to_tf(result[1])
        return q_arm, q_head, np.stack([T_right, T_left], axis=0)

    T_meas = np.asarray(result, dtype=np.float64).reshape(-1)
    if T_meas.size != 16:
        raise RuntimeError(
            f"Expected marker transform with 16 values for side='{side}', got shape {np.asarray(result).shape}"
        )
    return q_arm, q_head, T_meas.reshape(4, 4)


def capture_dataset(robot, arm_idx, marker_transform, head_idx=None):
    q_arm_list = []
    q_head_list = []
    T_meas_list = []

    print("\nPress 'e' + Enter to capture both arms")
    print("Press 'q' + Enter to quit\n")

    while True:
        key = input().strip()

        if key == "e":
            q_arm, q_head, T_meas = capture_one_sample(
                robot=robot,
                arm_idx=arm_idx,
                marker_transform=marker_transform,
                side="all",
                head_idx=head_idx,
            )
            if T_meas is None:
                print("Marker not detected.")
                continue

            q_arm_list.append(q_arm)
            if q_head is not None:
                q_head_list.append(q_head)
            T_meas_list.append(T_meas)

            print(f"Captured sample {len(q_arm_list)}")
            print("q_arm =", np.round(q_arm, 3))
            if q_head is not None:
                print("q_head =", np.round(q_head, 3))
            print("marker_right =", np.round(T_meas[0], 3))
            print("marker_left =", np.round(T_meas[1], 3))

        elif key == "q":
            break

        time.sleep(0.05)

    q_head_arr = np.array(q_head_list) if q_head_list else None
    return np.array(q_arm_list), q_head_arr, np.array(T_meas_list)


# ============================================================
# Simulation
# ============================================================

def generate_sim_measurements(
    robot,
    dyn_model,
    q_arm_list,
    q_head_list,
    arm_idx,
    head_idx,
    q_nominal,
    ndof,
    ee_links,
    mount_to_cam_nom,
    ee_to_marker_nom,
    camera_link="link_head_2",
):
    q_offset_true = np.deg2rad([3, 0, 1, 4, -3, 2, 1, -2, 1, -1, 3, -4, 2, -2])
    q_head_offset_true = np.deg2rad([2.0, -1.5])
    xi_t5_cam_true = np.array([0.01, -0.02, 0.03, 0.04, 0.05, -0.06])

    optimize_arm = ndof in ARM_OPTIMIZATION_NDOF
    optimize_head = ndof in HEAD_OPTIMIZATION_NDOF and q_head_list is not None and head_idx is not None
    use_head_kinematics = q_head_list is not None and head_idx is not None
    optimize_camera = ndof in CAMERA_OPTIMIZATION_NDOF

    if use_head_kinematics:
        base_link = camera_link
        # In head mode, the camera extrinsic is a static transform from the
        # head mount link to the camera, so it must not depend on the current q.
        T_mount_to_cam_nom = make_transform(mount_to_cam_nom)
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    else:
        base_link = "link_torso_5"
        T_mount_to_cam_nom = make_transform(mount_to_cam_nom)
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    T_list = []

    if q_head_list is None:
        q_head_iter = [None] * len(q_arm_list)
    else:
        q_head_iter = q_head_list

    for q_arm, q_head in zip(q_arm_list, q_head_iter):
        q_full = prepare_q_full(
            q_nominal=q_nominal,
            arm_idx=arm_idx,
            q_cmd=q_arm,
            q_offset=q_offset_true if optimize_arm else None,
            head_idx=head_idx if use_head_kinematics else None,
            q_head=q_head,
            q_head_offset=q_head_offset_true if optimize_head else None,
        )

        T_pair = []
        for arm_side in ARM_SIDES:
            _, T_fk = compute_fk(robot, dyn_model, q_full, ee_links[arm_side], base_link=base_link)
            T_ee_to_marker = make_transform(ee_to_marker_nom[arm_side])
            T_meas = np.linalg.inv(T_mount_to_cam_true) @ T_fk @ T_ee_to_marker
            T_pair.append(T_meas)
        T_list.append(np.stack(T_pair, axis=0))

    return np.array(T_list)


# ============================================================
# Optimizer
# ============================================================

class CalibrationOptimizer:
    def __init__(
        self,
        robot,
        arm_idx,
        ee_links,
        mount_to_cam_nom,
        ee_to_marker_nom,
        ndof,
        head_idx=None,
        camera_link="link_head_2",
        max_iter=500,
        eps=1e-6,
        lambda_cam=DEFAULT_LAMBDA_CAM,
    ):
        self.robot = robot
        self.dyn_model = robot.get_dynamics()
        self.model = robot.model()

        self.arm_idx = np.array(arm_idx, dtype=int)
        self.head_idx = np.array(head_idx, dtype=int) if head_idx is not None else None
        self.ee_links = dict(ee_links)
        self.mount_to_cam_nom = mount_to_cam_nom
        self.ee_to_marker_nom = dict(ee_to_marker_nom)
        self.camera_link = camera_link

        self.optimize_arm = ndof in ARM_OPTIMIZATION_NDOF
        self.optimize_head = ndof in HEAD_OPTIMIZATION_NDOF and self.head_idx is not None
        self.use_head_kinematics = self.head_idx is not None
        self.optimize_camera = ndof in CAMERA_OPTIMIZATION_NDOF

        self.max_iter = max_iter
        self.eps = eps
        self.lambda_cam = lambda_cam
        self.q_nominal = robot.get_state().position.copy()
        self.numeric_jac_eps = 1e-7

        if self.use_head_kinematics:
            self.base_link = self.camera_link
            self.T_mount_to_cam_nom = make_transform(self.mount_to_cam_nom)
        else:
            self.base_link = "link_torso_5"
            self.T_mount_to_cam_nom = make_transform(self.mount_to_cam_nom)

    def joint_param_dim(self):
        dim = 0
        if self.optimize_arm:
            dim += len(self.arm_idx)
        if self.optimize_head:
            dim += len(self.head_idx)
        return dim

    def total_dim(self):
        dim = self.joint_param_dim()
        if self.optimize_camera:
            dim += 6
        return dim

    def get_nominal_mount_to_cam(self):
        return self.T_mount_to_cam_nom.copy()

    def get_nominal_ee_to_marker(self, arm_side):
        return make_transform(self.ee_to_marker_nom[str(arm_side)])

    def unpack_params(self, dx):
        cursor = 0
        q_arm_offset = np.zeros(len(self.arm_idx))
        q_head_offset = np.zeros(len(self.head_idx)) if self.head_idx is not None else None
        xi_mount_cam = np.zeros(6)

        if self.optimize_arm:
            q_arm_offset = dx[cursor:cursor + len(self.arm_idx)]
            cursor += len(self.arm_idx)
        if self.optimize_head:
            q_head_offset = dx[cursor:cursor + len(self.head_idx)]
            cursor += len(self.head_idx)
        if self.optimize_camera:
            xi_mount_cam = dx[cursor:cursor + 6]

        return q_arm_offset, q_head_offset, xi_mount_cam

    def pack_joint_jacobian(self, Jb):
        parts = []
        if self.optimize_arm:
            parts.append(Jb[:, self.arm_idx])
        if self.optimize_head:
            parts.append(Jb[:, self.head_idx])
        if not parts:
            return np.zeros((6, 0))
        return np.concatenate(parts, axis=1)

    def evaluate_sample(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam):
        q_full = prepare_q_full(
            q_nominal=self.q_nominal,
            arm_idx=self.arm_idx,
            q_cmd=q_arm,
            q_offset=q_arm_offset if self.optimize_arm else None,
            head_idx=self.head_idx,
            q_head=q_head,
            q_head_offset=q_head_offset if self.optimize_head else None,
        )

        state = self.dyn_model.make_state(
            [self.base_link, self.ee_links[str(arm_side)]],
            self.model.robot_joint_names
        )
        state.set_q(q_full)
        self.dyn_model.compute_forward_kinematics(state)
        self.dyn_model.compute_diff_forward_kinematics(state)

        T_fk = self.dyn_model.compute_transformation(state, 0, 1)
        Jb_full = self.dyn_model.compute_body_jacobian(state, 0, 1)
        Jb_joint = self.pack_joint_jacobian(Jb_full)

        T_mount_to_cam_nom = self.get_nominal_mount_to_cam()
        T_mount_to_cam = (
            T_mount_to_cam_nom @ se3_exp(xi_mount_cam)
            if self.optimize_camera else T_mount_to_cam_nom
        )
        T_ee_to_marker = self.get_nominal_ee_to_marker(arm_side)

        T_model = np.linalg.inv(T_mount_to_cam) @ T_fk @ T_ee_to_marker
        return Jb_joint, T_mount_to_cam, T_ee_to_marker, T_model

    def build_camera_jacobian_numeric(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam, T_model_ref):
        J_cam = np.zeros((6, 6))

        for i in range(6):
            delta = np.zeros(6)
            delta[i] = self.numeric_jac_eps

            _, _, _, T_model_plus = self.evaluate_sample(q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam + delta)
            _, _, _, T_model_minus = self.evaluate_sample(q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam - delta)

            xi_plus = se3_log(np.linalg.inv(T_model_ref) @ T_model_plus)
            xi_minus = se3_log(np.linalg.inv(T_model_ref) @ T_model_minus)
            J_cam[:, i] = (xi_plus - xi_minus) / (2 * self.numeric_jac_eps)

        return J_cam

    def build_jacobian(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam, Jb_joint, T_ee_to_marker, T_model):
        joint_dim = self.joint_param_dim()
        J_joint = adjoint(np.linalg.inv(T_ee_to_marker)) @ Jb_joint if joint_dim > 0 else np.zeros((6, 0))

        if joint_dim > 0 and self.optimize_camera:
            J = np.zeros((6, joint_dim + 6))
            J[:, :joint_dim] = J_joint
            J[:, joint_dim:] = self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )
            return J

        if self.optimize_camera:
            return self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )

        return J_joint

    def compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
        dim = self.total_dim()
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        total_err = 0.0

        if q_head_list is None:
            q_head_iter = [None] * len(q_arm_list)
        else:
            q_head_iter = q_head_list

        for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_iter, T_meas_list):
            for side_idx, arm_side in enumerate(ARM_SIDES):
                T_meas = T_meas_pair[side_idx]

                Jb_joint, _, T_ee_to_marker, T_model = self.evaluate_sample(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                )

                T_err = np.linalg.inv(T_model) @ T_meas
                xi = se3_log(T_err)
                J = self.build_jacobian(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                    Jb_joint,
                    T_ee_to_marker,
                    T_model,
                )

                H += J.T @ J
                g += J.T @ xi
                total_err += np.linalg.norm(xi)

        if self.optimize_camera and self.lambda_cam > 0.0:
            cam_slice = slice(dim - 6, dim)
            H[cam_slice, cam_slice] += self.lambda_cam * np.eye(6)
            g[cam_slice] += -self.lambda_cam * xi_mount_cam

        dx = np.linalg.pinv(H) @ g
        return dx, total_err

    def apply_update(self, q_arm_offset, q_head_offset, xi_mount_cam, dx):
        dq_arm, dq_head, dxi = self.unpack_params(dx)
        if self.optimize_arm:
            q_arm_offset += dq_arm
        if self.optimize_head and q_head_offset is not None:
            q_head_offset += dq_head
        if self.optimize_camera:
            xi_mount_cam += dxi
        return q_arm_offset, q_head_offset, xi_mount_cam

    def get_calibrated_t5_to_cam(self, xi_mount_cam):
        T_mount_to_cam = self.get_nominal_mount_to_cam() @ se3_exp(xi_mount_cam)
        if self.use_head_kinematics:
            _, T_t5_to_mount = compute_fk(
                robot=self.robot,
                dyn_model=self.dyn_model,
                q_full=self.q_nominal,
                ee_link=self.camera_link,
                base_link="link_torso_5",
            )
            T_calib = T_t5_to_mount @ T_mount_to_cam
        else:
            T_calib = T_mount_to_cam

        p = T_calib[:3, 3]
        rpy = rot_to_euler_zyx(T_calib[:3, :3])

        return [
            p[0], p[1], p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]
        
    def get_calibrated_mount_to_cam(self, xi_mount_cam):
        T_mount_to_cam = self.get_nominal_mount_to_cam() @ se3_exp(xi_mount_cam)

        p = T_mount_to_cam[:3, 3]
        rpy = rot_to_euler_zyx(T_mount_to_cam[:3, :3])

        return [
            p[0], p[1], p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]

    def optimize(self, q_arm_list, q_head_list, T_meas_list):
        if self.use_head_kinematics and q_head_list is None:
            raise RuntimeError(
                "Head kinematics are enabled for this ndof, but q_head_list is missing."
            )

        q_arm_offset = np.zeros(len(self.arm_idx))
        q_head_offset = np.zeros(len(self.head_idx)) if self.optimize_head else None
        xi_mount_cam = np.zeros(6)

        for it in range(self.max_iter):
            dx, total_err = self.compute_step(
                q_arm_list,
                q_head_list,
                T_meas_list,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
            )
            q_arm_offset, q_head_offset, xi_mount_cam = self.apply_update(
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                dx,
            )

            print(f"[{it}] |dx|={np.linalg.norm(dx):.3e}, |err|={total_err:.3e}")

            if np.linalg.norm(dx) < self.eps:
                print("Converged.")
                break

        # t5_to_cam_new = self.get_calibrated_t5_to_cam(xi_mount_cam)
        # return q_arm_offset, q_head_offset, xi_mount_cam, t5_to_cam_new
        mount_to_cam_new = self.get_calibrated_mount_to_cam(xi_mount_cam)
        t5_to_cam_new = self.get_calibrated_t5_to_cam(xi_mount_cam)
        return q_arm_offset, q_head_offset, xi_mount_cam, mount_to_cam_new, t5_to_cam_new


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(args, robot, dyn_model, config):
    marker_transform = None
    head_config = get_head_config(robot.model())

    if args.mode == "live":
        marker_transform = create_live_marker_transform()
        q_arm_list, q_head_list, T_meas_list = capture_dataset(
            robot=robot,
            arm_idx=config["arm_idx"],
            marker_transform=marker_transform,
            head_idx=head_config["head_idx"],
        )
        save_npz_dataset(args.path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)

    elif args.mode == "npz":
        q_arm_list, q_head_list, T_meas_list = load_npz_dataset(args.path)
        validate_dataset_for_ndof(args.ndof, q_arm_list, q_head_list, T_meas_list)
        print("size =", np.size(q_arm_list))

    else:  # sim
        sample_count = 100
        q_arm_list = np.random.uniform(-5, 5, (sample_count, 14))
        if args.ndof in HEAD_OPTIMIZATION_NDOF:
            q_head_list = np.random.uniform(-0.2, 0.2, (sample_count, 2))
        else:
            q_head_ref = robot.get_state().position[head_config["head_idx"]].copy()
            q_head_list = np.tile(q_head_ref, (len(q_arm_list), 1))
        q_nominal = robot.get_state().position.copy()
        T_meas_list = generate_sim_measurements(
            robot=robot,
            dyn_model=dyn_model,
            q_arm_list=q_arm_list,
            q_head_list=q_head_list,
            arm_idx=config["arm_idx"],
            head_idx=head_config["head_idx"],
            q_nominal=q_nominal,
            ndof=args.ndof,
            ee_links=config["ee_links"],
            mount_to_cam_nom=config["mount_to_cam_nom"],
            ee_to_marker_nom=config["ee_to_marker_nom"],
            camera_link=head_config["camera_link"],
        )

    return q_arm_list, q_head_list, T_meas_list, marker_transform


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndof", type=int, default=14, choices=[2, 6, 14, 16, 20, 22])
    parser.add_argument("--ip", type=str, default="192.168.30.1:50051")
    parser.add_argument("--model", type=str, default="a", choices=["a", "m"])
    parser.add_argument("--mode", type=str, required=True, choices=["live", "npz", "sim"])
    parser.add_argument(
        "--path",
        type=str,
        default="captured_dataset.npz",
        help="Path to npz dataset (default: captured_dataset.npz)"
    )
    parser.add_argument("--lambda-cam", type=float, default=100.0)
    args = parser.parse_args()

    marker_transform = None

    robot = create_robot(args.ip, args.model)
    dyn_model = robot.get_dynamics()
    config = get_both_arm_config(robot.model())

    q_arm_list, q_head_list, T_meas_list, marker_transform = prepare_dataset(
        args=args,
        robot=robot,
        dyn_model=dyn_model,
        config=config,
    )

    print("Dataset saved.")

    optimizer = CalibrationOptimizer(
        robot=robot,
        arm_idx=config["arm_idx"],
        ee_links=config["ee_links"],
        mount_to_cam_nom=config["mount_to_cam_nom"],
        ee_to_marker_nom=config["ee_to_marker_nom"],
        ndof=args.ndof,
        head_idx=get_head_config(robot.model())["head_idx"],
        lambda_cam=args.lambda_cam,
    )

    q_arm_offset, q_head_offset, xi_t5_cam, mount_to_cam_new, t5_to_cam_new = optimizer.optimize(
        q_arm_list,
        q_head_list,
        T_meas_list,
    )
    right_arm_offset, left_arm_offset = split_arm_offsets(q_arm_offset)

    print("\n===== RESULT =====")
    print("Right arm joint offset (deg):")
    print(np.rad2deg(right_arm_offset))
    if left_arm_offset is not None:
        print("Left arm joint offset (deg):")
        print(np.rad2deg(left_arm_offset))
    if q_head_offset is not None:
        print("Head joint offset (deg):")
        print(np.rad2deg(q_head_offset))
    print("T5-to-camera xi:")
    print(xi_t5_cam)
    print("Calibrated mount_to_cam:")
    print(mount_to_cam_new)
    print("Calibrated t5_to_cam:")
    print(t5_to_cam_new)

    save_result("calibration_result.json", q_arm_offset, xi_t5_cam, q_head_offset=q_head_offset)
    print("Result saved to calibration_result.json")

    if marker_transform is not None:
        marker_transform.camera.monitoring(Flag=False)


if __name__ == "__main__":
    main()
