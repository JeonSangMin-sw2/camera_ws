import argparse
import json
import math
import time
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
    result_dict = {
        "joint_offset_deg": np.rad2deg(q_offset).tolist(),
        "head_joint_offset_deg": (
            np.rad2deg(q_head_offset).tolist()
            if q_head_offset is not None else None
        ),
        "xi_t5_cam": np.asarray(xi_t5_cam).tolist(),
        "xi_cam": np.asarray(xi_t5_cam).tolist(),
    }
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)


def create_robot(ip):
    robot = rby.create_robot_a(ip)
    robot.connect()
    robot.power_on(".*")
    robot.servo_on(".*")
    robot.reset_fault_control_manager()
    robot.enable_control_manager(False)
    return robot


def load_camera_nominals():
    with open(SETTING_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    camera_cfg = config.get("camera", {})
    return {
        "t5_to_cam_nom": camera_cfg["T5_to_cam"],
        "ee_to_marker_left": camera_cfg["Tf_to_marker_left"],
        "ee_to_marker_right": camera_cfg["Tf_to_marker_right"],
    }


def get_arm_config(model, arm):
    camera_nominals = load_camera_nominals()

    if arm == "right":
        return {
            "arm_idx": model.right_arm_idx[:7],
            "ee_link": "ee_right",
            "t5_to_cam_nom": camera_nominals["t5_to_cam_nom"],
            "ee_to_marker_nom": camera_nominals["ee_to_marker_right"],
        }

    return {
        "arm_idx": model.left_arm_idx[:7],
        "ee_link": "ee_left",
        "t5_to_cam_nom": camera_nominals["t5_to_cam_nom"],
        "ee_to_marker_nom": camera_nominals["ee_to_marker_left"],
    }


def get_head_config(model):
    return {
        "head_idx": model.head_idx[:2],
        "camera_link": "link_head_2",
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


def infer_camera_mount_to_cam(robot, q_ref, t5_to_cam_nom, camera_link="link_head_2"):
    dyn_model = robot.get_dynamics()
    state, T_t5_to_camera_mount = compute_fk(
        robot=robot,
        dyn_model=dyn_model,
        q_full=q_ref,
        ee_link=camera_link,
        base_link="link_torso_5",
    )
    _ = state
    T_t5_to_cam = make_transform(t5_to_cam_nom)
    return np.linalg.inv(T_t5_to_camera_mount) @ T_t5_to_cam


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


def capture_one_sample(robot, arm_idx, marker_transform, sampling_time=2, side="left", head_idx=None):
    state = robot.get_state()
    q_full = state.position.copy()
    q_arm = q_full[arm_idx].copy()
    q_head = q_full[head_idx].copy() if head_idx is not None else None

    result = marker_transform.get_marker_transform(sampling_time=sampling_time, side=side)
    if result is None:
        return None, None, None

    T_meas = np.array(result).reshape(4, 4)
    return q_arm, q_head, T_meas


def capture_dataset(robot, arm_idx, marker_transform, head_idx=None):
    q_arm_list = []
    q_head_list = []
    T_meas_list = []

    print("\nPress 'e' + Enter to capture")
    print("Press 'q' + Enter to quit\n")

    while True:
        key = input().strip()

        if key == "e":
            q_arm, q_head, T_meas = capture_one_sample(
                robot=robot,
                arm_idx=arm_idx,
                marker_transform=marker_transform,
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
            print("marker =", np.round(T_meas, 3))

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
    ee_link,
    t5_to_cam_nom,
    ee_to_marker_nom,
    camera_link="link_head_2",
):
    q_offset_true = np.deg2rad([3, 0, 1, 4, -3, 2, 1])
    q_head_offset_true = np.deg2rad([2.0, -1.5])
    xi_t5_cam_true = np.array([0.01, -0.02, 0.03, 0.04, 0.05, -0.06])

    optimize_arm = ndof in (7, 13, 15)
    optimize_head = ndof in (2, 15) and q_head_list is not None and head_idx is not None
    optimize_camera = ndof in (6, 13, 15)

    if optimize_head:
        base_link = camera_link
        T_mount_to_cam_nom = infer_camera_mount_to_cam(
            robot=robot,
            q_ref=q_nominal,
            t5_to_cam_nom=t5_to_cam_nom,
            camera_link=camera_link,
        )
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    else:
        base_link = "link_torso_5"
        T_mount_to_cam_nom = make_transform(t5_to_cam_nom)
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    T_ee_to_marker = make_transform(ee_to_marker_nom)

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
            head_idx=head_idx if optimize_head else None,
            q_head=q_head,
            q_head_offset=q_head_offset_true if optimize_head else None,
        )

        _, T_fk = compute_fk(robot, dyn_model, q_full, ee_link, base_link=base_link)
        T_meas = np.linalg.inv(T_mount_to_cam_true) @ T_fk @ T_ee_to_marker
        T_list.append(T_meas)

    return np.array(T_list)


# ============================================================
# Optimizer
# ============================================================

class CalibrationOptimizer:
    def __init__(
        self,
        robot,
        arm_idx,
        ee_link,
        t5_to_cam_nom,
        ee_to_marker_nom,
        ndof,
        head_idx=None,
        camera_link="link_head_2",
        max_iter=500,
        eps=1e-6,
        lambda_cam=10.0,
    ):
        self.robot = robot
        self.dyn_model = robot.get_dynamics()
        self.model = robot.model()

        self.arm_idx = np.array(arm_idx, dtype=int)
        self.head_idx = np.array(head_idx, dtype=int) if head_idx is not None else None
        self.ee_link = ee_link
        self.t5_to_cam_nom = t5_to_cam_nom
        self.ee_to_marker_nom = ee_to_marker_nom
        self.camera_link = camera_link

        self.optimize_arm = ndof in (7, 13, 15)
        self.optimize_head = ndof in (2, 15) and self.head_idx is not None
        self.optimize_camera = ndof in (6, 13, 15)

        self.max_iter = max_iter
        self.eps = eps
        self.lambda_cam = lambda_cam
        self.q_nominal = robot.get_state().position.copy()
        self.numeric_jac_eps = 1e-7

        if self.optimize_head:
            self.base_link = self.camera_link
            self.T_mount_to_cam_nom = infer_camera_mount_to_cam(
                robot=robot,
                q_ref=self.q_nominal,
                t5_to_cam_nom=self.t5_to_cam_nom,
                camera_link=self.camera_link,
            )
        else:
            self.base_link = "link_torso_5"
            self.T_mount_to_cam_nom = make_transform(self.t5_to_cam_nom)

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

    def get_nominal_ee_to_marker(self):
        return make_transform(self.ee_to_marker_nom)

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

    def evaluate_sample(self, q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam):
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
            [self.base_link, self.ee_link],
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
        T_ee_to_marker = self.get_nominal_ee_to_marker()

        T_model = np.linalg.inv(T_mount_to_cam) @ T_fk @ T_ee_to_marker
        return Jb_joint, T_mount_to_cam, T_ee_to_marker, T_model

    def build_camera_jacobian_numeric(self, q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam, T_model_ref):
        J_cam = np.zeros((6, 6))

        for i in range(6):
            delta = np.zeros(6)
            delta[i] = self.numeric_jac_eps

            _, _, _, T_model_plus = self.evaluate_sample(q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam + delta)
            _, _, _, T_model_minus = self.evaluate_sample(q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam - delta)

            xi_plus = se3_log(np.linalg.inv(T_model_ref) @ T_model_plus)
            xi_minus = se3_log(np.linalg.inv(T_model_ref) @ T_model_minus)
            J_cam[:, i] = (xi_plus - xi_minus) / (2 * self.numeric_jac_eps)

        return J_cam

    def build_jacobian(self, q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam, Jb_joint, T_ee_to_marker, T_model):
        joint_dim = self.joint_param_dim()
        J_joint = adjoint(np.linalg.inv(T_ee_to_marker)) @ Jb_joint if joint_dim > 0 else np.zeros((6, 0))

        if joint_dim > 0 and self.optimize_camera:
            J = np.zeros((6, joint_dim + 6))
            J[:, :joint_dim] = J_joint
            J[:, joint_dim:] = self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
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

        for q_arm, q_head, T_meas in zip(q_arm_list, q_head_iter, T_meas_list):
            Jb_joint, _, T_ee_to_marker, T_model = self.evaluate_sample(
                q_arm,
                q_head,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
            )

            T_err = np.linalg.inv(T_model) @ T_meas
            xi = se3_log(T_err)
            J = self.build_jacobian(q_arm, q_head, q_arm_offset, q_head_offset, xi_mount_cam, Jb_joint, T_ee_to_marker, T_model)

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
        if self.optimize_head:
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

    def optimize(self, q_arm_list, q_head_list, T_meas_list):
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

        t5_to_cam_new = self.get_calibrated_t5_to_cam(xi_mount_cam)
        return q_arm_offset, q_head_offset, xi_mount_cam, t5_to_cam_new


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
            head_idx=head_config["head_idx"] if args.ndof in (2, 15) else None,
        )
        save_npz_dataset(args.path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)

    elif args.mode == "npz":
        q_arm_list, q_head_list, T_meas_list = load_npz_dataset(args.path)
        print("size =", np.size(q_arm_list))

    else:  # sim
        q_arm_list = np.random.uniform(-5, 5, (100, 7))
        q_head_list = np.random.uniform(-0.2, 0.2, (100, 2)) if args.ndof in (2, 15) else None
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
            ee_link=config["ee_link"],
            t5_to_cam_nom=config["t5_to_cam_nom"],
            ee_to_marker_nom=config["ee_to_marker_nom"],
            camera_link=head_config["camera_link"],
        )

    return q_arm_list, q_head_list, T_meas_list, marker_transform


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndof", type=int, default=7, choices=[2, 6, 7, 13, 15])
    parser.add_argument("--ip", type=str, default="192.168.30.1:50051")
    parser.add_argument("--mode", type=str, required=True, choices=["live", "npz", "sim"])
    parser.add_argument(
        "--path",
        type=str,
        default="captured_dataset.npz",
        help="Path to npz dataset (default: captured_dataset.npz)"
    )
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--lambda-cam", type=float, default=100.0)
    args = parser.parse_args()

    marker_transform = None

    robot = create_robot(args.ip)
    dyn_model = robot.get_dynamics()
    config = get_arm_config(robot.model(), args.arm)

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
        ee_link=config["ee_link"],
        t5_to_cam_nom=config["t5_to_cam_nom"],
        ee_to_marker_nom=config["ee_to_marker_nom"],
        ndof=args.ndof,
        head_idx=get_head_config(robot.model())["head_idx"] if args.ndof in (2, 15) else None,
        lambda_cam=args.lambda_cam,
    )

    q_arm_offset, q_head_offset, xi_t5_cam, t5_to_cam_new = optimizer.optimize(q_arm_list, q_head_list, T_meas_list)

    print("\n===== RESULT =====")
    print("Arm joint offset (deg):")
    print(np.rad2deg(q_arm_offset))
    if q_head_offset is not None:
        print("Head joint offset (deg):")
        print(np.rad2deg(q_head_offset))
    print("T5-to-camera xi:")
    print(xi_t5_cam)
    print("Calibrated t5_to_cam:")
    print(t5_to_cam_new)

    save_result("calibration_result.json", q_arm_offset, xi_t5_cam, q_head_offset=q_head_offset)
    print("Result saved to calibration_result.json")

    if marker_transform is not None:
        marker_transform.camera.monitoring(Flag=False)


if __name__ == "__main__":
    main()
