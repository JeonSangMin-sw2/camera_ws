import argparse
import json
import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rby1_sdk as rby

from marker_detection import Marker_Transform


np.set_printoptions(suppress=True, precision=6)


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
    return data["q"], data["marker"]


def save_result(path, q_offset, xi_cam):
    result_dict = {
        "joint_offset_deg": np.rad2deg(q_offset).tolist(),
        "xi_cam": np.asarray(xi_cam).tolist(),
    }
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)


def create_robot(ip):
    robot = rby.create_robot_m(ip)
    robot.connect()
    robot.power_on(".*")
    robot.servo_on("^(?!.*head).*")
    robot.reset_fault_control_manager()
    robot.enable_control_manager(False)
    return robot


def get_arm_config(model, arm):
    if arm == "right":
        return {
            "arm_idx": model.right_arm_idx[:7],
            "ee_link": "ee_right",
            "tool_to_cam_nom": [
                0.01079, -0.094527, -0.028914,
                154.992754, -0.269972, -179.718444
            ],
        }

    return {
        "arm_idx": model.left_arm_idx[:7],
        "ee_link": "ee_left",
        "tool_to_cam_nom": [
            -0.009187, 0.094257, -0.028313,
            154.667827, -0.320824, -0.268186
        ],
    }


def compute_fk(robot, dyn_model, q_full, ee_link):
    state = dyn_model.make_state(
        ["link_torso_5", ee_link],
        robot.model().robot_joint_names
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    return state, dyn_model.compute_transformation(state, 0, 1)


def prepare_q_full(q_nominal, arm_idx, q_cmd, q_offset=None):
    q_full = q_nominal.copy()
    q_full[arm_idx] = q_cmd if q_offset is None else (q_cmd + q_offset)
    return q_full


# ============================================================
# Capture dataset
# ============================================================

def capture_dataset(robot, arm_idx, marker_transform):
    q_cmd_list = []
    T_meas_list = []

    print("\nPress 'e' + Enter to capture")
    print("Press 'q' + Enter to quit\n")

    while True:
        key = input().strip()

        if key == "e":
            state = robot.get_state()
            q_full = state.position.copy()
            q_cmd = q_full[arm_idx].copy()

            result = marker_transform.get_marker_transform(sampling_time=2)
            if result is None:
                print("Marker not detected.")
                continue

            T_meas = np.array(result).reshape(4, 4)

            q_cmd_list.append(q_cmd)
            T_meas_list.append(T_meas)

            print(f"Captured sample {len(q_cmd_list)}")
            print("q =", np.round(q_cmd, 3))
            print("marker =", np.round(T_meas, 3))

        elif key == "q":
            break

        time.sleep(0.05)

    return np.array(q_cmd_list), np.array(T_meas_list)


# ============================================================
# Simulation
# ============================================================

def generate_sim_measurements(
    robot,
    dyn_model,
    q_cmd_list,
    arm_idx,
    q_nominal,
    ndof,
    ee_link,
    tool_to_cam_nom
):
    q_offset_true = np.deg2rad([3, -2, 1, 4, -3, 2, 1])
    xi_cam_true = np.array([0.01, -0.02, 0.03, 0.04, 0.05, -0.06])

    optimize_joint = ndof in (7, 13)
    optimize_camera = ndof in (6, 13)

    T_tool_to_cam_nom = make_transform(tool_to_cam_nom)
    T_tool_to_cam_true = T_tool_to_cam_nom @ se3_exp(xi_cam_true)

    T_list = []

    for q_cmd in q_cmd_list:
        q_full = prepare_q_full(
            q_nominal=q_nominal,
            arm_idx=arm_idx,
            q_cmd=q_cmd,
            q_offset=q_offset_true if optimize_joint else None,
        )

        _, T_fk = compute_fk(robot, dyn_model, q_full, ee_link)
        T_meas = T_fk @ (T_tool_to_cam_true if optimize_camera else T_tool_to_cam_nom)
        T_list.append(T_meas)

    return np.array(T_list)


# ============================================================
# Optimizer
# ============================================================

class CalibrationOptimizer:
    def __init__(self, robot, arm_idx, ee_link, tool_to_cam_nom, ndof, max_iter=500, eps=1e-6):
        self.robot = robot
        self.dyn_model = robot.get_dynamics()
        self.model = robot.model()

        self.arm_idx = arm_idx
        self.ee_link = ee_link
        self.tool_to_cam_nom = tool_to_cam_nom

        self.optimize_joint = ndof in (7, 13)
        self.optimize_camera = ndof in (6, 13)

        self.max_iter = max_iter
        self.eps = eps
        self.q_nominal = robot.get_state().position.copy()

    def get_nominal_tool_to_cam(self):
        return make_transform(self.tool_to_cam_nom)

    def evaluate_sample(self, q_cmd, q_offset, xi_cam):
        q_full = prepare_q_full(
            q_nominal=self.q_nominal,
            arm_idx=self.arm_idx,
            q_cmd=q_cmd,
            q_offset=q_offset if self.optimize_joint else None,
        )

        state = self.dyn_model.make_state(
            ["link_torso_5", self.ee_link],
            self.model.robot_joint_names
        )
        state.set_q(q_full)
        self.dyn_model.compute_forward_kinematics(state)
        self.dyn_model.compute_diff_forward_kinematics(state)

        T_fk = self.dyn_model.compute_transformation(state, 0, 1)
        Jb = self.dyn_model.compute_body_jacobian(state, 0, 1)[:, self.arm_idx]

        T_tool_to_cam_nom = self.get_nominal_tool_to_cam()
        T_tool_to_cam = (
            T_tool_to_cam_nom @ se3_exp(xi_cam)
            if self.optimize_camera else T_tool_to_cam_nom
        )

        T_model = T_fk @ T_tool_to_cam
        return Jb, T_tool_to_cam, T_model

    def build_jacobian(self, Jb, T_tool_to_cam):
        if self.optimize_joint and self.optimize_camera:
            J = np.zeros((6, 13))
            J[:, :7] = adjoint(np.linalg.inv(T_tool_to_cam)) @ Jb
            J[:, 7:] = np.eye(6)
            return J

        if self.optimize_camera:
            return np.eye(6)

        return adjoint(np.linalg.inv(T_tool_to_cam)) @ Jb

    def compute_step(self, q_cmd_list, T_meas_list, q_offset, xi_cam):
        dim = 13 if (self.optimize_joint and self.optimize_camera) else (6 if self.optimize_camera else 7)
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        total_err = 0.0

        for q_cmd, T_meas in zip(q_cmd_list, T_meas_list):
            Jb, T_tool_to_cam, T_model = self.evaluate_sample(q_cmd, q_offset, xi_cam)

            T_err = np.linalg.inv(T_model) @ T_meas
            xi = se3_log(T_err)
            J = self.build_jacobian(Jb, T_tool_to_cam)

            H += J.T @ J
            g += J.T @ xi
            total_err += np.linalg.norm(xi)

        dx = np.linalg.pinv(H) @ g
        return dx, total_err

    def apply_update(self, q_offset, xi_cam, dx):
        if self.optimize_joint and self.optimize_camera:
            q_offset += dx[:7]
            xi_cam += dx[7:]
        elif self.optimize_camera:
            xi_cam += dx
        else:
            q_offset += dx
        return q_offset, xi_cam

    def get_calibrated_tool_to_cam(self, xi_cam):
        T_nom = self.get_nominal_tool_to_cam()
        T_calib = T_nom @ se3_exp(xi_cam)

        p = T_calib[:3, 3]
        rpy = rot_to_euler_zyx(T_calib[:3, :3])

        return [
            p[0], p[1], p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]

    def optimize(self, q_cmd_list, T_meas_list):
        q_offset = np.zeros(7)
        xi_cam = np.zeros(6)

        for it in range(self.max_iter):
            dx, total_err = self.compute_step(q_cmd_list, T_meas_list, q_offset, xi_cam)
            q_offset, xi_cam = self.apply_update(q_offset, xi_cam, dx)

            print(f"[{it}] |dx|={np.linalg.norm(dx):.3e}, |err|={total_err:.3e}")

            if np.linalg.norm(dx) < self.eps:
                print("Converged.")
                break

        tool_to_cam_new = self.get_calibrated_tool_to_cam(xi_cam)
        return q_offset, xi_cam, tool_to_cam_new


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(args, robot, dyn_model, config):
    marker_transform = None

    if args.mode == "live":
        marker_transform = Marker_Transform(
            Stereo=True,
            serial_number=None,
            monitoring=False
        )
        q_cmd_list, T_meas_list = capture_dataset(
            robot=robot,
            arm_idx=config["arm_idx"],
            marker_transform=marker_transform,
        )
        np.savez_compressed(args.path, q=q_cmd_list, marker=T_meas_list)

    elif args.mode == "npz":
        q_cmd_list, T_meas_list = load_npz_dataset(args.path)
        print("size =", np.size(q_cmd_list))

    else:  # sim
        q_cmd_list = np.random.uniform(-5, 5, (100, 7))
        q_nominal = robot.get_state().position.copy()
        T_meas_list = generate_sim_measurements(
            robot=robot,
            dyn_model=dyn_model,
            q_cmd_list=q_cmd_list,
            arm_idx=config["arm_idx"],
            q_nominal=q_nominal,
            ndof=args.ndof,
            ee_link=config["ee_link"],
            tool_to_cam_nom=config["tool_to_cam_nom"],
        )

    return q_cmd_list, T_meas_list, marker_transform


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndof", type=int, default=7, choices=[6, 7, 13])
    parser.add_argument("--ip", type=str, default="192.168.30.1:50051")
    parser.add_argument("--mode", type=str, required=True, choices=["live", "npz", "sim"])
    parser.add_argument(
        "--path",
        type=str,
        default="captured_dataset.npz",
        help="Path to npz dataset (default: captured_dataset.npz)"
    )
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
    args = parser.parse_args()

    marker_transform = None

    robot = create_robot(args.ip)
    dyn_model = robot.get_dynamics()
    config = get_arm_config(robot.model(), args.arm)

    q_cmd_list, T_meas_list, marker_transform = prepare_dataset(
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
        tool_to_cam_nom=config["tool_to_cam_nom"],
        ndof=args.ndof,
    )

    q_offset, xi_cam, tool_to_cam_new = optimizer.optimize(q_cmd_list, T_meas_list)

    print("\n===== RESULT =====")
    print("Joint offset (deg):")
    print(np.rad2deg(q_offset))
    print("Camera xi:")
    print(xi_cam)
    print("Calibrated tool_to_cam:")
    print(tool_to_cam_new)

    save_result("calibration_result.json", q_offset, xi_cam)
    print("Result saved to calibration_result.json")

    if marker_transform is not None:
        marker_transform.camera.monitoring(Flag=False)


if __name__ == "__main__":
    main()