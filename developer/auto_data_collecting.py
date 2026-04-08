import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import rby1_sdk as rby

D2R = np.pi / 180.0


def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)


def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def make_T(R, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def init_robot(address, model, power, servo):
    robot = rby.create_robot(address, model)

    if not robot.connect():
        print("connect fail")
        sys.exit(1)

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            print("power on fail")
            sys.exit(1)

    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            print("servo on fail")
            sys.exit(1)

    cm = robot.get_control_manager_state()
    if cm.state in [
        rby.ControlManagerState.State.MinorFault,
        rby.ControlManagerState.State.MajorFault,
    ]:
        if not robot.reset_fault_control_manager():
            print("fault reset fail")
            sys.exit(1)

    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("control manager enable fail")
        sys.exit(1)

    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("manipulability_threshold", "1e6")
    return robot


def go_ready(robot):
    q_torso = np.array([0, 60, -120, 60, 0, 0]) * D2R
    q_right =np.array(np.deg2rad([-43.975, -21.385, -20.251, -104.030, 83.705, -62.694, 33.967]))
    q_left = np.array(np.deg2rad([-63.975, 21.385, 20.251, -104.030, -83.705, -62.694, -33.967]))
    # q_right =np.array(np.deg2rad([-53.975, -21.385, 0.251, -104.030, -83.705, 62.694, 33.967]))
    # q_left = np.array(np.deg2rad([-53.975, 21.385, -0.251, -104.030, 83.705, 62.694, -33.967]))
    
    q = np.concatenate([q_torso, q_right, q_left])

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(3.0)
            )
        )
    )

    rv = robot.send_command(cmd, 10).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        print("ready pose fail")
        sys.exit(1)


def make_cartesian_cmd(T_right=None, T_left=None, head_position=None, min_time=0.8, hold_time=5.0):
    body = rby.BodyComponentBasedCommandBuilder()

    if T_right is not None:
        body.set_right_arm_command(
            rby.CartesianCommandBuilder()
            .add_target(
                "link_torso_5",
                "ee_right",
                T_right,
                0.2,
                0.5,
                0.3,
            )
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(min_time)
        )

    if T_left is not None:
        body.set_left_arm_command(
            rby.CartesianCommandBuilder()
            .add_target(
                "link_torso_5",
                "ee_left",
                T_left,
                0.2,
                0.5,
                0.3,
            )
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(min_time)
        )

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(body)
    if head_position is not None:
        cmd.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(min_time)
            .set_position(head_position)
        )

    return rby.RobotCommandBuilder().set_command(cmd)


def compute_target_T(pivot, tip_offset, roll_deg, pitch_deg, yaw_deg):
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # Bottom-up: fixed base orientation, only apply requested R/P/Y.
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    p_ee = pivot - R @ tip_offset
    return make_T(R, p_ee)


def linspace_symmetric(max_abs_deg, steps):
    if steps <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-max_abs_deg, max_abs_deg, steps, dtype=np.float64)


def build_rpy_samples(mode, roll_vals, pitch_vals, yaw_vals):
    if mode == "grid":
        return [(float(r), float(p), float(y)) for r, p, y in itertools.product(roll_vals, pitch_vals, yaw_vals)]

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


def compute_head_target(base_head_q, roll_delta_deg, pitch_delta_deg, yaw_delta_deg, head_max_deg):
    # Head has 2-DoF(yaw/pitch), so roll is merged into yaw tracking.
    yaw_cmd_deg = float(np.clip(roll_delta_deg + yaw_delta_deg, -head_max_deg, head_max_deg))
    pitch_cmd_deg = float(np.clip(pitch_delta_deg, -head_max_deg, head_max_deg))
    return base_head_q + np.deg2rad(np.array([yaw_cmd_deg, pitch_cmd_deg], dtype=np.float64))


def get_output_path(out_arg):
    if out_arg:
        return Path(out_arg).expanduser().resolve()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (Path(__file__).resolve().parents[1] / "result" / f"auto_q_{stamp}.npz").resolve()


def save_dataset(
    out_path,
    mode,
    sampling_mode,
    pivot_right,
    pivot_left,
    tip_offset,
    q_arm_samples,
    q_right_samples,
    q_left_samples,
    q_head_samples,
    q_full_samples,
    rpy_saved_samples,
    T_right_samples,
    T_left_samples,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q_arm_arr = np.asarray(q_arm_samples, dtype=np.float64)
    q_right_arr = np.asarray(q_right_samples, dtype=np.float64)
    q_left_arr = np.asarray(q_left_samples, dtype=np.float64)
    q_head_arr = np.asarray(q_head_samples, dtype=np.float64)
    q_full_arr = np.asarray(q_full_samples, dtype=np.float64)
    rpy_arr = np.asarray(rpy_saved_samples, dtype=np.float64)
    T_right_arr = np.asarray(T_right_samples, dtype=np.float64)
    T_left_arr = np.asarray(T_left_samples, dtype=np.float64)

    np.savez_compressed(
        out_path,
        q=q_arm_arr,
        q_arm=q_arm_arr,
        q_right=q_right_arr,
        q_left=q_left_arr,
        q_head=q_head_arr,
        q_full=q_full_arr,
        rpy_deg=rpy_arr,
        target_T_right=T_right_arr,
        target_T_left=T_left_arr,
        mode=np.array(mode),
        sampling=np.array(sampling_mode),
        pivot_right=np.asarray(pivot_right, dtype=np.float64),
        pivot_left=np.asarray(pivot_left, dtype=np.float64),
        tip_offset=np.asarray(tip_offset, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="a")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="torso_.*|right_arm_.*|left_arm_.*|head_.*")

    parser.add_argument("--mode", type=str, default="both", choices=["right", "left", "both"])
    parser.add_argument("--sampling_mode", type=str, default="axis", choices=["axis", "grid"])

    parser.add_argument("--pivot_x", type=float, default=0.35)
    parser.add_argument("--pivot_y", type=float, default=0.0)
    parser.add_argument("--arm_gap_y", type=float, default=0.16, help="left-right y spacing in meters")
    parser.add_argument("--arm_gap_z", type=float, default=0.2, help="left-right z spacing in meters")
    parser.add_argument("--head_height_z", type=float, default=0.2)
    parser.add_argument("--pivot_z", type=float, default=None, help="legacy alias for head_height_z")

    parser.add_argument("--tip_offset_x", type=float, default=0.0)
    parser.add_argument("--tip_offset_y", type=float, default=0.0)
    parser.add_argument("--tip_offset_z", type=float, default=-0.075)

    parser.add_argument("--max_roll_deg", type=float, default=10.0)
    parser.add_argument("--max_pitch_deg", type=float, default=10.0)
    parser.add_argument("--max_yaw_deg", type=float, default=10.0)
    parser.add_argument("--init_roll_deg", type=float, default=0.0)
    parser.add_argument("--init_pitch_deg", type=float, default=90.0)
    parser.add_argument("--init_yaw_deg", type=float, default=-90.0)
    parser.add_argument("--roll_steps", type=int, default=5)
    parser.add_argument("--pitch_steps", type=int, default=5)
    parser.add_argument("--yaw_steps", type=int, default=5)

    parser.add_argument("--move_time", type=float, default=1.2)
    parser.add_argument("--settle_time", type=float, default=0.6)
    parser.add_argument("--hold_time", type=float, default=0.0)
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--head_max_deg", type=float, default=3.0)

    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    robot = init_robot(args.address, args.model, args.power, args.servo)
    go_ready(robot)
    time.sleep(1.0)

    model = robot.model()
    right_idx = np.array(model.right_arm_idx[:7], dtype=int)
    left_idx = np.array(model.left_arm_idx[:7], dtype=int)
    head_idx = np.array(model.head_idx[:2], dtype=int)
    base_head_q = robot.get_state().position[head_idx].copy()

    pivot_z = args.head_height_z if args.pivot_z is None else args.pivot_z
    half_gap = max(float(args.arm_gap_y), 0.0) * 0.5
    half_gap_z = max(float(args.arm_gap_z), 0.0) * 0.5
    center_y = args.pivot_y
    # right is slightly lower, left is slightly higher when arm_gap_z > 0.
    pivot_right = np.array([args.pivot_x, center_y - half_gap, pivot_z - half_gap_z], dtype=np.float64)
    pivot_left = np.array([args.pivot_x, center_y + half_gap, pivot_z + half_gap_z], dtype=np.float64)
    tip_offset = np.array([args.tip_offset_x, args.tip_offset_y, args.tip_offset_z], dtype=np.float64)

    roll_vals = linspace_symmetric(args.max_roll_deg, args.roll_steps)
    pitch_vals = linspace_symmetric(args.max_pitch_deg, args.pitch_steps)
    yaw_vals = linspace_symmetric(args.max_yaw_deg, args.yaw_steps)
    rpy_samples = build_rpy_samples(args.sampling_mode, roll_vals, pitch_vals, yaw_vals)

    print(f"mode         : {args.mode}")
    print(f"sampling_mode: {args.sampling_mode}")
    print(f"arm_gap_y    : {args.arm_gap_y}")
    print(f"arm_gap_z    : {args.arm_gap_z}")
    print(f"pivot_right  : {pivot_right}")
    print(f"pivot_left   : {pivot_left}")
    print(f"init_rpy_deg : ({args.init_roll_deg}, {args.init_pitch_deg}, {args.init_yaw_deg})")
    print(f"tip_offset   : {tip_offset}")
    print(f"num_samples  : {len(rpy_samples)}")
    print(f"head_max_deg : {args.head_max_deg}")

    q_arm_samples = []
    q_right_samples = []
    q_left_samples = []
    q_head_samples = []
    q_full_samples = []
    rpy_saved_samples = []
    T_right_samples = []
    T_left_samples = []

    for i, (roll_delta_deg, pitch_delta_deg, yaw_delta_deg) in enumerate(rpy_samples, start=1):
        roll_deg = args.init_roll_deg + roll_delta_deg
        pitch_deg = args.init_pitch_deg + pitch_delta_deg
        yaw_deg = args.init_yaw_deg + yaw_delta_deg

        T_right = None
        T_left = None

        if args.mode in ["right", "both"]:
            T_right = compute_target_T(
                pivot_right, tip_offset, roll_deg, pitch_deg, yaw_deg
            )

        if args.mode in ["left", "both"]:
            T_left = compute_target_T(
                pivot_left, tip_offset, roll_deg, pitch_deg, -yaw_deg
            )

        head_target = compute_head_target(
            base_head_q=base_head_q,
            roll_delta_deg=roll_delta_deg,
            pitch_delta_deg=pitch_delta_deg,
            yaw_delta_deg=yaw_delta_deg,
            head_max_deg=args.head_max_deg,
        )

        cmd = make_cartesian_cmd(
            T_right=T_right,
            T_left=T_left,
            head_position=head_target,
            min_time=args.move_time,
            hold_time=args.hold_time,
        )

        rv = robot.send_command(cmd, args.priority).get()
        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            print(f"[{i}/{len(rpy_samples)}] command failed, skip sample")
            continue

        time.sleep(args.settle_time)
        q_full = robot.get_state().position.copy()
        q_right = q_full[right_idx].copy()
        q_left = q_full[left_idx].copy()
        q_head = q_full[head_idx].copy()

        if args.mode == "right":
            q_arm = q_right.copy()
        elif args.mode == "left":
            q_arm = q_left.copy()
        else:
            q_arm = np.concatenate([q_right, q_left])

        q_arm_samples.append(q_arm)
        q_right_samples.append(q_right)
        q_left_samples.append(q_left)
        q_head_samples.append(q_head)
        q_full_samples.append(q_full)
        rpy_saved_samples.append([roll_deg, pitch_deg, yaw_deg])
        T_right_samples.append(T_right if T_right is not None else np.full((4, 4), np.nan, dtype=np.float64))
        T_left_samples.append(T_left if T_left is not None else np.full((4, 4), np.nan, dtype=np.float64))

        print(
            f"[{i}/{len(rpy_samples)}] "
            f"rpy=({roll_deg:.2f}, {pitch_deg:.2f}, {yaw_deg:.2f}) deg "
            f"saved"
        )

    out_path = get_output_path(args.out)
    save_dataset(
        out_path=out_path,
        mode=args.mode,
        sampling_mode=args.sampling_mode,
        pivot_right=pivot_right,
        pivot_left=pivot_left,
        tip_offset=tip_offset,
        q_arm_samples=q_arm_samples,
        q_right_samples=q_right_samples,
        q_left_samples=q_left_samples,
        q_head_samples=q_head_samples,
        q_full_samples=q_full_samples,
        rpy_saved_samples=rpy_saved_samples,
        T_right_samples=T_right_samples,
        T_left_samples=T_left_samples,
    )

    print(f"saved npz: {out_path}")
    print(f"saved samples: {len(q_arm_samples)}")

    try:
        robot.cancel_control()
    except Exception:
        pass


if __name__ == "__main__":
    main()
