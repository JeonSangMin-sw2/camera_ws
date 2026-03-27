import argparse
import time
import signal
import sys
import numpy as np
import rby1_sdk as rby

D2R = np.pi / 180.0
running = True


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
    q_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R

    # 필요하면 여기만 수정
    # q_right = np.array([-1.368 ,-0.837,  1.110 ,-0.994 ,-1.745 , 1.782, -0.509])
    q_right = np.array([-0.606 ,-1.634, 0.960, -2.410,  0.410 , 1.047 ,-2.196])
    
    
    # q_left = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R
    q_left =  np.array([-0.606 ,1.634, -0.960, -2.410,  -0.410 , 1.047 ,2.196])
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


def make_cartesian_cmd(T_right=None, T_left=None, min_time=0.2, hold_time=1e6):
    body = rby.BodyComponentBasedCommandBuilder()

    if T_right is not None:
        body.set_right_arm_command(
            rby.CartesianCommandBuilder()
            .add_target(
                "link_torso_5",
                "ee_right",
                T_right,
                0.2,   # linear vel
                0.5,   # angular vel
                0.3,   # accel
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
                0.2,   # linear vel
                0.5,   # angular vel
                0.3,   # accel
            )
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(min_time)
        )

    return (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body)
        )
    )


def get_base_rotation(direction: str):
    # "down": ee의 기본 방향을 아래로 두고 cone motion
    # "x":    ee의 기본 방향을 x축 방향으로 두고 cone motion
    #
    # 여기서 실제 ee 축 정의가 현장과 다를 수 있으므로
    # down / x가 기대와 다르면 아래 행렬만 바꿔주면 됨.

    if direction == "down":
        return np.eye(3)

    if direction == "x":
        # ee 기본축(z축이라고 가정)을 x축으로 돌리는 용도
        return rot_y(-np.pi / 2)

    raise ValueError(f"unsupported direction: {direction}")

def compute_target_T(pivot, tip_offset, roll_deg, pitch_deg, yaw_deg, direction):
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    R_base = get_base_rotation(direction)

    if direction == "x":
        R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ R_base
    else:
        R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ R_base

    p_ee = pivot - R @ tip_offset
    return make_T(R, p_ee)

def main():
    global running

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="m")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="torso_.*|right_arm_.*|left_arm_.*")

    parser.add_argument("--mode", type=str, default="right", choices=["right", "left", "both"])
    parser.add_argument("--direction", type=str, default="down", choices=["down", "x"])

    # right pivot
    parser.add_argument("--pivot_x", type=float, default=0.28)
    parser.add_argument("--pivot_y", type=float, default=-0.40)
    parser.add_argument("--pivot_z", type=float, default=-0.1)

    # ee -> tip offset (ee frame)
    parser.add_argument("--tip_offset_x", type=float, default=0.0)
    parser.add_argument("--tip_offset_y", type=float, default=0.0)
    parser.add_argument("--tip_offset_z", type=float, default=-0.09191)
    parser.add_argument("--axis", type=str, default="roll", choices=["roll", "pitch", "yaw"])
    parser.add_argument("--max_roll_deg", type=float, default=20.0)
    parser.add_argument("--max_pitch_deg", type=float, default=20.0)
    parser.add_argument("--max_yaw_deg", type=float, default=20.0)
    parser.add_argument("--freq", type=float, default=0.1)

    # streaming
    parser.add_argument("--dt", type=float, default=0.05)  # 20 Hz
    parser.add_argument("--duration", type=float, default=0.0,
                        help="0이면 무한 실행, 양수면 해당 초 후 종료")

    args = parser.parse_args()

    robot = init_robot(args.address, args.model, args.power, args.servo)
    go_ready(robot)
    time.sleep(1.0)

    # right pivot
    pivot_right = np.array([args.pivot_x, args.pivot_y, args.pivot_z], dtype=np.float64)

    # left pivot: y 대칭
    pivot_left = np.array([args.pivot_x, -args.pivot_y, args.pivot_z], dtype=np.float64)

    # ee -> tip
    tip_offset = np.array(
        [args.tip_offset_x, args.tip_offset_y, args.tip_offset_z],
        dtype=np.float64,
    )

    print(f"mode       : {args.mode}")
    print(f"direction  : {args.direction}")
    print(f"axis       : {args.axis}")
    print(f"pivot_right: {pivot_right}")
    print(f"pivot_left : {pivot_left}")
    print(f"tip_offset : {tip_offset}")

    stream = robot.create_command_stream(priority=1)

    def handler(sig, frame):
        global running
        running = False
        try:
            robot.cancel_control()
        except Exception:
            pass

    signal.signal(signal.SIGINT, handler)

    start = time.time()

    while running:
        t = time.time() - start

        if args.duration > 0.0 and t > args.duration:
            break

        s = np.sin(2 * np.pi * args.freq * t)

        roll_deg = 0.0
        pitch_deg = 0.0
        yaw_deg = 0.0

        if args.axis == "roll":
            roll_deg = args.max_roll_deg * s
        elif args.axis == "pitch":
            pitch_deg = args.max_pitch_deg * s
        else:
            yaw_deg = args.max_yaw_deg * s

        T_right = None
        T_left = None

        if args.mode in ["right", "both"]:
            T_right = compute_target_T(
                pivot_right, tip_offset, roll_deg, pitch_deg, yaw_deg, args.direction
            )

        if args.mode in ["left", "both"]:
            T_left = compute_target_T(
                pivot_left, tip_offset, roll_deg, pitch_deg, -yaw_deg, args.direction
            )

        cmd = make_cartesian_cmd(T_right=T_right, T_left=T_left)
        stream.send_command(cmd)

        time.sleep(args.dt)

    try:
        robot.cancel_control()
    except Exception:
        pass


if __name__ == "__main__":
    main()
