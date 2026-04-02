import sys
import time
import math
import argparse
import numpy as np
import rby1_sdk as rby
from marker_detection import *


D2R = np.pi / 180.0
MINIMUM_TIME = 2.0
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-4
STOP_POSITION_TRACKING_ERROR = 1e-3


def make_transform(data):
    x, y, z, roll_deg, pitch_deg, yaw_deg = data
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    T = np.eye(4, dtype=np.float32)
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


def initialize_robot(address, model_name, power, servo):
    robot = rby.create_robot(address, model_name)

    if not robot.connect():
        print("Failed to connect robot")
        sys.exit(1)

    robot.set_parameter("default.acceleration_limit_scaling", "1.0")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "20")
    robot.set_parameter("default.angular_acceleration_limit", "10")
    robot.set_parameter("manipulability_threshold", "1e4")

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            print("Failed to power on")
            sys.exit(1)

    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            print("Failed to servo on")
            sys.exit(1)

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MinorFault,
        rby.ControlManagerState.State.MajorFault,
    ]:
        if not robot.reset_fault_control_manager():
            print("Failed to reset fault")
            sys.exit(1)

    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("Failed to enable control manager")
        sys.exit(1)

    return robot


def go_to_ready_pose(robot, model_name):
    if model_name == "a":
        torso = np.array([0, 45, -90, 45, 0, 0]) * D2R
    else:
        torso = np.array([0, 60, -120, 60, 0, 0]) * D2R

    right_arm = np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R
    left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R
    q = np.concatenate([torso, right_arm, left_arm])

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyCommandBuilder().set_command(
                    rby.JointPositionCommandBuilder()
                    .set_position(q)
                    .set_minimum_time(MINIMUM_TIME)
                )
            )
        )
    )

    rv = robot.send_command(cmd, 10).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        print("Failed to go to ready pose")
        sys.exit(1)


def move_left_arm_cartesian(robot, pose):
    T_left = make_transform(pose)

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder().set_left_arm_command(
                    rby.CartesianCommandBuilder()
                    .add_target(
                        "link_torso_5",
                        "ee_left",
                        T_left,
                        LINEAR_VELOCITY_LIMIT,
                        ANGULAR_VELOCITY_LIMIT,
                        ACCELERATION_LIMIT,
                    )
                    .set_minimum_time(MINIMUM_TIME)
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(1)
                    )
                    .set_stop_orientation_tracking_error(
                        STOP_ORIENTATION_TRACKING_ERROR
                    )
                    .set_stop_position_tracking_error(
                        STOP_POSITION_TRACKING_ERROR
                    )
                )
            )
        )
    )

    rv = robot.send_command(cmd, 10).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        print("Failed to conduct motion")
        sys.exit(1)


# def print_fk(robot):
#     model = robot.model()
#     dyn = robot.get_dynamics()
#     q = robot.get_state().position.copy()

#     state_r = dyn.make_state(["link_torso_5", "ee_right"], model.robot_joint_names)
#     state_l = dyn.make_state(["link_torso_5", "ee_left"], model.robot_joint_names)

#     state_r.set_q(q)
#     state_l.set_q(q)

#     dyn.compute_forward_kinematics(state_r)
#     dyn.compute_forward_kinematics(state_l)

#     T_right = dyn.compute_transformation(state_r, 0, 1)
#     T_left = dyn.compute_transformation(state_l, 0, 1)

#     print("Right FK matrix:")
#     print(T_right)
#     print("Left FK matrix:")
#     print(T_left)


def main(address, model_name, power, servo):
    marker = Marker_Transform(Stereo=False, monitoring = False)
    robot = initialize_robot(address, model_name, power, servo)

    go_to_ready_pose(robot, model_name)
    time.sleep(1)

    poses = [
        [0.4, 0.2, 0.0, 0, -90, 0],
        [0.5, 0.2, 0.0, 0, -90, 0],
        [0.5, 0.3, 0.0, 0, -90, 0],
        [0.5, 0.3, 0.1, 0, -90, 0],
    ]

    move_left_arm_cartesian(robot, poses[0])
    # print_fk(robot)
    time.sleep(2)
    data = marker.get_marker_transform(sampl_size=2)
    print(data)
    for pose in poses[1:]:
        move_left_arm_cartesian(robot, pose)
        time.sleep(2)
        data = marker.get_marker_transform(sample_size=2)
        print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--model", type=str, default="a")
    parser.add_argument("--power", type=str, default=".*")
    parser.add_argument("--servo", type=str, default="^(?!.*head).*")
    args = parser.parse_args()

    main(args.address, args.model, args.power, args.servo)