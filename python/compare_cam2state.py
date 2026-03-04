# Motion Demo
# This example is part of the RB-Y1 SDK examples. See --help for arguments.
#
# Usage example:
#     python 09_demo_motion.py --help
#
# Copyright (c) 2025 Rainbow Robotics. All rights reserved.
#
# DISCLAIMER:
# This is a sample code provided for educational and reference purposes only.
# Rainbow Robotics shall not be held liable for any damages or malfunctions resulting from
# the use or misuse of this demo code. Please use with caution and at your own discretion.
import pyrealsense2 as rs
import rby1_sdk
import numpy as np
import sys
import time
import argparse
import re
from rby1_sdk import *
import cv2
import socket
import struct
import math
import threading


D2R = np.pi / 180  # Degree to Radian conversion factor
MINIMUM_TIME = 3
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e+1
STOP_POSITION_TRACKING_ERROR = 1e-1
WEIGHT = 1
STOP_COST = WEIGHT * WEIGHT * 2e-3
MIN_DELTA_COST = WEIGHT * WEIGHT * 2e-3
PATIENCE = 10

robot = None

def compute_T_fk(q):
    model = robot.model()
    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(
        ["link_torso_5", "ee_right"],
        model.robot_joint_names
    )
    dyn_state.set_q(q)
    dyn_robot.compute_forward_kinematics(dyn_state)
    T_fk = dyn_robot.compute_transformation(dyn_state, 0, 1)
    
    print("Full FK matrix:")
    print(T_fk)
    return np.round(T_fk[:3, 3], 2)

def cb(rs):
    # print(f"Timestamp: {rs.timestamp - rs.ft_sensor_right.time_since_last_update}")
    # position = rs.position * 180 / 3.141592
    # print(f"torso [deg]: {position[2:2 + 6]}")
    # print(f"right arm [deg]: {position[8:8 + 7]}")
    # print(f"left arm [deg]: {position[15:15 + 7]}")
    position = rs.position
    # print("T_cam pos:")
    print(compute_T_fk(position))
    
def example_joint_position_command_1(robot, model_name):
    print("joint position command example 2")

    # Define joint positions
    if model_name == "a":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
    elif model_name == "m":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
        
    q_joint_right_arm =  np.array([
    -0.95036944, -1.88217164, -0.3146673 ,
    -2.14031707, -0.0138135 , -1.33042154,
    -1.37736902
    ])
    q_joint_left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R

    # Combine joint positions
    q = np.concatenate([q_joint_torso, q_joint_right_arm, q_joint_left_arm])

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyCommandBuilder().set_command(
                JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(MINIMUM_TIME)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def example_joint_position_command_2(robot, model_name):
    print("joint position command example 2")

    # Define joint positions
    if model_name == "a":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
    elif model_name == "m":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
        
    q_joint_right_arm = np.array([
    -1.18386486, -1.02091358,  0.26658612,
    -1.98402569,  0.23533566, -1.47095335,
    -0.810287
])
    q_joint_left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R

    # Combine joint positions
    q = np.concatenate([q_joint_torso, q_joint_right_arm, q_joint_left_arm])

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyCommandBuilder().set_command(
                JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(MINIMUM_TIME)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def example_cartesian_command_1(robot, model_name):

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    # T_torso[:3, :3] = np.eye(3)
    # T_torso[:3, 3] = [0, 0, 1]
    T_right = np.array([
    [-0.99002546, -0.07476161,  0.11941658,  0.05861272],
    [-0.07933656, -0.40460166, -0.9110451 , -0.2069812 ],
    [ 0.11642735, -0.9114319 ,  0.39463466,  0.2939942 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    # angle = -np.pi / 2
    # T_right[:3, :3] = np.array(
    #     [
    #         [np.cos(angle), 0, np.sin(angle)],
    #         [0, 1, 0],
    #         [-np.sin(angle), 0, np.cos(angle)],
    #     ]
    # )
    # T_right[:3, 3] = [0.5, -0.3, 1.0]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            # .set_torso_command(
            #     CartesianCommandBuilder()
            #     .add_target(
            #         "base",
            #         target_link,
            #         T_torso,
            #         LINEAR_VELOCITY_LIMIT,
            #         ANGULAR_VELOCITY_LIMIT,
            #         ACCELERATION_LIMIT,
            #     )
            #     .set_minimum_time(MINIMUM_TIME)
            #     .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
            #     .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            # )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "link_torso_5",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0




def example_cartesian_command_2(robot, model_name):

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    # T_torso[:3, :3] = np.eye(3)
    # T_torso[:3, 3] = [0, 0, 1]
    T_right = np.array([
    [-0.67121524,  0.13929945,  0.7280562 ,  0.20793527],
    [-0.74017674, -0.07281009, -0.6684587 , -0.15390995],
    [-0.0401061 , -0.9875699 ,  0.15197748,  0.2366795 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    # angle = -np.pi / 2
    # T_right[:3, :3] = np.array(
    #     [
    #         [np.cos(angle), 0, np.sin(angle)],
    #         [0, 1, 0],
    #         [-np.sin(angle), 0, np.cos(angle)],
    #     ]
    # )
    # T_right[:3, 3] = [0.5, -0.3, 1.0]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            # .set_torso_command(
            #     CartesianCommandBuilder()
            #     .add_target(
            #         "base",
            #         target_link,
            #         T_torso,
            #         LINEAR_VELOCITY_LIMIT,
            #         ANGULAR_VELOCITY_LIMIT,
            #         ACCELERATION_LIMIT,
            #     )
            #     .set_minimum_time(MINIMUM_TIME)
            #     .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
            #     .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            # )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "link_torso_5",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0



def example_cartesian_command_3(robot, model_name):
    print("move X axis")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.4, -0.3, 0.9]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def example_cartesian_command_4(robot, model_name):
    print("move Y axis")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.4, -0.2, 0.9]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def go_to_home_pose_1(robot, model_name):
    print("Go to home pose 1")

    if model_name == "a":
        q_joint_torso = np.zeros(6)
    elif model_name == "m":
        q_joint_torso = np.zeros(6)
        
    q_joint_right_arm = np.zeros(7)
    q_joint_left_arm = np.zeros(7)

    q_joint_right_arm[1] = -135 * D2R
    q_joint_left_arm[1] = 135 * D2R

    # Send command to go to ready position
    rv = robot.send_command(
        RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                BodyComponentBasedCommandBuilder()
                .set_torso_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_torso)
                )
                .set_right_arm_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_right_arm)
                )
                .set_left_arm_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_left_arm)
                )
            )
        ),
        10,
    ).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def go_to_home_pose_2(robot, model_name):
    print("Go to home pose 2")

    if model_name =="a":
        target_joint = np.zeros(20)
    elif model_name == "m":
        target_joint = np.zeros(20)
        
    # Send command to go to home pose
    rv = robot.send_command(
        RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                JointPositionCommandBuilder()
                .set_position(target_joint)
                .set_minimum_time(MINIMUM_TIME)
            )
        ),
        10,
    ).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def main(address, model_name, power, servo):
    print("Attempting to connect to the robot...")


    global robot
    # robot_global = rby1_sdk.create_robot(address, model_name)
    
    robot = rby1_sdk.create_robot(address, model_name)

    if not robot.connect():
        print("Error: Unable to establish connection to the robot at")
        sys.exit(1)

    print("Successfully connected to the robot")

    print("Starting state update...")
    robot.start_state_update(cb, 10)

    # robot.factory_reset_all_parameters()
    robot.set_parameter("default.acceleration_limit_scaling", "1.0")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "20")
    robot.set_parameter("default.angular_acceleration_limit", "10")
    robot.set_parameter("manipulability_threshold", "1e4")
    # robot.set_time_scale(1.0)

    print("parameters setting is done")

    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)

    if not robot.is_power_on(power):
        rv = robot.power_on(power)
        if not rv:
            print("Failed to power on")
            exit(1)

    print(servo)
    if not robot.is_servo_on(servo):
        rv = robot.servo_on(servo)
        if not rv:
            print("Fail to servo on")
            exit(1)

    control_manager_state = robot.get_control_manager_state()

    if (
        control_manager_state.state == rby1_sdk.ControlManagerState.State.MinorFault
        or control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault
    ):

        if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
            print(
                "Warning: Detected a Major Fault in the Control Manager!!!!!!!!!!!!!!!."
            )
        else:
            print(
                "Warning: Detected a Minor Fault in the Control Manager@@@@@@@@@@@@@@@@."
            )

        print("Attempting to reset the fault...")
        if not robot.reset_fault_control_manager():
            print("Error: Unable to reset the fault in the Control Manager.")
            sys.exit(1)
        print("Fault reset successfully.")

    print("Control Manager state is normal. No faults detected.")

    print("Enabling the Control Manager...")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("Error: Failed to enable the Control Manager.")
        sys.exit(1)
    print("Control Manager enabled successfully.")

    BASE, EE = 0, 1
    def compute_T_fk(q):
        model = robot.model()
        dyn_robot = robot.get_dynamics()
        dyn_state = dyn_robot.make_state(["link_torso_5", "ee_right"], model.robot_joint_names)
        q_ = q
        dyn_state.set_q(q_)
        dyn_robot.compute_forward_kinematics(dyn_state)
        T_fk = dyn_robot.compute_transformation(dyn_state, BASE, EE)
        return np.round(T_fk[:3, 3],2)

    # marker_transform = Marker_Transform()
    print("motion based state_q")
    example_joint_position_command_1(robot, model_name)
    time.sleep(4)
    print("motion based T_cam")
    example_cartesian_command_1(robot, model_name)
    time.sleep(4)
    print("break time====")

    
    # time.sleep(1)
    # print("motion based state_q2")
    # example_joint_position_command_2(robot, model_name)
    # time.sleep(1)
    # print("motion based T_cam")
    # example_cartesian_command_2(robot, model_name)
    # result = marker_transform.get_marker_transform()
    print("camera_resulte=")
    print(result)
    print("t5 to ee=")
    print(compute_T_fk(robot.get_state().position))
    # if not example_cartesian_command_2(robot, model_name):
    #     print("finish motion")
    # compute_T_fk(robot.get_state().position)
    # result = marker_transform.get_marker_transform()
    # print("camera_resulte=")
    # print(result)
    # print("t5 to ee=")
    # print(compute_T_fk(robot.get_state().position))
    # if not example_cartesian_command_3(robot, model_name):
    #     print("finish motion")
    # compute_T_fk(robot.get_state().position)
    # result = marker_transform.get_marker_transform()
    # print("camera_resulte=")
    # print(result)
    # print("t5 to ee=")
    # print(compute_T_fk(robot.get_state().position))
    # if not example_cartesian_command_4(robot, model_name):
    #     print("finish motion")
    # compute_T_fk(robot.get_state().position)
    # result = marker_transform.get_marker_transform()
    # print("camera_resulte=")
    # print(result)
    # print("t5 to ee=")
    # print(compute_T_fk(robot.get_state().position))
    # if not go_to_home_pose_1(robot):
    #     print("finish motion")
    # if not go_to_home_pose_2(robot, model_name):
    #     print("finish motion")

    print("end of demo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="09_demo_motion")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default='a', help="Robot Model Name (default: 'a')")
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default=".*",
        help="Servo name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    main(address=args.address, model_name = args.model, power=args.power, servo=args.servo)
