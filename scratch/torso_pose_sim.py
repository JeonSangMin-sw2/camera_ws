"""Move the SIMULATOR (127.0.0.1 only) to one torso pose for screenshots; arms and head at zero.
usage: python scratch/torso_pose_sim.py t0 t1 t2 t3 t4 t5   (deg)"""
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.robot.robot_core import connect_robot_session

torso = np.radians([float(v) for v in sys.argv[1:7]])
robot = connect_robot_session("127.0.0.1:50051", "m", include_head=True, simulated=True, log_callback=lambda *_: None)
body = (rby.BodyComponentBasedCommandBuilder()
        .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(2.5).set_position(torso))
        .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(2.5).set_position(np.radians([0, -15, 0, -20, 0, 0, 0])))
        .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(2.5).set_position(np.radians([0, 15, 0, -20, 0, 0, 0]))))
cmd = rby.RobotCommandBuilder().set_command(
    rby.ComponentBasedCommandBuilder().set_body_command(body)
    .set_head_command(rby.JointPositionCommandBuilder().set_minimum_time(2.5).set_position(np.zeros(2))))
rv = robot.send_command(cmd, 10).get()
q = np.degrees(robot.get_state().position[4:10])
print("finish", rv.finish_code, "torso deg", np.round(q, 2))
