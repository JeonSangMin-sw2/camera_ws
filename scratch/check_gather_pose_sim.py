"""Joint values of the check pose with the hands brought in to t5 y = +-GAP_HALF (wizard 'Bring Together').

Simulator only (127.0.0.1). Moves to the version's check_calib (torso 0/30/-60/30/0/0), reads both ee poses
in link_torso_5, shifts them along y to +-GAP_HALF keeping x, z and orientation, sends that Cartesian target
and prints the reached arm joints (deg) for config/ready_poses.yaml <version>.check_calib_gather.
usage: python scratch/check_gather_pose_sim.py <version 1.2|1.3> [gap_half_m=0.10]
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rby1_sdk as rby  # noqa: E402
from core.robot.robot_core import connect_robot_session  # noqa: E402
from core.robot.home_offset import movej  # noqa: E402
from core.robot.motion import compute_fk, make_dual_arm_head_cmd  # noqa: E402

VERSION = sys.argv[1]
GAP_HALF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
TORSO = np.radians([0, 30, -60, 30, 0, 0])
check = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))[f"v{VERSION}"]["check_calib"]

robot = connect_robot_session("127.0.0.1:50051", "m", include_head=True, simulated=True)
dyn, model = robot.get_dynamics(), robot.model()
assert movej(robot, torso=TORSO, right_arm=np.radians(check["right_arm"]), left_arm=np.radians(check["left_arm"]), minimum_time=5)
# home_offset.movej always sends the torso to zero; hold it at the check torso instead.
cmd = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
    rby.BodyComponentBasedCommandBuilder()
    .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(4).set_position(TORSO))
    .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(4).set_position(np.radians(check["right_arm"])))
    .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(4).set_position(np.radians(check["left_arm"])))))
assert robot.send_command(cmd, 10).get().finish_code == rby.RobotCommandFeedback.FinishCode.Ok

q = np.array(robot.get_state().position)
_, T_r = compute_fk(robot, dyn, q, "ee_right")
_, T_l = compute_fk(robot, dyn, q, "ee_left")
print("check pose ee y (mm):", round(T_r[1, 3] * 1000, 1), round(T_l[1, 3] * 1000, 1))
T_r2, T_l2 = T_r.copy(), T_l.copy()
T_r2[1, 3], T_l2[1, 3] = -GAP_HALF, GAP_HALF
cmd = make_dual_arm_head_cmd(T_r2, T_l2, ["right", "left"], min_time=5.0, pos_tolerance_m=0.001, ori_tolerance_rad=0.002)
feedback = robot.send_command(cmd, 10).get()
print("finish:", feedback.finish_code)
q = np.array(robot.get_state().position)
_, T_r3 = compute_fk(robot, dyn, q, "ee_right")
_, T_l3 = compute_fk(robot, dyn, q, "ee_left")
print("reached ee (mm): right", np.round(T_r3[:3, 3] * 1000, 1), "left", np.round(T_l3[:3, 3] * 1000, 1))
rot_err = [np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))) for a, b in ((T_r, T_r3), (T_l, T_l3))]
print("orientation change (deg):", np.round(rot_err, 3), "| torso (deg):", np.round(np.degrees(q[list(model.torso_idx)]), 2))
print(f"  check_calib_gather:  # v{VERSION}, hands at t5 y = +-{GAP_HALF * 1000:.0f} mm")
print("    right_arm:", [round(float(v), 4) for v in np.degrees(q[list(model.right_arm_idx)])])
print("    left_arm:", [round(float(v), 4) for v in np.degrees(q[list(model.left_arm_idx)])])
robot.disconnect()
