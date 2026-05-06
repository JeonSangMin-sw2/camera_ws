import time
import numpy as np
import rby1_sdk as rby
from dataclasses import dataclass

D2R = np.pi / 180.0

@dataclass
class AutoCollectionConfig:
    angle_step_deg: float = 5.0
    position_step_m: float = 0.03
    max_x: float = 0.5
    move_time: float = 1.2
    settle_time: float = 0.6
    hold_time: float = 3.0
    priority: int = 10

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

def apply_cartesian_offset(T, dx=0.0, dy=0.0, dz=0.0, droll_deg=0.0, dpitch_deg=0.0, dyaw_deg=0.0):
    T_new = T.copy()
    T_new[0, 3] += dx
    T_new[1, 3] += dy
    T_new[2, 3] += dz
    
    R_off = rot_z(np.deg2rad(dyaw_deg)) @ rot_y(np.deg2rad(dpitch_deg)) @ rot_x(np.deg2rad(droll_deg))
    # Apply rotation in tool frame (right-multiply) to keep marker in view more easily
    T_new[:3, :3] = T_new[:3, :3] @ R_off
    return T_new

def compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5"):
    state = dyn_model.make_state(
        [base_link, ee_link],
        robot.model().robot_joint_names
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    return state, dyn_model.compute_transformation(state, 0, 1)

def build_incremental_motion_plan(robot, dyn_model, config: AutoCollectionConfig):
    """
    현재 자세를 읽어서 X축으로 전진하며 RPY/YZ 오프셋 타겟들을 생성합니다.
    """
    state = robot.get_state()
    q_full = state.position
    _, T_base_right = compute_fk(robot, dyn_model, q_full, "ee_right", "link_torso_5")
    _, T_base_left = compute_fk(robot, dyn_model, q_full, "ee_left", "link_torso_5")
    
    plan = []
    T_curr_right = T_base_right.copy()
    T_curr_left = T_base_left.copy()
    
    while True:
        curr_x = T_curr_right[0, 3]
        if curr_x > config.max_x:
            break
            
        rpy_targets = [
            (config.angle_step_deg, 0, 0), (-config.angle_step_deg, 0, 0),
            (0, config.angle_step_deg, 0), (0, -config.angle_step_deg, 0),
            (0, 0, config.angle_step_deg), (0, 0, -config.angle_step_deg)
        ]
        for dr, dp, dy in rpy_targets:
            tr = apply_cartesian_offset(T_curr_right, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            tl = apply_cartesian_offset(T_curr_left, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            plan.append({
                "T_right": tr, "T_left": tl,
                "desc": f"RPY: ({dr},{dp},{dy})"
            })
            
        yz_targets = [
            (0, config.position_step_m, 0), (0, -config.position_step_m, 0),
            (0, 0, config.position_step_m), (0, 0, -config.position_step_m)
        ]
        for dx, dy, dz in yz_targets:
            tr = apply_cartesian_offset(T_curr_right, dx=dx, dy=dy, dz=dz)
            tl = apply_cartesian_offset(T_curr_left, dx=dx, dy=dy, dz=dz)
            plan.append({
                "T_right": tr, "T_left": tl,
                "desc": f"Pos: ({dx},{dy},{dz})"
            })
            
        T_curr_right = apply_cartesian_offset(T_curr_right, dx=config.position_step_m)
        T_curr_left = apply_cartesian_offset(T_curr_left, dx=config.position_step_m)
        
    return plan

def move_to_auto_ready_pose(robot, active_arms, minimum_time=5.0, priority=10):
    # Step 1: Joint Ready Pose (go_to_ready_pose 기준)
    q_torso = np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R
    
    if "right" in active_arms:
        q_right = np.array([-45, -30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    else:
        q_right = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
        
    if "left" in active_arms:
        q_left = np.array([-45, 30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    else:
        q_left = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
        
    q_ready = np.concatenate([q_torso, q_right, q_left])
    
    print("Step 1: Moving to Joint Ready Pose...")
    cmd1 = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.JointPositionCommandBuilder()
            .set_position(q_ready)
            .set_minimum_time(minimum_time)
        )
    )
    rv1 = robot.send_command(cmd1, priority).get()
    if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to Step 1: Joint Ready Pose.")

    # Step 2: Cartesian Checking Pose (go_to_calibration_checking_pose 기준, offset=0.2)
    # Raising Z-axis to 0.2m (down 20cm from previous 0.4m) and rotating 6th axis (wrist) by 180 degrees (@ rot_z(180))
    T_right = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(90*D2R), [0.3, -0.2, 0.2])
    T_right[:3, :3] = T_right[:3, :3] @ rot_z(180*D2R)
    
    T_left = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(-90*D2R), [0.3, 0.2, 0.2])
    T_left[:3, :3] = T_left[:3, :3] @ rot_z(180*D2R)

    body2 = rby.BodyComponentBasedCommandBuilder()
    body2.set_torso_command(
        rby.JointPositionCommandBuilder()
        .set_position(q_torso)
        .set_minimum_time(minimum_time)
    )

    if "right" in active_arms:
        body2.set_right_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_right", T_right, 0.5, 1.0, 0.3)
            .set_stop_position_tracking_error(0.005)
            .set_stop_orientation_tracking_error(0.02)
            .set_minimum_time(minimum_time)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
        )
    else:
        body2.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
            .set_minimum_time(minimum_time)
        )

    if "left" in active_arms:
        body2.set_left_arm_command(
            rby.CartesianCommandBuilder()
            .add_target("link_torso_5", "ee_left", T_left, 0.5, 1.0, 0.3)
            .set_stop_position_tracking_error(0.005)
            .set_stop_orientation_tracking_error(0.02)
            .set_minimum_time(minimum_time)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
        )
    else:
        body2.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
            .set_minimum_time(minimum_time)
        )

    print("Step 2: Moving to Cartesian Checking Pose...")
    cmd2 = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body2)
    )
    rv2 = robot.send_command(cmd2, priority).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to Step 2: Cartesian Checking Pose.")

def make_dual_arm_head_cmd(T_right, T_left, active_arms, min_time=1.2, hold_time=3.0):
    body = rby.BodyComponentBasedCommandBuilder()

    # Always lock Torso to stable pose
    body.set_torso_command(
        rby.JointPositionCommandBuilder()
        .set_position(np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R)
        .set_minimum_time(min_time)
    )

    if "right" in active_arms:
        if T_right is not None:
            body.set_right_arm_command(
                rby.CartesianCommandBuilder()
                .add_target("link_torso_5", "ee_right", T_right, 0.2, 0.5, 0.3)
                .set_stop_position_tracking_error(0.001)
                .set_stop_orientation_tracking_error(0.005)
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold_time))
                .set_minimum_time(min_time)
            )
    else:
        # Lock inactive right arm
        body.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
            .set_minimum_time(min_time)
        )

    if "left" in active_arms:
        if T_left is not None:
            body.set_left_arm_command(
                rby.CartesianCommandBuilder()
                .add_target("link_torso_5", "ee_left", T_left, 0.2, 0.5, 0.3)
                .set_stop_position_tracking_error(0.001)
                .set_stop_orientation_tracking_error(0.005)
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold_time))
                .set_minimum_time(min_time)
            )
    else:
        # Lock inactive left arm
        body.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
            .set_minimum_time(min_time)
        )

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(body)
    return rby.RobotCommandBuilder().set_command(cmd)

def execute_auto_motion_step(robot, config, motion_plan_step, active_arms):
    T_right = motion_plan_step["T_right"]
    T_left = motion_plan_step["T_left"]

    cmd = make_dual_arm_head_cmd(
        T_right=T_right,
        T_left=T_left,
        active_arms=active_arms,
        min_time=config.move_time,
        hold_time=config.hold_time,
    )
    rv = robot.send_command(cmd, config.priority).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Auto motion command failed: {rv.finish_code}")

    time.sleep(config.settle_time)
    return motion_plan_step
