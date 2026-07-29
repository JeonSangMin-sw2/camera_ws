import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config, generate_sim_measurements
from main_ui import build_incremental_motion_plan
import rby1_sdk as rby

r_holder = rby.create_robot("127.0.0.1", "m")
class MockRobot:
    def model(self): return r_holder.model()
    def get_dynamics(self): return r_holder.get_dynamics()
    def get_state(self):
        class S: position = np.zeros(20)
        return S()

mock_robot = MockRobot()
dyn_model = mock_robot.get_dynamics()
model = mock_robot.model()

active_arms = ["right", "left"]
cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

class AutoConfig:
    def __init__(self):
        self.angle_step_deg = 5.0
        self.max_cartesian_x_m = 0.4
        self.step_x_m = 0.03
        self.cartesian_offset_m = 0.03
auto_config = AutoConfig()

plan = build_incremental_motion_plan(mock_robot, dyn_model, auto_config, active_arms, include_head_motion=True)

q_arm_list = []
q_head_list = []
q_right_curr = np.zeros(7)
q_left_curr = np.zeros(7)
q_head_curr = np.zeros(2)

for p in plan:
    p_type = p.get("type", "cartesian")
    if p_type == "joint":
        if "offsets_dict" in p and p["offsets_dict"] is not None:
            for j_i, off in p["offsets_dict"].items():
                if j_i == 2:
                    q_right_curr[j_i] += np.deg2rad(off)
                    q_left_curr[j_i] += np.deg2rad(-off)
                else:
                    q_right_curr[j_i] += np.deg2rad(off)
                    q_left_curr[j_i] += np.deg2rad(off)
        else:
            j_i = p["joint_idx"]
            off = p["offset_deg"]
            if j_i == 2:
                q_right_curr[j_i] += np.deg2rad(off)
                q_left_curr[j_i] += np.deg2rad(-off)
            else:
                q_right_curr[j_i] += np.deg2rad(off)
                q_left_curr[j_i] += np.deg2rad(off)
    elif p_type == "restore_baseline":
        q_right_curr = np.zeros(7)
        q_left_curr = np.zeros(7)

    if "head_q" in p and p["head_q"] is not None:
        q_head_curr = p["head_q"]

    q_arm_list.append(np.concatenate([q_right_curr, q_left_curr]))
    q_head_list.append(q_head_curr.copy())

q_arm_list = np.array(q_arm_list)
q_head_list = np.array(q_head_list)

T_meas_list = generate_sim_measurements(
    robot=mock_robot,
    dyn_model=dyn_model,
    q_arm_list=q_arm_list,
    q_head_list=q_head_list,
    arm_idx=cfg["arm_idx"],
    head_idx=head_cfg["head_idx"],
    q_nominal=np.zeros(20),
    optimize_arm=True,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg["head_base_to_cam_nom"],
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    camera_position_noise_std_m=0.000042,
    camera_orientation_noise_std_deg=0.01
)

staged_offsets = {
    "right": {"joint3": -0.4955, "joint5": -5.4697, "joint6": -2.3879},
    "left":  {"joint3": -0.6914, "joint5": 2.9833,  "joint6": -3.5370}
}

opt_st1 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=cfg["arm_idx"],
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    head_idx=head_cfg["head_idx"],
    eps=1e-6,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=staged_offsets,
)
q_arm_offset, q_head_offset, xi_cam, _, _ = opt_st1.optimize(q_arm_list, q_head_list, T_meas_list)

opt_st3 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=cfg["arm_idx"],
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    head_idx=head_cfg["head_idx"],
    eps=1e-7,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=staged_offsets,
)
q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, _ = opt_st3.optimize(
    q_arm_list, q_head_list, T_meas_list,
    q_arm_offset_init=q_arm_offset,
    q_head_offset_init=q_head_offset,
    xi_mount_cam_init=xi_cam
)

calc_r = np.rad2deg(q_arm_offset[:7])
calc_l = np.rad2deg(q_arm_offset[7:])
calc_h = np.rad2deg(q_head_offset)

gt_r = np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3])
gt_l = np.array([-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5])
gt_h = np.array([0.8, -1.6])

err_r = np.abs(calc_r - gt_r)
err_l = np.abs(calc_l - gt_l)
err_h = np.abs(calc_h - gt_h)

with open("/home/rainbow/camera_ws/scratch/sweep_eval.txt", "w") as f:
    f.write(f"RIGHT ERRORS: {np.round(err_r, 4)}\n")
    f.write(f"LEFT ERRORS : {np.round(err_l, 4)}\n")
    f.write(f"HEAD ERRORS : {np.round(err_h, 4)}\n")
    f.write(f"MAX ERROR   : {max(np.max(err_r), np.max(err_l), np.max(err_h)):.4f} deg\n")
    f.flush()

os._exit(0)
