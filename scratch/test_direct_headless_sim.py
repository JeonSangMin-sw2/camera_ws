import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rby1_sdk as rby
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_arm_config, get_head_config
from core.robot_motion import build_incremental_motion_plan, AutoCollectionConfig
from main_ui import SimulatedMarkerTransform

D2R = np.pi / 180.0

# 1. Connect to model / robot
robot = rby.create_robot("127.0.0.1", "a")
model = robot.model()
dyn = robot.get_dynamics()

camera_config = {
    "mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
    "head_base_to_cam": [0.098, 0.009, 0.012, -90.0, 0.0, -90.0],
    "Tf_to_marker_right": [0.0, -0.054, -0.048, 90.0, 0.0, 180.0],
    "Tf_to_marker_left": [0.0, 0.054, -0.048, 90.0, 0.0, 0.0],
}

# 2. Create SimulatedMarkerTransform with headless mode (include_head_motion=False)
sim_marker = SimulatedMarkerTransform(robot, camera_config, "1.2", include_head_motion=False)

# 3. Generate Motion Plan for headless robot
config = AutoCollectionConfig()
config.angle_step_deg = 5.0
config.position_step_m = 0.03
config.step_x_m = 0.03
config.max_x = 0.4
config.max_loops = 1

plan = build_incremental_motion_plan(robot, dyn, config, ["right", "left"], include_head_motion=False)
print(f"Generated {len(plan)} motion steps for headless calibration.")

# 4. Collect samples by executing plan in simulation
q_arm_list = []
T_meas_list = []

# Mock robot joint state: set ready pose
q_state = np.zeros(len(robot.get_state().position))

for step in plan:
    # Get joint targets
    tr = step.get("T_right")
    tl = step.get("T_left")
    
    # Capture marker transform from simulated camera
    T_r = sim_marker.get_marker_transform(0, "right")[0]
    T_l = sim_marker.get_marker_transform(0, "left")[0]
    
    # Save commanded q
    r_idx = list(model.right_arm_idx)
    l_idx = list(model.left_arm_idx)
    q_arm = np.concatenate([q_state[r_idx], q_state[l_idx]])
    
    q_arm_list.append(q_arm)
    T_meas_list.append([T_r, T_l])

q_arm_list = np.array(q_arm_list)
T_meas_list = np.array(T_meas_list)

print(f"Collected {len(q_arm_list)} samples.")

# 5. Run QP Optimization with Locked Step 1 values
gt_r = BaseCalibrator.MOCK_GT_OFFSETS["right"]
gt_l = BaseCalibrator.MOCK_GT_OFFSETS["left"]

# Ground truth values:
# Right: J6=+2.30, J5=+5.40, J3=+0.50 -> offset = -2.30, -5.40, -0.50
# Left:  J6=+3.50, J5=-3.00, J3=+0.70 -> offset = -3.50, +3.00, -0.70
joint_offsets_to_apply = {
    "right": {"joint3": -gt_r["joint3"], "joint5": -gt_r["joint5_v12"], "joint6": -gt_r["joint6"]},
    "left":  {"joint3": -gt_l["joint3"], "joint5": -gt_l["joint5_v12"], "joint6": -gt_l["joint6"]},
}

cfg_r = get_arm_config(model, "right", version="1.2")
opt_r = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_r["arm_idx"],
    ee_links={"right": cfg_r["ee_link"]},
    mount_to_cam_nom=camera_config["mount_to_cam"],
    head_base_to_cam_nom=camera_config["head_base_to_cam"],
    ee_to_marker_nom={"right": camera_config["Tf_to_marker_right"]},
    active_arms=["right"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=None,
    use_head_kinematics=False,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=joint_offsets_to_apply,
)

qr, hr, xir, mount_to_cam_r, head_base_to_cam_r = opt_r.optimize(
    q_arm_list[:, :7], None, T_meas_list[:, 0]
)

print("\n================ RIGHT ARM OPTIMIZATION RESULT ================")
print("Calibrated Right Arm Offsets (deg):", np.round(np.rad2deg(qr), 4))
print("GT Right Arm Offsets (deg):        ", [-gt_r.get(f"joint{i}", 0.0) if i != 5 else -gt_r["joint5_v12"] for i in range(7)])

cfg_l = get_arm_config(model, "left", version="1.2")
opt_l = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_l["arm_idx"],
    ee_links={"left": cfg_l["ee_link"]},
    mount_to_cam_nom=camera_config["mount_to_cam"],
    head_base_to_cam_nom=camera_config["head_base_to_cam"],
    ee_to_marker_nom={"left": camera_config["Tf_to_marker_left"]},
    active_arms=["left"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=None,
    use_head_kinematics=False,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=joint_offsets_to_apply,
)

ql, hl, xil, mount_to_cam_l, head_base_to_cam_l = opt_l.optimize(
    q_arm_list[:, 7:], None, T_meas_list[:, 1]
)

print("\n================ LEFT ARM OPTIMIZATION RESULT ================")
print("Calibrated Left Arm Offsets (deg):", np.round(np.rad2deg(ql), 4))
print("GT Left Arm Offsets (deg):       ", [-gt_l.get(f"joint{i}", 0.0) if i != 5 else -gt_l["joint5_v12"] for i in range(7)])
