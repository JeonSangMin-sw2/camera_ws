import sys
sys.path.insert(0, '/home/rainbow/camera_ws')

import numpy as np
import rby1_sdk
from core.calibration_optimizer import QPCalibrationOptimizer, compute_fk
from core.calibration_core import get_both_arm_config, get_arm_config, get_head_config

class OfflineRobotWrapper:
    def __init__(self, version="1.2"):
        urdf_path = '/home/rainbow/sdk/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf'
        dyn_cfg = rby1_sdk.dynamics.load_robot_from_urdf(urdf_path, 'base')
        self._dyn = rby1_sdk.dynamics.Robot_24(dyn_cfg)
        self._model = rby1_sdk.Model_A()
    def model(self): return self._model
    def get_dynamics(self): return self._dyn
    def get_state(self):
        class State: position = np.zeros(24)
        return State()

robot = OfflineRobotWrapper("1.2")
dyn = robot.get_dynamics()
model = robot.model()
D2R = np.pi / 180.0

# 1. Load today's dataset
data_path = '/home/rainbow/camera_ws/result/result_step2/dataset_20260903_053554.npz'
data = np.load(data_path)
q_arm_list = data['q_arm']
q_head_list = data['q_head']
marker_list = data['marker']
print(f"Loaded NPZ dataset: {len(q_arm_list)} samples from {data_path}")

# 2. Step 1 Circle Fit Values from today
step1_offsets = {
    'right': {'joint3': -2.0542, 'joint5': -5.0109, 'joint6': -1.0999},
    'left':  {'joint3': -2.3009, 'joint5': -0.6298, 'joint6':  0.1883}
}

cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

print("\n" + "="*70)
print("TEST A: DUAL-ARM UNIFIED OPTIMIZATION")
print("Conditions: Head Fixed [0, 0] (no tilt), Step 1 Locked (J3, J5, J6), Camera CAD Nominal")
print("="*70)

opt_unified = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    active_arms=["right", "left"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=head_cfg["head_idx"],
    use_head_kinematics=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=step1_offsets,
    max_iter=50,
)
q_arm_unified, _, _, _, _ = opt_unified.optimize(q_arm_list, q_head_list, marker_list)
r_unified = np.rad2deg(q_arm_unified[:7])
l_unified = np.rad2deg(q_arm_unified[7:])

print("\n--- RESULTS OF TEST A (Unified) ---")
print(f"  Right Arm J0..J6 (deg): {np.round(r_unified, 4).tolist()}")
print(f"  Left  Arm J0..J6 (deg): {np.round(l_unified, 4).tolist()}")
print(f"  Head Yaw / Pitch (deg): [0.0, 0.0] (Fixed to CAD nominal)")

print("\n" + "="*70)
print("TEST B: 3-STAGE PIPELINE (Right -> Left -> Dual) WITH STEP 1 LOCKED & HEAD FIXED")
print("Conditions: Stage 1 Right Arm (Head fixed), Stage 2 Left Arm (Head fixed), Stage 3 Dual Fine")
print("="*70)

# Stage 1: Right Arm with Step 1 locked & Head fixed
cfg_r = get_arm_config(model, "right", version="1.2")
opt_r = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_r["arm_idx"],
    ee_links={"right": cfg_r["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom={"right": cfg["ee_to_marker_nom"]["right"]},
    active_arms=["right"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=head_cfg["head_idx"],
    use_head_kinematics=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=step1_offsets,
    max_iter=50,
)
qr_b, _, _, _, _ = opt_r.optimize(q_arm_list[:, :7], q_head_list, marker_list[:, 0])

# Stage 2: Left Arm with Step 1 locked & Head fixed
cfg_l = get_arm_config(model, "left", version="1.2")
opt_l = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_l["arm_idx"],
    ee_links={"left": cfg_l["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom={"left": cfg["ee_to_marker_nom"]["left"]},
    active_arms=["left"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=head_cfg["head_idx"],
    use_head_kinematics=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=step1_offsets,
    max_iter=50,
)
ql_b, _, _, _, _ = opt_l.optimize(q_arm_list[:, 7:], q_head_list, marker_list[:, 1])

# Stage 3: Dual-Arm Fine
opt_3b = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    active_arms=["right", "left"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=head_cfg["head_idx"],
    use_head_kinematics=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=step1_offsets,
    max_iter=50,
)
q_init_b = np.concatenate([qr_b, ql_b])
q_arm_3b, _, _, _, _ = opt_3b.optimize(
    q_arm_list, q_head_list, marker_list,
    q_arm_offset_init=q_init_b,
)
r_3b = np.rad2deg(q_arm_3b[:7])
l_3b = np.rad2deg(q_arm_3b[7:])

print("\n--- RESULTS OF TEST B (3-Stage Head-Fixed) ---")
print(f"  Right Arm J0..J6 (deg): {np.round(r_3b, 4).tolist()}")
print(f"  Left  Arm J0..J6 (deg): {np.round(l_3b, 4).tolist()}")

# 3. Cartesian Symmetry Evaluation at Ready Pose
print("\n" + "="*70)
print("CARTESIAN SYMMETRY EVALUATION AT READY POSE")
print("="*70)

q_torso = np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R
q_right_ready = np.array([-45, -30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
q_left_ready  = np.array([-45,  30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R

def check_sym(r_deg, l_deg, label):
    q_full = np.zeros(24)
    q_full[list(model.torso_idx)] = q_torso
    q_full[list(model.right_arm_idx)] = q_right_ready + r_deg * D2R
    q_full[list(model.left_arm_idx)] = q_left_ready + l_deg * D2R
    
    _, T_r = compute_fk(robot, dyn, q_full, 'ee_right', 'link_torso_5')
    _, T_l = compute_fk(robot, dyn, q_full, 'ee_left', 'link_torso_5')
    
    pr = T_r[:3, 3] * 1000
    pl = T_l[:3, 3] * 1000
    sym_x = abs(pr[0] - pl[0])
    sym_y = abs(pr[1] + pl[1])
    sym_z = abs(pr[2] - pl[2])
    dist3d = np.sqrt(sym_x**2 + sym_y**2 + sym_z**2)
    print(f"\n[{label}]")
    print(f"  Right EE (mm): [{pr[0]:.2f}, {pr[1]:.2f}, {pr[2]:.2f}]")
    print(f"  Left  EE (mm): [{pl[0]:.2f}, {pl[1]:.2f}, {pl[2]:.2f}]")
    print(f"  Symmetry Error: dX = {sym_x:.2f} mm, dY = {sym_y:.2f} mm, dZ = {sym_z:.2f} mm")
    print(f"  Total 3D Distance Error = {dist3d:.2f} mm")

# A. Baseline
r_base = np.array([ 0.4906,  2.1059,  0.0002,  1.9297, -0.0079,  4.0269,  0.0031])
l_base = np.array([ 0.2630, -1.7961,  0.0013,  3.9805,  0.0004,  0.2017,  0.0000])
check_sym(r_base, l_base, "1. USER BASELINE (Before Today - had bent left elbow)")

# B. Flawed Result from today (Optimized from result_20260903_055911.json)
r_flawed = np.array([-0.1361,  1.1309,  0.4924,  2.0042,  0.4608,  4.9609,  1.1499])
l_flawed = np.array([-1.6034, -0.1501, -0.7728,  2.2509, -0.9682,  0.6065, -0.1936])
check_sym(r_flawed, l_flawed, "2. FLAWED STEP 2 (Today's run with Head Tilt & -1.6° J0)")

# C. Proposed Test A (Unified)
check_sym(r_unified, l_unified, "3. PROPOSED TEST A (Unified Dual-Arm + Step 1 Lock + Head Fixed)")

# D. Proposed Test B (3-Stage)
check_sym(r_3b, l_3b, "4. PROPOSED TEST B (3-Stage + Step 1 Lock + Head Fixed)")
