import numpy as np
from core.calibration_core import make_transform, create_robot, load_npz_dataset, get_arm_config, get_head_config
from core.calibration_optimizer import rot_to_euler_zyx, QPCalibrationOptimizer

# Load data
npz = "result/dataset_20260507_155725.npz"
q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz)
q_arm_list = q_arm_list[:, :7]
T_meas_list = T_meas_list[:, 0]

# Setup
robot = create_robot("127.0.0.1:50051")
model = robot.model()
cfg = get_arm_config(model, "right")
head_cfg = get_head_config(model)

# Apply 180 Roll fix to Tf_to_marker
ee_nom = list(cfg["ee_to_marker_nom"])
ee_nom[3] += 180.0

optimizer = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links={"right": cfg["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    t5_to_cam_nom=cfg.get("t5_to_cam_nom"),
    ee_to_marker_nom={"right": ee_nom},
    ndof=7,
    head_idx=head_cfg["head_idx"],
    optimize_head=True,
    active_arms=["right"]
)

q_arm = q_arm_list[0]
q_head = q_head_list[0]
T_meas = T_meas_list[0]

_, _, _, T_model = optimizer.evaluate_sample(
    q_arm=q_arm,
    q_head=q_head,
    arm_side="right",
    q_arm_offset=np.zeros(7),
    q_head_offset=np.zeros(2),
    xi_mount_cam=np.zeros(6)
)

diff_trans = T_model[:3, 3] - T_meas[:3, 3]
R_err = T_model[:3, :3].T @ T_meas[:3, :3]
diff_euler = np.rad2deg(rot_to_euler_zyx(R_err))

print(f"Diff Pos: {diff_trans}")
print(f"Diff Rot: {diff_euler}")
