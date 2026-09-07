import numpy as np
import yaml
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_arm_config, make_transform
from core.calibration.CalibratorBase import BaseCalibrator

# Load dataset
data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260907_164411.npz')
q_arm_list = data['q_arm']
T_meas_list = data['marker']

# Mock robot
class MockRobot:
    def __init__(self):
        import rby1_sdk
        self._model = rby1_sdk.create_robot_a()
    def model(self):
        return self._model
    def get_dynamics(self):
        return self._model.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(23)
        return State()

robot = MockRobot()
dyn_model = robot.get_dynamics()

# Arm configuration
cfg_r = get_arm_config(robot.model(), "right", "1.2")
cfg_l = get_arm_config(robot.model(), "left", "1.2")

arm_idx = cfg_r["arm_idx"] + cfg_l["arm_idx"]
ee_links = {"right": cfg_r["ee_link"], "left": cfg_l["ee_link"]}
ee_to_marker_nom = {"right": cfg_r["ee_to_marker_nom"], "left": cfg_l["ee_to_marker_nom"]}

with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
    setting = yaml.safe_load(f)

# Step 1 anchors
jo = setting.get("joint_offset", {})
anchors = {
    "right": {"joint3": jo["right"]["joint3"], "joint5": jo["right"]["joint5"], "joint6": jo["right"]["joint6"]},
    "left": {"joint3": jo["left"]["joint3"], "joint5": jo["left"]["joint5"], "joint6": jo["left"]["joint6"]}
}

mount_to_cam_nom = setting["camera"]["mount_to_cam_nominal"]
head_base_to_cam_nom = setting["camera"]["head_base_to_cam_nominal"]

# GT offsets
gt_r = [0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3]
gt_l = [-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5]

print("=" * 60)
print("EXPERIMENT: Headless Mode (no head joints)")
print("=" * 60)

for opt_cam in [False, True]:
    print(f"\n--- Testing with optimize_camera = {opt_cam} ---")
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=mount_to_cam_nom,
        head_base_to_cam_nom=head_base_to_cam_nom,
        ee_to_marker_nom=ee_to_marker_nom,
        active_arms=["right", "left"],
        optimize_arm=True,
        optimize_head=False,
        optimize_camera=opt_cam,
        head_idx=None,
        use_head_kinematics=False,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1.0,
        use_sag=False,
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=anchors,
        camera_pos_bound_m=0.010,
        camera_rot_bound_rad=np.deg2rad(3.0),
        eps=1e-7,
        max_iter=30,
    )
    
    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
        q_arm_list, None, T_meas_list
    )
    
    r_calc = np.rad2deg(q_arm_offset[:7])
    l_calc = np.rad2deg(q_arm_offset[7:])
    
    print("Right Arm J0-J6 errors:")
    r_err = np.abs(r_calc - gt_r)
    print("  Calc:", np.round(r_calc, 3))
    print("  GT:  ", gt_r)
    print("  Err: ", np.round(r_err, 4), "Max:", np.round(np.max(r_err), 4))
    
    print("Left Arm J0-J6 errors:")
    l_err = np.abs(l_calc - gt_l)
    print("  Calc:", np.round(l_calc, 3))
    print("  GT:  ", gt_l)
    print("  Err: ", np.round(l_err, 4), "Max:", np.round(np.max(l_err), 4))
    
    if opt_cam:
        print("Calibrated head_base_to_cam:", head_base_to_cam_new)
        print("Nominal head_base_to_cam:   ", head_base_to_cam_nom)
