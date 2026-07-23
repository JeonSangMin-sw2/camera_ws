import os
import sys
import json
import numpy as np

sys.path.append("/home/rainbow/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer

class OfflineRobot:
    def __init__(self, dyn_robot):
        self.dyn_robot = dyn_robot
        self._model = self._create_model_meta()
        
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self.dyn_robot
        
    def get_state(self):
        class State:
            def __init__(self, dof):
                self.position = np.zeros(dof)
        return State(self.dyn_robot.get_dof())
        
    def _create_model_meta(self):
        class ModelMeta:
            def __init__(self, dyn_robot):
                self.robot_joint_names = dyn_robot.get_joint_names()
                self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
                self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
                self.head_idx = [self.robot_joint_names.index(f"head_{i}") for i in range(2)] if "head_0" in self.robot_joint_names else []
                self.torso_idx = [self.robot_joint_names.index(f"torso_{i}") for i in range(6)] if "torso_0" in self.robot_joint_names else []
        return ModelMeta(self.dyn_robot)

def load_offline_robot():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        return None
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def diagnose():
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260722_175935.npz"
    baseline_path = "/home/rainbow/camera_ws/config/home_reset_baseline.json"
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    gt_right = np.array(baseline["right_arm_joint_offset_deg"])
    gt_left = np.array(baseline["left_arm_joint_offset_deg"])

    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    robot = load_offline_robot()
    if robot is None:
        return

    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())

    print("=========================================================================")
    print(" BASELINE GROUND TRUTH (home_reset_baseline.json):")
    print(" RIGHT ARM:", np.round(gt_right, 4))
    print(" LEFT ARM :", np.round(gt_left, 4))
    print("=========================================================================")

    # Case A: Camera FIXED (optimize_camera=False, optimize_head=False)
    opt_fixed_cam = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-6,
        optimize_head=False,
        optimize_camera=False,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=False,
    )
    q_arm_fixed, _, _, _, _ = opt_fixed_cam.optimize(q_arm_list, q_head_list, T_meas_list)
    r_fixed = np.rad2deg(q_arm_fixed[:7])
    l_fixed = np.rad2deg(q_arm_fixed[7:14])

    print("\n--- CASE A: CAMERA FIXED (Camera Extrinsics Locked to Nominal) ---")
    print("RIGHT ARM ESTIMATED:", np.round(r_fixed, 4))
    print("RIGHT ARM DIFF     :", np.round(np.abs(r_fixed - gt_right), 4))
    print("LEFT ARM ESTIMATED :", np.round(l_fixed, 4))
    print("LEFT ARM DIFF      :", np.round(np.abs(l_fixed - gt_left), 4))

    # Case B: Standard 3-stage with camera optimized but enforce_joint_offset_limits=True (Current Code in main_ui.py)
    opt_ui_1 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=True, # Current main_ui behavior
    )
    q_arm_off, q_head_off, xi_cam, _, _ = opt_ui_1.optimize(q_arm_list, q_head_list, T_meas_list)

    opt_ui_2 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=False,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=True,
    )
    q_arm_off, q_head_off, _, _, _ = opt_ui_2.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    opt_ui_3 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=True,
    )
    q_arm_ui, q_head_ui, xi_cam_ui, _, _ = opt_ui_3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    r_ui = np.rad2deg(q_arm_ui[:7])
    l_ui = np.rad2deg(q_arm_ui[7:14])

    print("\n--- CASE B: CURRENT MAIN_UI CODE (enforce_joint_offset_limits=True) ---")
    print("RIGHT ARM ESTIMATED:", np.round(r_ui, 4))
    print("RIGHT ARM DIFF     :", np.round(np.abs(r_ui - gt_right), 4))
    print("LEFT ARM ESTIMATED :", np.round(l_ui, 4))
    print("LEFT ARM DIFF      :", np.round(np.abs(l_ui - gt_left), 4))

if __name__ == "__main__":
    diagnose()
