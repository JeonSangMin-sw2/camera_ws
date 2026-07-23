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
        print(f"URDF path {urdf_path} does not exist!")
        return None
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def run_eval():
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

    # Test 1: 3-Stage Optimizer with enforce_joint_offset_limits=False, max_iter=100
    opt1 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-4,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=False,
    )
    q_arm_off, q_head_off, xi_cam, _, _ = opt1.optimize(q_arm_list, q_head_list, T_meas_list)

    opt2 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        max_iter=100,
        eps=1e-4,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=False,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        enforce_joint_offset_limits=False,
    )
    q_arm_off, q_head_off, _, _, _ = opt2.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    opt3 = QPCalibrationOptimizer(
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
        enforce_joint_offset_limits=False,
    )
    q_arm_final, q_head_final, xi_cam_final, _, _ = opt3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    r1_deg = np.rad2deg(q_arm_final[:7])
    l1_deg = np.rad2deg(q_arm_final[7:14])

    print("\n" + "="*80)
    print(" RESULTS (Unconstrained / No Bounds Clipping):")
    print("="*80)
    print("RIGHT ARM:")
    for j in range(7):
        diff = abs(r1_deg[j] - gt_right[j])
        print(f"  J{j} : Ground-Truth = {gt_right[j]:+7.4f}° | Calculated = {r1_deg[j]:+7.4f}° | Diff = {diff:6.4f}° {'[OK <= 0.1deg]' if diff <= 0.1 else '[MISMATCH!]'}")

    print("\nLEFT ARM:")
    for j in range(7):
        diff = abs(l1_deg[j] - gt_left[j])
        print(f"  J{j} : Ground-Truth = {gt_left[j]:+7.4f}° | Calculated = {l1_deg[j]:+7.4f}° | Diff = {diff:6.4f}° {'[OK <= 0.1deg]' if diff <= 0.1 else '[MISMATCH!]'}")
    print("="*80)

if __name__ == "__main__":
    run_eval()
