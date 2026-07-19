import sys
import os
import numpy as np
import rby1_sdk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.MarkerCalibrator import MarkerCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer

def main():
    print("Connecting to robot at 127.0.0.1:50051...")
    robot = rby1_sdk.create_robot("127.0.0.1:50051", "m")
    if not robot.connect():
        print("[ERROR] Failed to connect to simulated robot!")
        return

    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260720_000929.npz"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset: {dataset_path}")
    q_arm_list, q_head_list, _ = load_npz_dataset(dataset_path)
    
    # Initialize calibrators
    marker_cal = MarkerCalibrator(None, robot)
    marker_cal.robot_version = "1.2"
    marker_cal.load_camera_config()
    
    # Empty joint offsets
    joint_offsets = {
        "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
        "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
    }
    marker_cal.joint_offsets = joint_offsets

    model = robot.model()
    cfg = get_both_arm_config(model, version="1.2")
    head_cfg = get_head_config(model)

    print("Regenerating simulated marker poses with head offsets...")
    new_T_meas_list = []
    for q_arm, q_head in zip(q_arm_list, q_head_list):
        # Reconstruct q_full
        q_full = robot.get_state().position.copy()
        q_full[cfg["arm_idx"]] = q_arm
        if head_cfg["head_idx"] is not None and q_head is not None:
            q_full[head_cfg["head_idx"]] = q_head
            
        T_meas_right = marker_cal.get_simulated_marker_pose("right", q_actual=q_full)
        T_meas_left = marker_cal.get_simulated_marker_pose("left", q_actual=q_full)
        new_T_meas_list.append(np.stack([T_meas_right, T_meas_left], axis=0))

    new_T_meas_list = np.array(new_T_meas_list)

    print("Running QP Optimizer with optimize_head=True...")
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=False,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=False,
    )

    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
        q_arm_list,
        q_head_list,
        new_T_meas_list,
    )

    print("\n" + "="*50)
    print("   OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Ground-Truth Head Offsets (deg): Pan: 0.80, Tilt: -1.50")
    print(f"Calibrated Head Offsets (deg):   Pan: {np.rad2deg(q_head_offset[0]):+.4f}, Tilt: {np.rad2deg(q_head_offset[1]):+.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
