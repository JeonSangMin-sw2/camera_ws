import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.mock_robot import PureMockRobot
from core.calibration.JointCalibrator import JointCalibrator

def load_dataset(filepath, arm_idx, sweep_joint_idx):
    dataset = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(x.strip()) for x in line.strip().split(",")]
            angle = parts[0]
            # T_cam2marker_flat is in parts[10:26]
            pose = np.array(parts[10:26]).reshape(4, 4)
            
            q_full = np.zeros(20)
            # set the swept joint position in radians
            q_full[arm_idx[sweep_joint_idx]] = np.radians(angle)
            dataset.append((q_full, pose))
    return dataset

def main():
    robot = PureMockRobot(arm_side="right", model_name="v12")
    
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    calibrator.camera_config = {
        "mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]
    }
    
    arm_idx = robot.model().right_arm_idx
    
    path_A = "/home/rainbow/camera_ws/sweep_points_right_joint_A_axis_4 (1).txt"
    path_B = "/home/rainbow/camera_ws/sweep_points_right_joint_B_axis_6 (1).txt"
    
    dataset_A = load_dataset(path_A, arm_idx, 4)
    dataset_B = load_dataset(path_B, arm_idx, 6)
    
    initial_joint_pos = [0.0] * 7
    
    res = calibrator.compute_calibration_results(
        arm_side="right",
        mode="wrist_pitch",
        dataset_A=dataset_A,
        dataset_B=dataset_B,
        initial_joint_pos=initial_joint_pos,
        current_offset_deg=0.0,
        use_angle_based_fitting=True,
        save_debug=False,
        log_callback=None
    )
    
    print("\n--- Refactored Joint Calibration Results ---")
    if res:
        for k, v in res.items():
            if k.startswith("_") or isinstance(v, (list, np.ndarray)):
                continue
            print(f"  {k}: {v}")
    else:
        print("  Error: compute_calibration_results returned None!")

if __name__ == "__main__":
    main()
