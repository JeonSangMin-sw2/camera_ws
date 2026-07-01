import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.mock_robot import PureMockRobot
from core.calibration.JointCalibrator import JointCalibrator

def load_dataset_and_mirror(filepath, arm_idx, sweep_joint_idx, mirror=False):
    dataset = []
    # R_mirror mirrors across Y plane
    # T_mirror: X is same, Y is negated, Z is same
    # In camera frame:
    # Camera X points right. Mirroring left/right means camera X is negated!
    # Camera Y is down, Z is forward.
    # So if we mirror left/right: camera X coordinate is negated.
    # Also, camera rotation: Y and Z axes of rotation might flip.
    # Let's see: if we just negate the X coordinate of the pose translation, and mirror the rotation matrix.
    # Let's do proper mirroring of the 4x4 pose:
    # T_cam2marker = [R, p]
    # Mirrored T:
    # px_mirrored = -px
    # py_mirrored = py
    # pz_mirrored = pz
    # For rotation: mirror across X=0 plane in camera frame.
    # M = diag(-1, 1, 1)
    # R_mirrored = M @ R @ M
    M = np.diag([-1, 1, 1])
    
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(x.strip()) for x in line.strip().split(",")]
            angle = parts[0]
            pose = np.array(parts[10:26]).reshape(4, 4)
            
            if mirror:
                # Mirror the pose
                pose[:3, 3] = M @ pose[:3, 3]
                pose[:3, :3] = M @ pose[:3, :3] @ M
                
            q_full = np.zeros(20)
            q_full[arm_idx[sweep_joint_idx]] = np.radians(angle)
            dataset.append((q_full, pose))
    return dataset

def main():
    # 1. Run for RIGHT arm (no mirroring)
    robot_r = PureMockRobot(arm_side="right", model_name="v12")
    calib_r = JointCalibrator(marker_st=None, robot=robot_r)
    arm_idx_r = robot_r.model().right_arm_idx
    
    path_A = "/home/rainbow/camera_ws/sweep_points_right_joint_A_axis_4 (2).txt"
    path_B = "/home/rainbow/camera_ws/sweep_points_right_joint_B_axis_6 (2).txt"
    
    dataset_A_r = load_dataset_and_mirror(path_A, arm_idx_r, 4, mirror=False)
    dataset_B_r = load_dataset_and_mirror(path_B, arm_idx_r, 6, mirror=False)
    
    res_r = calib_r.compute_calibration_results(
        arm_side="right",
        mode="wrist_pitch",
        dataset_A=dataset_A_r,
        dataset_B=dataset_B_r,
        initial_joint_pos=[0.0]*7,
        current_offset_deg=0.0,
        use_angle_based_fitting=True,
        save_debug=False
    )
    print("RIGHT arm offset:", res_r["optimal_offset"])
    
    # 2. Run for LEFT arm (mirrored dataset)
    robot_l = PureMockRobot(arm_side="left", model_name="v12")
    calib_l = JointCalibrator(marker_st=None, robot=robot_l)
    arm_idx_l = robot_l.model().left_arm_idx
    
    dataset_A_l = load_dataset_and_mirror(path_A, arm_idx_l, 4, mirror=True)
    dataset_B_l = load_dataset_and_mirror(path_B, arm_idx_l, 6, mirror=True)
    
    res_l = calib_l.compute_calibration_results(
        arm_side="left",
        mode="wrist_pitch",
        dataset_A=dataset_A_l,
        dataset_B=dataset_B_l,
        initial_joint_pos=[0.0]*7,
        current_offset_deg=0.0,
        use_angle_based_fitting=True,
        save_debug=False
    )
    print("LEFT arm offset:", res_l["optimal_offset"])

if __name__ == "__main__":
    main()
