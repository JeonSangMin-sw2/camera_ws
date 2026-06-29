import os
import sys
import numpy as np

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

from scipy.spatial.transform import Rotation as R_scipy
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.mock_robot import PureMockRobot, pure_mock_compute_fk_impl

def main():
    file_A = "/home/rainbow/Downloads/camera_ws-main (1)/camera_ws-main/sweep_points_right_joint_A_axis_6.txt"
    poses_A = []
    angles_A = []
    with open(file_A) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = [float(x) for x in line.strip().split(",")]
            angles_A.append(parts[0])
            poses_A.append(np.array(parts[26:42]).reshape(4, 4))
            
    res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
        poses_A, angles_A, np.array([0.0, 0.0, 1.0])
    )
    c_A = res_A['c_opt']
    n_A = res_A['axis_opt'] / np.linalg.norm(res_A['axis_opt'])
    
    _ref = np.array([0.0, 0.0, 1.0]) if abs(n_A[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    _ex6 = np.cross(n_A, _ref); _ex6 /= np.linalg.norm(_ex6)
    _ey6 = np.cross(n_A, _ex6)
    
    def _to2d_j6(p):
        v = p - c_A
        vi = v - np.dot(v, n_A) * n_A
        return float(np.dot(vi, _ex6)), float(np.dot(vi, _ey6))
        
    pts_a_xyz = np.array([T[:3, 3] * 1000.0 for T in poses_A])
    angles_A_arr = np.array(angles_A)
    
    j6_plane_angles_rad = np.array([
        np.arctan2(*_to2d_j6(p)[::-1]) for p in pts_a_xyz
    ])
    j6_plane_angles_rad = np.unwrap(j6_plane_angles_rad)
    j6_plane_angles_deg = np.degrees(j6_plane_angles_rad)
    j6_coeffs = np.polyfit(angles_A_arr, j6_plane_angles_deg, 1)
    angle_at_j6_zero = j6_coeffs[1]
    
    # Pure mock robot right arm ready pose: J3=-117, other J=0
    arm_ready_deg = [-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0]
    q_ready = np.zeros(20)
    # PureMockRobot arm indices: range(7)
    for i, val in enumerate(arm_ready_deg):
        q_ready[i] = np.radians(val)
        
    robot = PureMockRobot(arm_side="right", model_name="m")
    
    # Solved marker calibration values from the file:
    tf_vec_solved = [0.1020437, -0.0019079, -0.0071342, 92.3037863, -2.3570705, -91.05938839]
    T_ee_to_marker_solved = BaseCalibrator.make_transform(tf_vec_solved)
    
    # FK nominal
    T_t5_to_ee_nom = pure_mock_compute_fk_impl(robot, None, q_ready, "ee_right", "link_torso_5")
    T_t5_to_marker_nom = T_t5_to_ee_nom @ T_ee_to_marker_solved
    p_marker_nom_t5 = T_t5_to_marker_nom[:3, 3] * 1000.0
    
    x_nom, y_nom = _to2d_j6(p_marker_nom_t5)
    angle_design_zero = np.degrees(np.arctan2(y_nom, x_nom))
    
    j6_offset_deg = angle_at_j6_zero - angle_design_zero
    j6_offset_deg = float((j6_offset_deg + 180.0) % 360.0 - 180.0)
    j6_correction_raw = -j6_offset_deg
    
    print("\n--- Test 5: PureMockRobot with SOLVED marker calibration ---")
    print(f"J6 encoder=0° Ref Angle (calculated) : {angle_at_j6_zero:.4f}°")
    print(f"J6 Design Nominal Angle (calculated)  : {angle_design_zero:.4f}°")
    print(f"★ J6 Calibration Angle                : {j6_correction_raw:+.4f}°")

if __name__ == "__main__":
    main()
