import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.mock_robot import get_mock_robot
from core.calibration.MarkerCalibrator import MarkerCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from additional_calib_ui import SimulatedMarkerTransform

def main():
    robot = get_mock_robot(model_name="a") # v1.2 robot
    marker_cal = MarkerCalibrator(None, robot)
    joint_cal = JointCalibrator(None, robot)
    
    marker_cal.robot_version = "1.2"
    joint_cal.robot_version = "1.2"
    
    marker_cal.load_camera_config()
    joint_cal.load_camera_config()
    
    marker_st = SimulatedMarkerTransform(robot, marker_cal.camera_config)
    marker_cal.marker_st = marker_st
    joint_cal.marker_st = marker_st
    
    # Run a dummy sweep for wrist_pitch (Joint 5) to collect the dataset
    print("Collecting dummy wrist_pitch sweeps...")
    res = joint_cal.perform_calibration_sweep_continuous(
        "right", "wrist_pitch",
        log_callback=print,
        current_offset_deg=0.0,
        sweep_duration=5.0
    )
    
    if not res:
        print("Failed to run sweep")
        return
        
    dataset_A = res['_dataset_A']
    dataset_B = res['_dataset_B']
    initial_joint_pos = res['_initial_joint_pos']
    
    print("\n--- Sensitivity analysis for wrist_pitch (Joint 5) ---")
    for offset in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        calc = joint_cal.compute_calibration_results(
            "right", "wrist_pitch",
            dataset_A, dataset_B,
            initial_joint_pos,
            current_offset_deg=offset,
            log_callback=None
        )
        if calc:
            diff_angle = np.degrees(calc.get('angle_between_normals', 0.0))
            print(f"Staged offset: {offset:+.1f}° | Normals Angle: {diff_angle:.6f}° | center_dist: {calc['center_dist']:.3f} mm | size_error: {abs(calc['r_A'] - calc['r_B']):.3f} mm")
        else:
            print(f"Staged offset: {offset:+.1f}° | Calculation failed")

if __name__ == "__main__":
    main()
