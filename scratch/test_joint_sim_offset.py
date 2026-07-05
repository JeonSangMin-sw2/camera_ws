import os
import sys
import numpy as np

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

from core.calibration.mock_robot import get_mock_robot
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator

def main():
    print("Running full Joint 6 Roll Calibration loop...")
    
    # Initialize mock robot and calibrator
    robot = get_mock_robot(model_name="m")
    robot.robot_version = "1.2"
    
    cal = JointCalibrator(marker_st=None, robot=robot)
    cal.robot_version = "1.2"
    
    # Call perform_joint_calibration with log printout
    res = cal.perform_joint_calibration(
        "right", "elbow",
        log_callback=print,
        sweep_duration=15.0
    )
    
    print("\nFinal calibration output:")
    print(res)

if __name__ == "__main__":
    main()
