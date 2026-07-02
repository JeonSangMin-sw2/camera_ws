import sys
import os
import numpy as np

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize QApplication to support Qt Signals/Slots without GUI windows
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if app is None:
    app = QApplication([])

from core.calibration.mock_robot import get_mock_robot, pure_mock_compute_fk_impl
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator
from additional_calib_ui import FullAutoWorker, SimulatedMarkerTransform

def test_full_auto_version(version_num):
    print(f"\n======================================")
    print(f" TESTING FULL AUTO CALIBRATION V{version_num}")
    print(f"======================================\n")
    
    # 1. Initialize mock robot
    model_name = "m" if version_num == "1.3" else "a"
    robot = get_mock_robot(model_name=model_name, robot_version=version_num)
    robot.is_pure_mock = True
    BaseCalibrator.compute_fk = staticmethod(pure_mock_compute_fk_impl)
    
    # Set default camera_config
    ver_key = "1.3" if version_num == "1.3" else "1.2"
    nominal_r = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[ver_key]["right"]
    nominal_l = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[ver_key]["left"]
    
    camera_config = {
        "camera_matrix": np.eye(3).tolist(),
        "dist_coeff": np.zeros(5).tolist(),
        "Tf_to_marker_right": list(nominal_r),
        "Tf_to_marker_left": list(nominal_l)
    }
    
    # Use actual SimulatedMarkerTransform from code
    marker_st = SimulatedMarkerTransform(robot, camera_config)
    
    # Staged joint offsets store
    joint_offsets_store = {
        "right": {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0},
        "left": {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0}
    }
    
    # Initialize calibrators
    marker_calibrator = MarkerCalibrator(marker_st, robot)
    joint_calibrator = JointCalibrator(marker_st, robot)
    
    marker_calibrator.camera_config = camera_config
    joint_calibrator.camera_config = camera_config
    
    # Correctly initialize nested joint_offsets dictionary for arms
    joint_offsets = {
        "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
        "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
    }
    marker_calibrator.joint_offsets = joint_offsets
    joint_calibrator.joint_offsets = joint_offsets
    robot.joint_offsets = joint_offsets
    
    marker_calibrator.robot_version = version_num
    joint_calibrator.robot_version = version_num
    
    import threading
    stop_event = threading.Event()
    
    worker = FullAutoWorker(
        joint_calibrator,
        marker_calibrator,
        ui_only=True,
        stop_event=stop_event,
        joint_offsets_store=joint_offsets_store
    )
    
    # Connect signals to print statements to see what happens
    worker.log_msg.connect(lambda msg: print(msg))
    
    # Run the worker directly
    worker.run()
    
    print("\nCalibration results:")
    for arm in ["right", "left"]:
        print(f"--- {arm.upper()} ARM ---")
        print(f"Joint offsets store: {joint_offsets_store[arm]}")
        
        # Verify that J6, J5, J3 offsets are computed and are non-zero (since Mock GT injected offsets are non-zero)
        j6 = joint_offsets_store[arm]["joint6"]
        j5 = joint_offsets_store[arm]["joint5"]
        j3 = joint_offsets_store[arm]["joint3"]
        assert j6 != 0.0, f"Joint 6 offset is zero for {arm} arm!"
        assert j5 != 0.0, f"Joint 5 offset is zero for {arm} arm!"
        assert j3 != 0.0, f"Joint 3 offset is zero for {arm} arm!"
        print(f"J6: {j6:.4f}°, J5: {j5:.4f}°, J3: {j3:.4f}°")

if __name__ == "__main__":
    print("==================================================")
    print("   RUNNING VERSION 1.2 INTEGRATION TEST")
    print("==================================================")
    test_full_auto_version("1.2")
    
    print("\n==================================================")
    print("   RUNNING VERSION 1.3 INTEGRATION TEST")
    print("==================================================")
    test_full_auto_version("1.3")
    print("\nALL TESTS PASSED SUCCESSFULLY!")
