import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk
from core.calibration.MarkerCalibrator import MarkerCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from additional_calib_ui import SimulatedMarkerTransform, FullAutoWorker

class DummySignal:
    def emit(self, *args, **kwargs):
        pass

def run_simulation():
    print("==================================================")
    print("   RUNNING V1.2 CALIBRATION SIMULATION")
    print("==================================================")
    
    # 1. Instantiate simulated robot
    robot = rby1_sdk.create_robot("127.0.0.1", "m")
    if not robot.connect():
        raise ConnectionError("Failed to connect to simulated robot at 127.0.0.1")
    
    # 2. Instantiate calibrators
    marker_cal = MarkerCalibrator(None, robot)
    joint_cal = JointCalibrator(None, robot)
    
    # Configure version 1.2
    marker_cal.robot_version = "1.2"
    joint_cal.robot_version = "1.2"
    
    # Initialize camera config from setting.yaml
    marker_cal.load_camera_config()
    joint_cal.load_camera_config()
    
    # 3. Setup simulated marker transform
    marker_st = SimulatedMarkerTransform(robot, marker_cal.camera_config)
    marker_cal.marker_st = marker_st
    joint_cal.marker_st = marker_st
    
    # 4. Initialize joint offsets store
    joint_offsets_store = {
        "right": {"joint6": 0.0, "joint5": 0.0, "joint3": 0.0},
        "left": {"joint6": 0.0, "joint5": 0.0, "joint3": 0.0}
    }
    
    # Initialize joint offsets in calibrators
    joint_offsets = {
        "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
        "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
    }
    joint_cal.joint_offsets = joint_offsets
    marker_cal.joint_offsets = joint_offsets
    
    # 5. Create stop event
    import threading
    stop_event = threading.Event()
    
    # 6. Instantiate worker
    worker = FullAutoWorker(
        joint_cal,
        marker_cal,
        ui_only=True,
        stop_event=stop_event,
        joint_offsets_store=joint_offsets_store
    )
    
    # 7. Mock worker signals to print logs directly to stdout
    def log_print(msg):
        print(msg)
        
    worker.log_msg = DummySignal()
    worker.log_msg.emit = log_print
    
    worker.status_signal = DummySignal()
    worker.bracket_finished_signal = DummySignal()
    worker.joint_finished_signal = DummySignal()
    worker.finished_signal = DummySignal()
    
    # 8. Run the calibration sequence
    print("[MOCK GT] Centralized Ground-Truth Offsets:")
    for arm in ["right", "left"]:
        mock_gt = joint_cal.MOCK_GT_OFFSETS[arm]
        j6_gt = mock_gt["joint6"]
        j5_gt = mock_gt["joint5_v12"]
        j3_gt = mock_gt["joint3"]
        pos_gt = [x * 1000.0 for x in mock_gt["bracket_pos"]]
        rpy_gt = mock_gt["bracket_rpy"]
        print(f"  --- {arm.upper()} ARM ---")
        print(f"  * Bracket Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
        print(f"  * Bracket Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")
        print(f"  * Joint Offsets: Joint 6: {j6_gt:+.2f}°, Joint 5: {j5_gt:+.2f}°, Joint 3: {j3_gt:+.2f}°")
    
    worker.run()

if __name__ == "__main__":
    run_simulation()
