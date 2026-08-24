import sys
import os
import threading
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator
from main_ui import FullAutoWorker

class DummySignal:
    def emit(self, *args, **kwargs):
        pass

def run_worker_test(robot_ver="1.3"):
    print(f"\n=======================================================")
    print(f"   TESTING FULL AUTO WORKER: v{robot_ver}")
    print(f"=======================================================\n")
    
    jc = JointCalibrator(None, None)
    mc = MarkerCalibrator(None, None)
    jc.robot_version = robot_ver
    mc.robot_version = robot_ver
    
    joint_offsets_store = {
        "right": {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0},
        "left":  {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0}
    }
    
    stop_event = threading.Event()
    worker = FullAutoWorker(
        jc, mc,
        stop_event=stop_event,
        joint_offsets_store=joint_offsets_store,
        save_debug=False
    )
    
    worker.log_msg = DummySignal()
    worker.log_msg.emit = lambda msg: print(msg)
    worker.status_signal = DummySignal()
    worker.joint_finished_signal = DummySignal()
    worker.bracket_finished_signal = DummySignal()
    worker.finished_signal = DummySignal()
    
    # Run the worker synchronously
    worker.run()
    
    print(f"\n=======================================================")
    print(f"   FINAL STORED OFFSETS (v{robot_ver}):")
    gt_right_j5 = BaseCalibrator.MOCK_GT_OFFSETS["right"]['joint5_v13' if robot_ver == '1.3' else 'joint5_v12']
    gt_left_j5 = BaseCalibrator.MOCK_GT_OFFSETS["left"]['joint5_v13' if robot_ver == '1.3' else 'joint5_v12']
    print(f"   RIGHT ARM: J6={joint_offsets_store['right']['joint6']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS['right']['joint6']:.2f}°), J5={joint_offsets_store['right']['joint5']:.4f}° (GT: {gt_right_j5:.2f}°), J3={joint_offsets_store['right']['joint3']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS['right']['joint3']:.2f}°)")
    print(f"   LEFT ARM : J6={joint_offsets_store['left']['joint6']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS['left']['joint6']:.2f}°), J5={joint_offsets_store['left']['joint5']:.4f}° (GT: {gt_left_j5:.2f}°), J3={joint_offsets_store['left']['joint3']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS['left']['joint3']:.2f}°)")
    print(f"=======================================================\n")

if __name__ == '__main__':
    run_worker_test("1.3")
    run_worker_test("1.2")
