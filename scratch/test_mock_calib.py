import sys
import os
import time
sys.path.append("/home/rainbow/camera_ws")
from core.calibration.JointCalibrator import JointCalibrator

def log_cb(msg):
    print(msg)

def stat_cb(msg):
    print("STATUS:", msg)

calib = JointCalibrator(marker_st=None, robot=None)
calib.robot_version = "1.2"
calib.load_camera_config()
calib.load_ready_poses()

res = calib.perform_joint_calibration(
    arm_side="right",
    mode="elbow",
    log_callback=log_cb,
    status_callback=stat_cb,
    current_offset_deg=0.0,
    sweep_duration=2.0,
    use_angle_based_fitting=True,
    save_debug=False
)
print("FINAL RES:", res)
