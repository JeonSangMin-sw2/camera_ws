import os
import sys
import time
import yaml
import numpy as np

# Ensure camera_ws and core are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

import rby1_sdk as rby
from core.calibration.Calibrator import HeadCameraCalibrator
from main_ui import SimulatedMarkerTransform

def run_offline_test():
    print("=== [TEST] Step 1.5 HeadCameraCalibrator Simulation Test ===")
    
    robot = rby.create_robot("127.0.0.1:50051", "m")
    if not robot.connect(1):
        print("[ERROR] Cannot connect to simulator at 127.0.0.1:50051")
        return False
    print("[SUCCESS] Connected to RBY1 Simulator.")

    # Power, Servo, Control Manager
    try:
        if not robot.is_power_on(".*"):
            robot.power_on(".*")
            time.sleep(0.5)
    except Exception as e:
        print("[WARN] Power config:", e)

    try:
        cm_state = robot.get_control_manager_state()
        if cm_state.state in [rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault]:
            robot.reset_fault_control_manager()
            time.sleep(0.5)
    except Exception as e:
        print("[WARN] CM fault reset:", e)

    try:
        robot.servo_on(".*")
        time.sleep(0.5)
        cm_state = robot.get_control_manager_state()
        if cm_state.state != rby.ControlManagerState.State.Enabled:
            robot.enable_control_manager(unlimited_mode_enabled=True)
            time.sleep(1.0)
        print(f"[INFO] Control Manager State: {robot.get_control_manager_state().state}")
    except Exception as e:
        print("[WARN] Servo / CM enable:", e)

    # 2. Setup SimulatedMarkerTransform
    setting_path = os.path.join(BASE_DIR, "config", "setting.yaml")
    with open(setting_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cam_cfg = cfg.get("camera", {})

    marker_st = SimulatedMarkerTransform(robot, cam_cfg, robot_version="1.2")
    
    # 3. Instantiate HeadCameraCalibrator
    calibrator = HeadCameraCalibrator(marker_st=marker_st, robot=robot)
    
    # 4. Move to Ready Pose
    print("\n[STEP 1] Moving to Step 1.5 Ready Pose...")
    ok_ready = calibrator.perform_move_to_ready_pose(arm_side="right", log_callback=print)
    if not ok_ready:
        print("[ERROR] perform_move_to_ready_pose failed!")
        return False

    # 5. Perform Head Sweep
    print("\n[STEP 2] Performing Head Pan & Tilt Sweeps...")
    res = calibrator.perform_head_sweep(
        arm_side="right",
        pan_range_deg=12.0,
        tilt_range_deg=7.5,
        num_steps=11,
        step_delay=0.3,
        log_callback=print
    )

    if not res or not res.get("success"):
        print("[ERROR] perform_head_sweep failed!")
        return False

    print("\n[STEP 3] Sweep results successfully obtained:")
    print("  Calibrated mount_to_cam:", res["calibrated_mount_to_cam"])
    print("  Head Offsets:", res["head_offsets_deg"])
    print("  Quality:", res["quality"])

    return True

if __name__ == "__main__":
    success = run_offline_test()
    if success:
        print("\n>>> ALL STEP 1.5 OFFLINE TESTS PASSED! <<<")
    else:
        sys.exit(1)
