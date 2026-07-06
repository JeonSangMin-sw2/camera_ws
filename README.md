[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for right arm...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[ERROR] Marker is not visible in ready pose.
[ERROR] Full Auto sequential calibration failed: Axis 4 marker sweep failed on right arm
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/additional_calib_ui.py", line 851, in run
    if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
RuntimeError: Axis 4 marker sweep failed on right arm

[INFO] Full Auto sequential calibration ended.
