
[ITERATION 6/6] Sweeping physically with staged offset 2.1302°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0403° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 2.1302°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 2.1302° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 right arm (Pass 1/2)...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[ERROR] Failed to move Marker Axis 4 to start pose or stop requested.
[ERROR] Full Auto sequential calibration failed: Axis 4 marker sweep failed on right arm



ration=12.0s)...
2026-08-05 16:24:46,070 [INFO]     -> Swept 172 dense raw coordinate frames during Joint B motion.
2026-08-05 16:24:46,071 [INFO] Swept 182 dense raw coordinate frames during Joint A motion... downsampled to 182 for optimization.
2026-08-05 16:24:46,072 [INFO] Swept 172 dense raw coordinate frames during Joint B motion... downsampled to 172 for optimization.
[READY POSE] Head disabled (Fixed Chest Camera): Joint 0 pitch lowered by +20° (-35.0°) for right_arm (joint)
2026-08-05 16:24:46,076 [INFO] [READY POSE] Head disabled (Fixed Chest Camera): Joint 0 pitch lowered by +20° (-35.0°) for right_arm (joint)
[READY POSE] Head disabled (Fixed Chest Camera): Joint 0 pitch lowered by +20° (-70.0°) for right_arm (marker)
2026-08-05 16:24:53,091 [INFO] [READY POSE] Head disabled (Fixed Chest Camera): Joint 0 pitch lowered by +20° (-70.0°) for right_arm (marker)
2026-08-05 16:25:00,483 [INFO] [INFO] Moving Marker Axis 4 to start sweep position...
[DEBUG MOVEJ ERROR] Failed to conduct movej. Finish code: FinishCode.Unknown
2026-08-05 16:25:00,485 [ERROR] Failed to conduct movej. Finish code: FinishCode.Unknown

Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 2114, in run
    if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
RuntimeError: Axis 4 marker sweep failed on right arm
