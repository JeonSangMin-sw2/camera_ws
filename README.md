
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
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 2114, in run
    if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
RuntimeError: Axis 4 marker sweep failed on right arm
