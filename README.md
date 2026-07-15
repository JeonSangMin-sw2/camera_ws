[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
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

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-9.79°, raw_diff=9.75°, optimal_offset=-0.04°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.0943 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.4138 mm
  * Estimated Circle Center Distance       : 47.6457 mm
  * Calculated Offset Correction           : -0.041345 deg
==================================================
[FULL AUTO] Staging J6 offset: -0.0413°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0413° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -8.4045°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -8.3544°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -8.2595°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0269° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -8.2595°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -8.2595° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.6752°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.7441°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.2999°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.9313°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -2.2067°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset -3.5523° exceeds safe bounds [-3.0°, 0.0°]. Clamping.

[ITERATION 6/6] Sweeping physically with staged offset -3.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.8194° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.2489°
  * Joint 5 Change      : 1.2040°
  * Joint 3 Change      : 0.1442°
  * Bracket Pos Change  : 0.2952 mm
  * Bracket Rot Change  : 0.3795°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[INFO] Pass 2: Reusing marker sweep datasets from Pass 1.

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-9.79°, raw_diff=9.75°, optimal_offset=-0.04°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.0943 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.4138 mm
  * Estimated Circle Center Distance       : 47.6457 mm
  * Calculated Offset Correction           : -0.041345 deg
==================================================
[FULL AUTO] Staging J6 offset: -0.0413°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0413° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -8.2595°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -9.3408°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0145° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -9.3408°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -9.3408° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.8194°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.5323°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0347°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.2166°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -1.0511°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 6/6] Sweeping physically with staged offset -1.6550°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[STOP] Stopping Full Auto Calibration...
[INFO] Control cancelled safely via temporary connection.
[ERROR] Failed to move Joint B to start pose or stop requested.
[ERROR] Iteration 6 sweep failed. Aborting calibration.
[ERROR] Full Auto sequential calibration failed: Elbow joint calibration failed on right arm
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 1926, in run
    raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
RuntimeError: Elbow joint calibration failed on right arm

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.2238°
    * Joint 5 (Wrist Pitch): -5.9148°
    * Joint 3 (Elbow)      : -2.8683°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.0413°
    * Joint 5 (Wrist Pitch): -9.3408°
    * Joint 3 (Elbow)      : -1.8194°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[Step2] Init Pose requested.
