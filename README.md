[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for right arm (Pass 1/2)...
[FULL AUTO] Moving right arm to marker ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for right arm (Pass 1/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: -1.0243°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: -4.0454°
[FULL AUTO] [Phase 1] Orthogonality Residual: 2.490°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -4.0454° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: -1.0243° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for right arm...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: -1.0243°, Joint 6: -4.0454°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.0106°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.2073°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.1292°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -2.1912°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0107° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1912°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1912° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 4.0454°
  * Joint 5 Change      : 1.0243°
  * Joint 3 Change      : 2.1912°
  * Bracket Pos Change  : 30.0022 mm
  * Bracket Rot Change  : 5.2639°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for right arm (Pass 2/2)...
[FULL AUTO] Moving right arm to marker ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for right arm (Pass 2/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: -1.9655°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: -2.0375°
[FULL AUTO] [Phase 1] Orthogonality Residual: 1.021°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -2.0375° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: -1.9655° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for right arm...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: -1.9655°, Joint 6: -2.0375°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-2.1912°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1912° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for left arm (Pass 1/2)...
[FULL AUTO] Moving left arm to marker ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for left arm (Pass 1/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: -0.2359°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: +2.3581°
[FULL AUTO] [Phase 1] Orthogonality Residual: 3.097°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 2.3581° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -0.2359° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for left arm...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: -0.2359°, Joint 6: 2.3581°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8789°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0474° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.8789°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.8789° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 2.3581°
  * Joint 5 Change      : 0.2359°
  * Joint 3 Change      : 1.8789°
  * Bracket Pos Change  : 31.0169 mm
  * Bracket Rot Change  : 1.4457°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for left arm (Pass 2/2)...
[FULL AUTO] Moving left arm to marker ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for left arm (Pass 2/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: -0.5525°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: +1.8129°
[FULL AUTO] [Phase 1] Orthogonality Residual: 3.137°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 1.8129° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -0.5525° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for left arm...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: -0.5525°, Joint 6: 1.8129°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-1.8789°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.8789° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.0, Z: +0.0 mm
  * Bracket Rot: R: -2.01, P: -2.06, Y: +0.61 deg
  * Joint Offsets: Joint 6: -2.04°, Joint 5: -1.97°, Joint 3: -2.19°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.0, Z: +0.0 mm
  * Bracket Rot: R: +0.55, P: -0.32, Y: -0.64 deg
  * Joint Offsets: Joint 6: +1.81°, Joint 5: -0.55°, Joint 3: -1.88°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Roll) : 1.8129°
    * Joint 5 (Wrist Pitch): -0.5525°
    * Joint 3 (Elbow)      : -1.8789°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Roll) : -2.0375°
    * Joint 5 (Wrist Pitch): -1.9655°
    * Joint 3 (Elbow)      : -2.1912°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[WARNING] Auto-centering head encountered a minor issue: list indices must be integers or slices, not tuple
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [0. 0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
