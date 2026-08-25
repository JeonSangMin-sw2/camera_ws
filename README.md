[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 right arm...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -0.2354°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0262° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2354°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2354° (click APPLY OFFSET to save).
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
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.40°, optimal_offset=0.34°

[ITERATION 2/6] Sweeping physically with staged offset 0.3396°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.00°, optimal_offset=0.36°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0197° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.3396°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.3396°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.3396° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.9919°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0940°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0563° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.0940°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.0940° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.3396°
  * Joint 5 Change      : 0.2354°
  * Joint 3 Change      : 2.0940°
  * Bracket Pos Change  : 1.6132 mm
  * Bracket Rot Change  : 0.5124°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.2354°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2354° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 right arm (Pass 2/2)...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (0.3396°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.3396° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.0940°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.0940° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 left arm...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -0.1390°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0012° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1390°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.1390° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 left arm (Pass 1/2)...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.15°, optimal_offset=-0.11°

[ITERATION 2/6] Sweeping physically with staged offset -0.1129°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.25°, optimal_offset=0.09°

[ITERATION 3/6] Sweeping physically with staged offset 0.0730°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.02°, optimal_offset=0.06°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0230° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0730°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0730°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0730° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8438°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0504°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.9819°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0555° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.9819°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9819° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.0730°
  * Joint 5 Change      : 0.1390°
  * Joint 3 Change      : 1.9819°
  * Bracket Pos Change  : 1.3239 mm
  * Bracket Rot Change  : 1.0410°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.1390°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.1390° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 left arm (Pass 2/2)...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (0.0730°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0730° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-1.9819°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9819° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.2, Z: -0.7 mm
  * Bracket Rot: R: +0.55, P: -0.19, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.34°, Joint 5: -0.24°, Joint 3: -2.09°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.3, Z: -0.8 mm
  * Bracket Rot: R: +0.88, P: -0.08, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.07°, Joint 5: -0.14°, Joint 3: -1.98°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0730°
    * Joint 5 (Wrist Pitch): -0.1390°
    * Joint 3 (Elbow)      : -1.9819°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.3396°
    * Joint 5 (Wrist Pitch): -0.2354°
    * Joint 3 (Elbow)      : -2.0940°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [-0.02  0.  ]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 52] Captured marker: R_pos=[ 85.7 -42.2 184.7]mm, L_pos=[-65.1 -41.3 183.5]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 53] Captured marker: R_pos=[ 78.9 -40.1 173.3]mm, L_pos=[-58.1 -39.2 172. ]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 54] Captured marker: R_pos=[ 89.3 -43.5 190.7]mm, L_pos=[-69.  -42.6 189.5]mm
Auto motion done: Joint 0 Offset: 5.0deg
[Sample 55] Captured marker: R_pos=[ 93.2 -44.9 196.7]mm, L_pos=[-72.9 -44.  195.5]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 56] Captured marker: R_pos=[ 85.5 -44.5 190.2]mm, L_pos=[-65.2 -39.1 178.1]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 57] Captured marker: R_pos=[ 84.9 -46.8 195.5]mm, L_pos=[-65.  -36.8 172.6]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 58] Captured marker: R_pos=[ 85.6 -39.9 179.4]mm, L_pos=[-65.  -43.5 189. ]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 59] Captured marker: R_pos=[ 85.5 -37.6 173.9]mm, L_pos=[-64.5 -45.8 194.4]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 60] Captured marker: R_pos=[ 90.4 -43.  181.9]mm, L_pos=[-69.8 -42.4 180.7]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 61] Captured marker: R_pos=[ 95.5 -44.  179.4]mm, L_pos=[-74.9 -43.6 178.1]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 62] Captured marker: R_pos=[ 81.3 -41.3 188.1]mm, L_pos=[-60.9 -40.3 186.8]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 63] Captured marker: R_pos=[ 77.5 -40.6 191.6]mm, L_pos=[-57.  -39.4 190.4]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 64] Captured marker: R_pos=[ 87.5 -47.1 181. ]mm, L_pos=[-63.7 -36.5 187.6]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 65] Captured marker: R_pos=[ 89.4 -52.  177.6]mm, L_pos=[-62.5 -31.7 192. ]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 66] Captured marker: R_pos=[ 84.1 -37.3 188.8]mm, L_pos=[-66.8 -46.2 179.7]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 67] Captured marker: R_pos=[ 82.8 -32.5 192.9]mm, L_pos=[-68.5 -51.1 175.9]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 68] Captured marker: R_pos=[ 82.4 -27.8 181.7]mm, L_pos=[-67.7 -55.5 186.5]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 69] Captured marker: R_pos=[ 89.4 -47.6 167. ]mm, L_pos=[-62.  -36.3 203.2]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 70] Captured marker: R_pos=[ 82.2 -37.1 203.9]mm, L_pos=[-68.5 -46.6 165.4]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 71] Captured marker: R_pos=[ 88.3 -56.5 188.1]mm, L_pos=[-62.2 -27.  180.7]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 72] Captured marker: R_pos=[ 77.1 -34.6 180.5]mm, L_pos=[-56.5 -45.4 201.5]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 73] Captured marker: R_pos=[ 94.7 -47.2 189.6]mm, L_pos=[-74.8 -40.5 167.7]mm
Auto motion done: Restore Baseline Pose
[Sample 74] Captured marker: R_pos=[ 85.6 -42.2 184.8]mm, L_pos=[-65.2 -41.3 183.5]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 75] Captured marker: R_pos=[ 91.7 -43.4 188.7]mm, L_pos=[-66.6 -42.7 191.6]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 76] Captured marker: R_pos=[ 94.1 -43.4 186.7]mm, L_pos=[-64.4 -43.  193.7]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 77] Captured marker: R_pos=[ 87.  -43.5 192.9]mm, L_pos=[-71.3 -42.5 187.5]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 78] Captured marker: R_pos=[ 84.8 -43.5 195.2]mm, L_pos=[-73.7 -42.4 185.6]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 79] Captured marker: R_pos=[ 88.9 -41.4 190.4]mm, L_pos=[-68.6 -40.4 189.3]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 80] Captured marker: R_pos=[ 88.5 -39.3 189.8]mm, L_pos=[-68.2 -38.2 188.9]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 81] Captured marker: R_pos=[ 89.8 -45.5 191.3]mm, L_pos=[-69.6 -44.9 189.8]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 82] Captured marker: R_pos=[ 90.3 -47.5 191.6]mm, L_pos=[-70.1 -47.  190. ]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 83] Captured marker: R_pos=[ 89.3 -45.9 191. ]mm, L_pos=[-69.  -40.3 189.4]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 84] Captured marker: R_pos=[ 89.4 -48.1 191.4]mm, L_pos=[-69.  -37.9 189.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 85] Captured marker: R_pos=[ 89.2 -41.2 190.6]mm, L_pos=[-68.9 -44.9 189.7]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 86] Captured marker: R_pos=[ 89.1 -38.7 190.6]mm, L_pos=[-69.  -47.3 189.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 87] Captured marker: R_pos=[ 92.4 -43.6 195.8]mm, L_pos=[-65.5 -42.7 185.4]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 88] Captured marker: R_pos=[ 95.  -43.9 201.5]mm, L_pos=[-61.9 -42.8 182.3]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 89] Captured marker: R_pos=[ 86.  -43.4 186.8]mm, L_pos=[-72.2 -42.7 194.5]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 90] Captured marker: R_pos=[ 82.5 -43.4 183.7]mm, L_pos=[-75.  -42.7 200.5]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 91] Captured marker: R_pos=[ 89.6 -39.9 186.2]mm, L_pos=[-69.3 -39.  184.9]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 92] Captured marker: R_pos=[ 89.6 -36.4 182.3]mm, L_pos=[-69.8 -35.6 181. ]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 93] Captured marker: R_pos=[ 89.1 -47.1 196.1]mm, L_pos=[-68.4 -46.4 194.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 94] Captured marker: R_pos=[ 88.8 -50.9 202.2]mm, L_pos=[-67.9 -50.1 200.8]mm
Auto motion done: Head Pan: -3.50deg
[Sample 95] Captured marker: R_pos=[ 74.8 -43.4 195.2]mm, L_pos=[-83.  -42.7 184.3]mm
Auto motion done: Head Pan: -1.75deg
[Sample 96] Captured marker: R_pos=[ 82.  -43.5 193.1]mm, L_pos=[-76.1 -42.7 187. ]mm
Auto motion done: Head Pan: +1.75deg
[Sample 97] Captured marker: R_pos=[ 96.3 -43.5 188.2]mm, L_pos=[-61.8 -42.5 191.8]mm
Auto motion done: Head Pan: +3.50deg
[Sample 98] Captured marker: R_pos=[103.4 -43.5 185.4]mm, L_pos=[-54.6 -42.5 193.9]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 99] Captured marker: R_pos=[ 89.3 -28.9 196.4]mm, L_pos=[-68.9 -28.1 195.1]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 100] Captured marker: R_pos=[ 89.2 -36.2 193.8]mm, L_pos=[-68.9 -35.3 192.4]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 101] Captured marker: R_pos=[ 89.3 -50.6 187.5]mm, L_pos=[-68.9 -49.6 186.4]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 102] Captured marker: R_pos=[ 89.3 -57.7 184.2]mm, L_pos=[-68.9 -56.7 183.1]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260825_115424.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05418028153288892), np.float64(-0.04871348566975604), np.float64(89.44610696018094), np.float64(0.1921119155928301), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05431203759229367), np.float64(-0.04877374321076509), np.float64(90.88051933517949), np.float64(-0.08123358851487462), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.094004357516254, 'joint5': -0.2353946888087283, 'joint6': 0.33962759829571076}, 'left': {'joint3': -1.981879212861343, 'joint5': -0.1390342180215238, 'joint6': 0.07302395778084926}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.07919deg, sigma_pos=0.2326mm
Right arm joint offset (deg): [ 0.03560644  2.02447217  0.91004745  2.04400424 -2.16886828  0.18539449
 -0.28962749]
Left arm joint offset (deg): [ 0.20898862 -2.26077911 -0.20505923  1.93187934  0.22651163  0.08903442
 -0.12302385]
Head joint offset (deg): [-0.55199836 -0.17282564]
mount_to_cam xi: [-2.90982501e-04  9.02993321e-05 -8.16141085e-03 -2.74366060e-03
 -2.59388858e-03 -4.08684912e-03]
mount_to_cam_new: [0.04291365143691579, 0.011754401101454672, 0.059583257818546724, -90.01665132617686, -0.46761514218522876, -90.00510584973165]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +0.0356° | Baseline =  +0.2604° | Diff = 0.2248°
   J1: Calc =  +2.0245° | Baseline =  +2.2229° | Diff = 0.1985°
   J2: Calc =  +0.9100° | Baseline =  -0.0002° | Diff = 0.9103°
   J3: Calc =  +2.0440° | Baseline =  +1.6406° | Diff = 0.4035°
   J4: Calc =  -2.1689° | Baseline =  -0.0009° | Diff = 2.1680°
   J5: Calc =  +0.1854° | Baseline =  +0.3087° | Diff = 0.1233°
   J6: Calc =  -0.2896° | Baseline =  -0.0145° | Diff = 0.2751°
 [LEFT ARM]
   J0: Calc =  +0.2090° | Baseline =  +0.0579° | Diff = 0.1511°
   J1: Calc =  -2.2608° | Baseline =  -2.7903° | Diff = 0.5295°
   J2: Calc =  -0.2051° | Baseline =  -0.0002° | Diff = 0.2048°
   J3: Calc =  +1.9319° | Baseline =  +2.1910° | Diff = 0.2591°
   J4: Calc =  +0.2265° | Baseline =  +0.1233° | Diff = 0.1032°
   J5: Calc =  +0.0890° | Baseline =  -0.0015° | Diff = 0.0906°
   J6: Calc =  -0.1230° | Baseline =  +0.0013° | Diff = 0.1243°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
Arm: both
Right move offset (deg): [-0.0356064414412684, -2.024472171677434, -0.9100474549373161, -2.0440042358460286, 2.1688682845200176, -0.18539448873398257, 0.28962749139224014]
Left move offset (deg): [-0.2089886226022773, 2.260779111926573, 0.20505923451933042, -1.9318793376690235, -0.22651162666124217, -0.08903442482288185, 0.12302384541888517]
Head move offset (deg): [0.5519983612632118, 0.17282564176178825]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
Arm: both
Right move offset (deg): [-0.0356064414412684, -2.024472171677434, -0.9100474549373161, -2.0440042358460286, 2.1688682845200176, -0.18539448873398257, 0.28962749139224014]
Left move offset (deg): [-0.2089886226022773, 2.260779111926573, 0.20505923451933042, -1.9318793376690235, -0.22651162666124217, -0.08903442482288185, 0.12302384541888517]
Head move offset (deg): [0.5519983612632118, 0.17282564176178825]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.260408609220297, -2.2229366491336635, 0.00021755105198019802, -1.6405524829826732, 0.0008789062499999999, -0.3087158203125, 0.014501953124999998]
Left move offset (deg): [-0.05786857982673268, 2.79030979269802, 0.00021755105198019802, -2.1909566444925743, -0.12326660156249998, 0.0015380859374999997, -0.0013183593749999999]
Head move offset (deg): [-0.0208740234375, -0.0]
Preview move complete. Inspect the robot pose before applying.
[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
Arm: both
Right move offset (deg): [-0.0356064414412684, -2.024472171677434, -0.9100474549373161, -2.0440042358460286, 2.1688682845200176, -0.18539448873398257, 0.28962749139224014]
Left move offset (deg): [-0.2089886226022773, 2.260779111926573, 0.20505923451933042, -1.9318793376690235, -0.22651162666124217, -0.08903442482288185, 0.12302384541888517]
Head move offset (deg): [0.5519983612632118, 0.17282564176178825]
Preview move complete. Inspect the robot pose before applying.
[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.260408609220297, -2.2229366491336635, 0.00021755105198019802, -1.6405524829826732, 0.0008789062499999999, -0.3087158203125, 0.014501953124999998]
Left move offset (deg): [-0.05786857982673268, 2.79030979269802, 0.00021755105198019802, -2.1909566444925743, -0.12326660156249998, 0.0015380859374999997, -0.0013183593749999999]
Head move offset (deg): [-0.0208740234375, -0.0]
Preview move complete. Inspect the robot pose before applying.
[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
Arm: both
Right move offset (deg): [-0.0356064414412684, -2.024472171677434, -0.9100474549373161, -2.0440042358460286, 2.1688682845200176, -0.18539448873398257, 0.28962749139224014]
Left move offset (deg): [-0.2089886226022773, 2.260779111926573, 0.20505923451933042, -1.9318793376690235, -0.22651162666124217, -0.08903442482288185, 0.12302384541888517]
Head move offset (deg): [0.5519983612632118, 0.17282564176178825]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'OPTIMIZED' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260825_121651.json
Arm: both
Right move offset (deg): [-0.0356064414412684, -2.024472171677434, -0.9100474549373161, -2.0440042358460286, 2.1688682845200176, -0.18539448873398257, 0.28962749139224014]
Left move offset (deg): [-0.2089886226022773, 2.260779111926573, 0.20505923451933042, -1.9318793376690235, -0.22651162666124217, -0.08903442482288185, 0.12302384541888517]
Head move offset (deg): [0.5519983612632118, 0.17282564176178825]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'OPTIMIZED' Zero Pose. Now resetting and applying home offset...
[APPLY] Saving optimized mount_to_cam to setting.yaml: [0.04291365143691579, 0.011754401101454672, 0.059583257818546724, -90.01665132617686, -0.46761514218522876, -90.00510584973165]
[ERROR] Failed to save optimized camera pose to setting.yaml: 'UnifiedCalibrationApp' object has no attribute '_update_camera_key_in_lines'
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.0940°, J5=-0.2354°, J6=0.3396°] L[J3=-1.9819°, J5=-0.1390°, J6=0.0730°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (.*) on...
[INFO] Turning servos (.*) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.0940°, J5=-0.2354°, J6=0.3396°] L[J3=-1.9819°, J5=-0.1390°, J6=0.0730°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Current pose home offset apply complete.
[SUCCESS] Saved offsets permanently to setting.yaml!
[INFO] Zeroed out applied arm offsets in baseline json: /home/nvidia/camera_ws/config/home_reset_baseline.json
Calibration Wizard Finished.
[Step2] Check Calibration State requested.
[Check State] Step 1: Moving to Joint Ready Pose...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
