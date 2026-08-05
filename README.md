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

[ITERATION 2/6] Sweeping physically with staged offset -0.5671°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -0.8130°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -0.9283°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0026° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.9283°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.9283° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.06°, optimal_offset=0.06°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0593° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.9573°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.6661°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.7265°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0085° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.7265°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.7265° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0000°
  * Joint 5 Change      : 0.9283°
  * Joint 3 Change      : 2.7265°
  * Bracket Pos Change  : 0.1595 mm
  * Bracket Rot Change  : 0.5720°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.9283°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.9283° (click APPLY OFFSET to save).
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

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.34°, optimal_offset=-0.26°

[ITERATION 2/6] Sweeping physically with staged offset -0.2649°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.05°, optimal_offset=-0.30°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0317° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2649°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.2649°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.2649° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.7265°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.7265° (click APPLY OFFSET to save).
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

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0037° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.08°, optimal_offset=-0.05°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0507° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.5778°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.1863°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0213° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1863°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1863° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.0000°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.1863°
  * Bracket Pos Change  : 0.2237 mm
  * Bracket Rot Change  : 0.5887°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (0.0000°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.09°, optimal_offset=-0.05°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0524° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1863°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1863° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.2, Z: +45.6 mm
  * Bracket Rot: R: -1.83, P: -0.00, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.26°, Joint 5: -0.93°, Joint 3: -2.73°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: +45.3 mm
  * Bracket Rot: R: +1.71, P: -0.35, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.00°, Joint 5: +0.00°, Joint 3: -2.19°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0000°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.1863°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.2649°
    * Joint 5 (Wrist Pitch): -0.9283°
    * Joint 3 (Elbow)      : -2.7265°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: 0.0deg
Marker not detected.
[WARNING] Marker not detected at the initial ready pose. Prompting posture adjustment...
[INFO] Right arm marker not visible. Showing teaching dialog...
[INFO] Preserved user-taught ready pose for right arm (elbow).
[INFO] Re-verifying marker visibility at the new posture...
[Sample 1] Captured marker: R_pos=[ 69.4 -16.  218.3]mm, L_pos=[-92.4  -8.5 203.9]mm
[INFO] Posture adjustment successful. Re-building motion plan from current pose...
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 2] Captured marker: R_pos=[ 64.3 -29.9 211.5]mm, L_pos=[-87.6 -21.8 198.1]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 3] Captured marker: R_pos=[ 59.4 -43.4 204. ]mm, L_pos=[-82.9 -34.8 191.6]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 4] Captured marker: R_pos=[ 74.6  -1.9 224.6]mm, L_pos=[-97.3   5.  209.2]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[ 79.4 -17.1 222.5]mm, L_pos=[-84.3  -7.9 200.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[ 89.5 -18.4 226.3]mm, L_pos=[-76.4  -7.6 197.8]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 59.6 -15.2 213.7]mm, L_pos=[-100.5   -9.3  206.5]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 49.9 -14.7 208.9]mm, L_pos=[-108.7  -10.4  208.8]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[ 77.1 -29.8 217.7]mm, L_pos=[-99.  -22.  202.3]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[ 85.2 -43.1 216.5]mm, L_pos=[-106.   -35.1  200.1]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 62.2  -2.1 218.7]mm, L_pos=[-86.2   5.1 205.3]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 55.5  12.1 218.8]mm, L_pos=[-80.4  18.9 205.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 70.6 -20.5 213.9]mm, L_pos=[-91.6  -3.9 208.2]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[ 72.1 -25.1 209.6]mm, L_pos=[-91.    0.6 212.5]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 68.5 -11.7 222.9]mm, L_pos=[-93.4 -13.1 199.8]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 67.7  -7.4 228.2]mm, L_pos=[-94.7 -17.9 196. ]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 47.1  -5.9 218.1]mm, L_pos=[-110.   -19.7  200.6]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 53.5 -23.8 200.6]mm, L_pos=[-108.4   -1.5  217.7]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[ 88.9 -10.  236.4]mm, L_pos=[-79.6 -17.  190.4]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[ 91.1 -27.3 217.1]mm, L_pos=[-74.    1.5 206.2]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 34.8  12.7 207.4]mm, L_pos=[-98.4  17.6 212.3]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[104.  -46.1 222.8]mm, L_pos=[-91.5 -33.6 195.5]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[ 69.4 -16.  218.3]mm, L_pos=[-92.4  -8.5 203.9]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[ 71.4 -16.  217.7]mm, L_pos=[-90.4  -8.6 204.7]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[ 73.4 -16.2 217.6]mm, L_pos=[-88.3  -8.5 205.7]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 67.5 -16.1 219. ]mm, L_pos=[-94.5  -8.5 203.3]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[ 65.6 -16.  219.9]mm, L_pos=[-96.5  -8.5 202.7]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[ 69.1 -15.9 218.1]mm, L_pos=[-91.9  -8.4 203.6]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[ 68.8 -15.6 218. ]mm, L_pos=[-91.4  -8.3 203.3]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[ 69.7 -16.3 218.5]mm, L_pos=[-92.8  -8.7 204.1]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[ 70.  -16.4 218.8]mm, L_pos=[-93.3  -8.8 204.3]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[ 69.5 -18.4 218.1]mm, L_pos=[-92.3  -6.2 204. ]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[ 69.6 -20.7 218.1]mm, L_pos=[-92.3  -3.9 204.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[ 69.4 -13.8 218.8]mm, L_pos=[-92.5 -10.9 203.8]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[ 69.3 -11.5 219.2]mm, L_pos=[-92.6 -13.3 203.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[ 85.3 -16.2 219.1]mm, L_pos=[-76.5  -8.7 203.2]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[101.  -16.2 219.7]mm, L_pos=[-60.6  -8.5 202.4]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[ 53.7 -15.8 217.7]mm, L_pos=[-108.3   -8.4  204.6]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[ 38.  -15.5 216.9]mm, L_pos=[-124.3   -8.2  205.3]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[ 69.7  -1.3 217.8]mm, L_pos=[-92.5   6.3 203.9]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[ 69.8  13.4 217. ]mm, L_pos=[-92.5  21.1 203.6]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[ 69.1 -30.7 219. ]mm, L_pos=[-92.3 -23.4 204.1]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[ 68.8 -45.4 219.9]mm, L_pos=[-92.  -38.1 204.3]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260805_221431.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05416021650221257), np.float64(-0.0023667590335386185), np.float64(91.83141394577045), np.float64(0.0013435658545367178), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05413135384325119), np.float64(-0.002729020755160572), np.float64(91.71169661851785), np.float64(-0.3521048315622579), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.726473874686557, 'joint5': -0.9282852004151552, 'joint6': -0.26489339977401477}, 'left': {'joint3': -2.1863306164844514, 'joint5': 0.0, 'joint6': 0.0}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.1922deg, sigma_pos=0.3333mm
Right arm joint offset (deg): [ 0.61281888  4.81875111  0.45428284  2.77647382 -0.44357781  0.9782851
  0.21489366]
Left arm joint offset (deg): [-0.68207764 -4.33011007  0.70785014  2.23633068  0.44287247  0.0500001
  0.05000026]
head_base-to-camera xi: [ 0.01416308 -0.00746878  0.00041717 -0.00210907 -0.00371316 -0.00087075]
head_base_to_cam_new: [0.10205285103156425, 0.010062629132623957, 0.041283808624917664, -86.74678102962154, 0.2326207733234383, -89.7014112806581]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260805_221431.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +0.6128° | Baseline =  +2.4335° | Diff = 1.8207°
   J1: Calc =  +4.8188° | Baseline =  +4.7766° | Diff = 0.0422°
   J2: Calc =  +0.4543° | Baseline =  -0.0002° | Diff = 0.4545°
   J3: Calc =  +2.7765° | Baseline =  +2.0676° | Diff = 0.7089°
   J4: Calc =  -0.4436° | Baseline =  -0.0020° | Diff = 0.4416°
   J5: Calc =  +0.9783° | Baseline =  +0.9187° | Diff = 0.0596°
   J6: Calc =  +0.2149° | Baseline =  +0.0026° | Diff = 0.2123°
 [LEFT ARM]
   J0: Calc =  -0.6821° | Baseline =  -0.0194° | Diff = 0.6627°
   J1: Calc =  -4.3301° | Baseline =  -5.0315° | Diff = 0.7014°
   J2: Calc =  +0.7079° | Baseline =  +0.0000° | Diff = 0.7079°
   J3: Calc =  +2.2363° | Baseline =  +2.2830° | Diff = 0.0467°
   J4: Calc =  +0.4429° | Baseline =  +0.0009° | Diff = 0.4420°
   J5: Calc =  +0.0500° | Baseline =  +0.0363° | Diff = 0.0137°
   J6: Calc =  +0.0500° | Baseline =  -0.0053° | Diff = 0.0553°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_221431.json
Arm: both
Right move offset (deg): [-0.6128188845267524, -4.818751109707167, -0.4542828381384349, -2.7764738154536284, 0.44357780923081497, -0.978285103827088, -0.21489365749816672]
Left move offset (deg): [0.6820776414696303, 4.330110071904625, -0.7078501384860189, -2.2363306757379893, -0.44287246791202134, -0.05000009608108779, -0.05000026088138836]
Preview move complete. Inspect the robot pose before applying.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_221431.json
Arm: both
Right move offset (deg): [-0.6128188845267524, -4.818751109707167, -0.4542828381384349, -2.7764738154536284, 0.44357780923081497, -0.978285103827088, -0.21489365749816672]
Left move offset (deg): [0.6820776414696303, 4.330110071904625, -0.7078501384860189, -2.2363306757379893, -0.44287246791202134, -0.05000009608108779, -0.05000026088138836]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-2.433526067450495, -4.776550897277228, 0.00021755105198019802, -2.0676051980198022, 0.0019775390625, -0.9186767578124999, -0.0026367187499999997]
Left move offset (deg): [0.019362043626237627, 5.03152073019802, -0.0, -2.282980739480198, -0.0008789062499999999, -0.0362548828125, 0.0052734374999999995]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'OPTIMIZED' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_221431.json
Arm: both
Right move offset (deg): [-0.6128188845267524, -4.818751109707167, -0.4542828381384349, -2.7764738154536284, 0.44357780923081497, -0.978285103827088, -0.21489365749816672]
Left move offset (deg): [0.6820776414696303, 4.330110071904625, -0.7078501384860189, -2.2363306757379893, -0.44287246791202134, -0.05000009608108779, -0.05000026088138836]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'OPTIMIZED' Zero Pose. Now resetting and applying home offset...
[APPLY] Saving optimized mount_to_cam to setting.yaml: [0.10205285103156425, 0.010062629132623957, 0.041283808624917664, -86.74678102962154, 0.2326207733234383, -89.7014112806581]
[APPLY] Saving optimized head_base_to_cam to setting.yaml: [0.10205285103156425, 0.010062629132623957, 0.041283808624917664, -86.74678102962154, 0.2326207733234383, -89.7014112806581]
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.7265°, J5=-0.9283°, J6=-0.2649°] L[J3=-2.1863°, J5=0.0000°, J6=0.0000°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (^(?!head_joint_).*$) on...
[INFO] Turning servos (^(?!head_joint_).*$) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.0'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.7265°, J5=-0.9283°, J6=-0.2649°] L[J3=-2.1863°, J5=0.0000°, J6=0.0000°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Current pose home offset apply complete.
[SUCCESS] Saved offsets permanently to setting.yaml!
[INFO] Zeroed out applied arm offsets in baseline json: /home/nvidia/camera_ws/config/home_reset_baseline.json
[Step2] Check Calibration State requested.
[Check State] Step 1: Moving to Joint Ready Pose...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
