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

[ITERATION 2/6] Sweeping physically with staged offset -0.1729°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -0.2606°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0356° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2606°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2606° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.18°, optimal_offset=-0.13°

[ITERATION 2/6] Sweeping physically with staged offset -0.1322°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.11°, optimal_offset=-0.21°

[ITERATION 3/6] Sweeping physically with staged offset -0.2103°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.15°, optimal_offset=-0.08°

[ITERATION 4/6] Sweeping physically with staged offset -0.0784°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.03°, optimal_offset=-0.05°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0404° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0784°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.0784°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0784° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8946°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0577° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.8946°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.8946° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0784°
  * Joint 5 Change      : 0.2606°
  * Joint 3 Change      : 1.8946°
  * Bracket Pos Change  : 0.1935 mm
  * Bracket Rot Change  : 0.1976°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.2606°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2606° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset -0.0784°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.09°, optimal_offset=-0.14°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0582° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0784°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.0784°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0784° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-1.8946°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.8946° (click APPLY OFFSET to save).
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
  * Step Correction: 0.0153° < 0.06° (reached resolution limit)
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.25°, optimal_offset=-0.18°

[ITERATION 2/6] Sweeping physically with staged offset -0.1833°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.09°, optimal_offset=-0.24°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0549° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1833°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1833°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1833° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.3598°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0972°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.2249°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -2.4011°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 6/6] Sweeping physically with staged offset -2.2336°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0041° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2336°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2336° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.1833°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.2336°
  * Bracket Pos Change  : 0.1913 mm
  * Bracket Rot Change  : 0.7201°
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

[ITERATION 1/6] Sweeping physically with staged offset -0.1833°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.19°, optimal_offset=-0.02°

[ITERATION 2/6] Sweeping physically with staged offset -0.0160°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.11°, optimal_offset=0.09°

[ITERATION 3/6] Sweeping physically with staged offset 0.0878°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.04°, optimal_offset=0.13°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0447° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0878°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0878°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0878° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.2336°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2336° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.1, Z: +45.7 mm
  * Bracket Rot: R: -1.80, P: +0.03, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.08°, Joint 5: -0.26°, Joint 3: -1.89°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.2, Z: +45.1 mm
  * Bracket Rot: R: +2.01, P: -0.35, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.09°, Joint 5: +0.00°, Joint 3: -2.23°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0878°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.2336°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.0784°
    * Joint 5 (Wrist Pitch): -0.2606°
    * Joint 3 (Elbow)      : -1.8946°
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
[Sample 1] Captured marker: R_pos=[ 97.9 -15.2 164.8]mm, L_pos=[-51.2  -8.9 166.3]mm
[INFO] Posture adjustment successful. Re-building motion plan from current pose...
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 2] Captured marker: R_pos=[ 93.6 -26.9 158.5]mm, L_pos=[-46.9 -20.6 159.8]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 3] Captured marker: R_pos=[ 89.5 -38.2 151.7]mm, L_pos=[-42.8 -32.  152.7]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 4] Captured marker: R_pos=[102.2  -3.3 170.5]mm, L_pos=[-55.5   3.1 172.1]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[104.5 -15.1 167.8]mm, L_pos=[-44.1  -9.6 161.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[111.2 -15.2 170.4]mm, L_pos=[-37.1 -10.5 157.4]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 91.4 -15.5 161.6]mm, L_pos=[-58.5  -8.5 170.4]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 85.  -16.  158.4]mm, L_pos=[-65.9  -8.2 174.1]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[105.2 -27.1 163.8]mm, L_pos=[-58.3 -21.6 165.7]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[112.9 -38.8 162.2]mm, L_pos=[-65.8 -34.  164.9]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 91.   -3.  165.4]mm, L_pos=[-44.4   4.  166.3]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 84.4   9.4 165.6]mm, L_pos=[-38.1  17.  165.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 99.9 -19.7 160.7]mm, L_pos=[-49.5  -4.6 170.3]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[102.1 -24.4 156.9]mm, L_pos=[-48.1  -0.4 174.8]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 96.2 -10.7 169.1]mm, L_pos=[-52.9 -13.2 162.3]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 94.6  -6.3 173.6]mm, L_pos=[-55.  -17.8 158.6]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 80.6  -7.1 166.5]mm, L_pos=[-68.7 -17.  166. ]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 90.1 -25.  150.9]mm, L_pos=[-63.8   0.2 182.9]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[109.   -6.4 179.6]mm, L_pos=[-41.8 -19.3 150.3]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[114.3 -24.4 162. ]mm, L_pos=[-33.1  -2.  165.4]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 70.1   7.8 157.6]mm, L_pos=[-54.2  18.4 175.2]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[124.7 -39.6 166.3]mm, L_pos=[-53.  -34.9 157.8]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[ 97.9 -15.2 164.8]mm, L_pos=[-51.2  -8.9 166.2]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[100.1 -14.9 164.7]mm, L_pos=[-49.1  -9.  166.9]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[102.2 -14.7 164.6]mm, L_pos=[-47.   -9.  167.4]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 95.7 -15.5 165.2]mm, L_pos=[-53.3  -8.9 165.9]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[ 93.4 -15.7 165.5]mm, L_pos=[-55.4  -8.7 165.6]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[ 97.1 -15.2 164.3]mm, L_pos=[-50.9  -8.6 166. ]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[ 96.4 -15.3 163.8]mm, L_pos=[-50.6  -8.4 165.8]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[ 98.7 -15.1 165.4]mm, L_pos=[-51.4  -9.1 166.3]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[ 99.4 -15.1 165.8]mm, L_pos=[-51.6  -9.3 166.5]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[ 98.  -17.6 164.6]mm, L_pos=[-50.9  -6.6 166.5]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[ 98.2 -20.  164.7]mm, L_pos=[-50.7  -4.3 166.9]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[ 97.8 -12.9 165.1]mm, L_pos=[-51.4 -11.2 166.1]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[ 97.6 -10.4 165.4]mm, L_pos=[-51.6 -13.6 166.2]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[113.9 -14.9 165.2]mm, L_pos=[-35.3  -8.7 165.5]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 37] Captured marker: R_pos=[ 81.9 -15.4 164.5]mm, L_pos=[-67.1  -9.1 167.1]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 38] Captured marker: R_pos=[ 65.9 -15.5 164. ]mm, L_pos=[-83.1  -9.1 167.8]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 39] Captured marker: R_pos=[ 97.8  -0.4 164.7]mm, L_pos=[-51.3   5.9 165.6]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 40] Captured marker: R_pos=[ 97.8  14.5 164.8]mm, L_pos=[-51.3  20.6 165. ]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 41] Captured marker: R_pos=[ 97.9 -30.1 165. ]mm, L_pos=[-51.  -23.5 166.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 42] Captured marker: R_pos=[ 97.9 -44.9 165.1]mm, L_pos=[-50.8 -38.2 167.6]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260805_210630.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05408627653915202), np.float64(-0.002331689011288095), np.float64(91.80486056136135), np.float64(-0.029517882858097998), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05415750754105468), np.float64(-0.0028800175512553074), np.float64(92.01426548828071), np.float64(-0.3489641526589062), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.8945585538359504, 'joint5': -0.26056876629106474, 'joint6': -0.07839559915102086}, 'left': {'joint3': -2.2336406167856393, 'joint5': 0.0, 'joint6': 0.08777887063343712}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.1304deg, sigma_pos=0.359mm
Right arm joint offset (deg): [ 2.30526828  4.77288914 -2.79647488  1.94455859 -0.34862454  0.31056881
  0.02839554]
Left arm joint offset (deg): [ 0.82580425 -5.02575134 -0.90086056  2.28364059 -0.07233479  0.04999996
 -0.03777893]
head_base-to-camera xi: [-0.02666682  0.00395614  0.00264078  0.0028346   0.00376038  0.00061738]
head_base_to_cam_new: [0.10310577182487221, 0.007965068261127386, 0.03761262026879975, -87.55847657148237, 0.23373976902288646, -90.12784851094183]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260805_210630.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +2.3053° | Baseline =  +0.2243° | Diff = 2.0810°
   J1: Calc =  +4.7729° | Baseline =  +4.8096° | Diff = 0.0367°
   J2: Calc =  -2.7965° | Baseline =  -2.6931° | Diff = 0.1034°
   J3: Calc =  +1.9446° | Baseline =  +2.4927° | Diff = 0.5481°
   J4: Calc =  -0.3486° | Baseline =  -0.0002° | Diff = 0.3484°
   J5: Calc =  +0.3106° | Baseline =  +0.3524° | Diff = 0.0419°
   J6: Calc =  +0.0284° | Baseline =  -0.0083° | Diff = 0.0367°
 [LEFT ARM]
   J0: Calc =  +0.8258° | Baseline =  -0.0431° | Diff = 0.8689°
   J1: Calc =  -5.0258° | Baseline =  -4.2979° | Diff = 0.7278°
   J2: Calc =  -0.9009° | Baseline =  +0.0000° | Diff = 0.9009°
   J3: Calc =  +2.2836° | Baseline =  +2.1640° | Diff = 0.1197°
   J4: Calc =  -0.0723° | Baseline =  -0.0007° | Diff = 0.0717°
   J5: Calc =  +0.0500° | Baseline =  +0.0358° | Diff = 0.0142°
   J6: Calc =  -0.0378° | Baseline =  -0.0004° | Diff = 0.0373°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_210630.json
Arm: both
Right move offset (deg): [-2.305268283226677, -4.772889140063242, 2.7964748754578115, -1.9445585859205783, 0.34862454391000386, -0.31056881431354844, -0.02839554023178735]
Left move offset (deg): [-0.8258042514366482, 5.025751341350455, 0.9008605588412304, -2.28364058695428, 0.07233478513842644, -0.04999995763028444, 0.03777893156937896]
Preview move complete. Inspect the robot pose before applying.
[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.2242951345915842, -4.809618657178218, 2.693064472462871, -2.492699953589109, 0.00021972656249999998, -0.35244140625, 0.008349609374999999]
Left move offset (deg): [0.04307510829207921, 4.297938582920793, -0.0, -2.1639803140470297, 0.0006591796874999999, -0.0358154296875, 0.00043945312499999996]
Preview move complete. Inspect the robot pose before applying.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_210630.json
Arm: both
Right move offset (deg): [-2.305268283226677, -4.772889140063242, 2.7964748754578115, -1.9445585859205783, 0.34862454391000386, -0.31056881431354844, -0.02839554023178735]
Left move offset (deg): [-0.8258042514366482, 5.025751341350455, 0.9008605588412304, -2.28364058695428, 0.07233478513842644, -0.04999995763028444, 0.03777893156937896]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.2242951345915842, -4.809618657178218, 2.693064472462871, -2.492699953589109, 0.00021972656249999998, -0.35244140625, 0.008349609374999999]
Left move offset (deg): [0.04307510829207921, 4.297938582920793, -0.0, -2.1639803140470297, 0.0006591796874999999, -0.0358154296875, 0.00043945312499999996]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'OPTIMIZED' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_210630.json
Arm: both
Right move offset (deg): [-2.305268283226677, -4.772889140063242, 2.7964748754578115, -1.9445585859205783, 0.34862454391000386, -0.31056881431354844, -0.02839554023178735]
Left move offset (deg): [-0.8258042514366482, 5.025751341350455, 0.9008605588412304, -2.28364058695428, 0.07233478513842644, -0.04999995763028444, 0.03777893156937896]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'OPTIMIZED' Zero Pose. Now resetting and applying home offset...
[APPLY] Saving optimized mount_to_cam to setting.yaml: [0.10310577182487221, 0.007965068261127386, 0.03761262026879975, -87.55847657148237, 0.23373976902288646, -90.12784851094183]
[APPLY] Saving optimized head_base_to_cam to setting.yaml: [0.10310577182487221, 0.007965068261127386, 0.03761262026879975, -87.55847657148237, 0.23373976902288646, -90.12784851094183]
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-1.8946°, J5=-0.2606°, J6=-0.0784°] L[J3=-2.2336°, J5=0.0000°, J6=0.0878°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (^(?!head_joint_).*$) on...
[INFO] Turning servos (^(?!head_joint_).*$) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.0'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-1.8946°, J5=-0.2606°, J6=-0.0784°] L[J3=-2.2336°, J5=0.0000°, J6=0.0878°]
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
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Step2] Zero Pose Check requested.
Moving robot to zero pose...

===== ZERO POSE CHECK COMPLETE =====
