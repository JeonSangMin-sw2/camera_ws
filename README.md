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

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0010° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.07°, optimal_offset=0.08°

[ITERATION 2/6] Sweeping physically with staged offset 0.0809°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.01°, optimal_offset=0.12°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0348° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0809°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0809°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0809° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.0871°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.2854°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0066° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2854°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2854° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0809°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.2854°
  * Bracket Pos Change  : 0.2647 mm
  * Bracket Rot Change  : 0.5113°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (0.0000°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (0.0809°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0809° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.2854°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2854° (click APPLY OFFSET to save).
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
  * Step Correction: 0.0257° < 0.06° (reached resolution limit)
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.03°, optimal_offset=-0.01°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0061° < 0.06° (reached resolution limit)
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

[ITERATION 2/6] Sweeping physically with staged offset -1.9682°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0574°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.1367°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0315° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1367°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1367° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.0000°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.1367°
  * Bracket Pos Change  : 0.2056 mm
  * Bracket Rot Change  : 0.3659°
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
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (0.0000°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1367°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1367° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: -0.5 mm
  * Bracket Rot: R: +1.00, P: -0.22, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.08°, Joint 5: +0.00°, Joint 3: -2.29°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: -0.2 mm
  * Bracket Rot: R: -0.42, P: -0.06, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.00°, Joint 5: +0.00°, Joint 3: -2.14°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0000°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.1367°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0809°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.2854°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [ 0.    -0.003]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (+0.0deg) + Head Tilt (-5.0deg)
[Sample 1] Captured marker: R_pos=[100.5 -16.2 190.1]mm, L_pos=[-82.7 -17.  189.4]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-5.0deg)
[Sample 2] Captured marker: R_pos=[ 93.  -36.5 169.5]mm, L_pos=[-75.3 -37.2 168.7]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-10.0deg)
[Sample 3] Captured marker: R_pos=[ 93.1 -17.5 176.8]mm, L_pos=[-75.3 -18.3 176.1]mm
Auto motion done: J0 (-10.0deg) + Head Tilt (-10.0deg)
[Sample 4] Captured marker: R_pos=[ 86.1 -37.9 155.9]mm, L_pos=[-68.6 -38.5 155.3]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+0.0deg)
[Sample 5] Captured marker: R_pos=[108.5 -16.3 203.3]mm, L_pos=[-90.7 -17.1 202.7]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+5.0deg)
[Sample 6] Captured marker: R_pos=[108.4 -37.7 196. ]mm, L_pos=[-90.6 -38.3 195.2]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+5.0deg)
[Sample 7] Captured marker: R_pos=[117.  -17.7 216.3]mm, L_pos=[-99.1 -18.3 215.6]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+10.0deg)
[Sample 8] Captured marker: R_pos=[116.9 -40.2 208.9]mm, L_pos=[-99.2 -40.7 208.1]mm
Auto motion done: Restore Baseline Pose
[Sample 9] Captured marker: R_pos=[100.3 -36.5 182.8]mm, L_pos=[-82.7 -37.2 182.1]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 10] Captured marker: R_pos=[107.3 -38.4 184.7]mm, L_pos=[-75.9 -35.4 179.8]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 11] Captured marker: R_pos=[114.3 -40.3 186.3]mm, L_pos=[-69.1 -33.8 177.2]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 12] Captured marker: R_pos=[ 93.5 -34.8 180.5]mm, L_pos=[-89.6 -39.  184. ]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 13] Captured marker: R_pos=[ 86.7 -33.1 178. ]mm, L_pos=[-96.5 -41.  185.6]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 14] Captured marker: R_pos=[104.8 -48.  174.5]mm, L_pos=[-87.1 -48.7 173.6]mm
Auto motion done: Joint 2 Offset: -5.0deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Joint 2 Offset: +2.5deg
[Sample 15] Captured marker: R_pos=[ 96.3 -24.6 190.8]mm, L_pos=[-78.7 -25.3 190.2]mm
Auto motion done: Joint 2 Offset: +5.0deg
[Sample 16] Captured marker: R_pos=[ 92.6 -12.3 198.4]mm, L_pos=[-75.  -13.1 197.7]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 17] Captured marker: R_pos=[101.9 -41.  178.4]mm, L_pos=[-81.4 -32.8 186.6]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 18] Captured marker: R_pos=[103.6 -45.7 174.2]mm, L_pos=[-80.2 -28.4 191.4]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 19] Captured marker: R_pos=[ 99.1 -32.1 187.5]mm, L_pos=[-84.2 -41.7 177.7]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 20] Captured marker: R_pos=[ 97.9 -27.8 192.1]mm, L_pos=[-85.9 -46.4 173.5]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 83.4 -24.2 186.9]mm, L_pos=[-98.8 -50.1 176.7]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[ 90.9 -42.5 170.1]mm, L_pos=[-95.  -32.4 195.3]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 23] Captured marker: R_pos=[112.8 -31.8 196.1]mm, L_pos=[-73.3 -43.2 169.5]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 24] Captured marker: R_pos=[116.5 -49.3 177.3]mm, L_pos=[-65.8 -24.9 186.1]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 25] Captured marker: R_pos=[ 77.2  -8.7 192.3]mm, L_pos=[-90.7 -17.2 202.5]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Restore Baseline Pose
[Sample 26] Captured marker: R_pos=[100.4 -36.5 182.8]mm, L_pos=[-82.7 -37.2 182.1]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 27] Captured marker: R_pos=[106.5 -37.5 190. ]mm, L_pos=[-88.8 -38.3 189.2]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 28] Captured marker: R_pos=[112.8 -38.4 196.9]mm, L_pos=[-95.1 -39.2 196. ]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 29] Captured marker: R_pos=[102.2 -33.7 198. ]mm, L_pos=[-81.7 -23.2 207.7]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 30] Captured marker: R_pos=[ 99.2 -24.9 202.9]mm, L_pos=[-76.6  -2.1 221.9]mm
Auto motion done: Restore Baseline Pose
[Sample 31] Captured marker: R_pos=[100.4 -36.5 182.8]mm, L_pos=[-82.7 -37.2 182.1]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 32] Captured marker: R_pos=[102.2 -36.2 181.2]mm, L_pos=[-80.9 -37.5 183.7]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 33] Captured marker: R_pos=[104.6 -35.8 179.2]mm, L_pos=[-78.7 -37.8 186.1]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 98.2 -36.9 184.8]mm, L_pos=[-85.  -36.8 180.1]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 96.4 -37.3 186.5]mm, L_pos=[-86.9 -36.5 178.6]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 36] Captured marker: R_pos=[100.  -34.4 182.7]mm, L_pos=[-82.4 -35.1 181.9]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 37] Captured marker: R_pos=[ 99.8 -32.6 182.7]mm, L_pos=[-82.2 -33.3 181.9]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 38] Captured marker: R_pos=[100.8 -38.6 182.9]mm, L_pos=[-83.1 -39.3 182.2]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 39] Captured marker: R_pos=[101.3 -40.3 183. ]mm, L_pos=[-83.5 -40.9 182.2]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 40] Captured marker: R_pos=[100.4 -38.8 182.7]mm, L_pos=[-82.7 -34.8 182.3]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 41] Captured marker: R_pos=[100.4 -40.7 182.6]mm, L_pos=[-82.7 -32.9 182.6]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 42] Captured marker: R_pos=[100.4 -34.2 183.1]mm, L_pos=[-82.7 -39.5 182. ]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 43] Captured marker: R_pos=[100.4 -32.3 183.4]mm, L_pos=[-82.8 -41.4 182. ]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 44] Captured marker: R_pos=[103.4 -37.  188.4]mm, L_pos=[-79.5 -36.8 177.5]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 45] Captured marker: R_pos=[105.2 -37.7 194.8]mm, L_pos=[-76.9 -36.6 173.7]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 46] Captured marker: R_pos=[ 97.  -36.2 178.1]mm, L_pos=[-85.7 -37.7 187.7]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 47] Captured marker: R_pos=[ 94.4 -36.  174.3]mm, L_pos=[-87.4 -38.4 193.9]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 48] Captured marker: R_pos=[100.6 -33.3 179.9]mm, L_pos=[-82.9 -33.7 179.2]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 49] Captured marker: R_pos=[100.8 -30.8 177.6]mm, L_pos=[-83.  -31.2 176.9]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 50] Captured marker: R_pos=[100.  -39.9 186.6]mm, L_pos=[-82.5 -40.7 185.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 51] Captured marker: R_pos=[ 99.7 -42.5 191. ]mm, L_pos=[-82.3 -43.5 190.5]mm
Auto motion done: Head Pan: -3.50deg
[Sample 52] Captured marker: R_pos=[ 85.9 -37.  188. ]mm, L_pos=[-96.8 -36.6 176.1]mm
Auto motion done: Head Pan: -1.75deg
[Sample 53] Captured marker: R_pos=[ 93.2 -36.8 185.5]mm, L_pos=[-89.8 -36.9 179.2]mm
Auto motion done: Head Pan: +1.75deg
[Sample 54] Captured marker: R_pos=[107.6 -36.3 179.9]mm, L_pos=[-75.5 -37.4 184.8]mm
Auto motion done: Head Pan: +3.50deg
[Sample 55] Captured marker: R_pos=[114.6 -36.  176.8]mm, L_pos=[-68.2 -37.6 187.3]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 56] Captured marker: R_pos=[100.4 -22.4 188. ]mm, L_pos=[-82.6 -23.1 187.4]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 57] Captured marker: R_pos=[100.4 -29.5 185.5]mm, L_pos=[-82.6 -30.1 184.8]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 58] Captured marker: R_pos=[100.3 -43.4 179.8]mm, L_pos=[-82.7 -44.  179.1]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 59] Captured marker: R_pos=[100.3 -50.3 176.7]mm, L_pos=[-82.7 -50.8 175.9]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_080119.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05391218681716278), np.float64(-0.04853584118851535), np.float64(89.00474609534308), np.float64(0.22017100664386158), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.054112123996108846), np.float64(-0.048189234798709256), np.float64(89.58335745834755), np.float64(-0.0629078165932443), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2853877559237685, 'joint5': 0.0, 'joint6': 0.08086111523160816}, 'left': {'joint3': -2.136694958302961, 'joint5': 0.0, 'joint6': 0.0}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.08759deg, sigma_pos=0.3613mm
Right arm joint offset (deg): [-1.24580252  1.49402724  1.10599826  2.23538775 -1.94764511 -0.05000001
 -0.03086111]
Left arm joint offset (deg): [-1.24272789 -0.91827409 -1.12030092  2.08669496  2.20285777 -0.05
 -0.05      ]
Head joint offset (deg): [-0.00672257  5.83412045]
mount_to_cam xi: [-0.00040449 -0.00032967 -0.00019727  0.00035006 -0.00762879 -0.00642385]
mount_to_cam_new: [0.03805471217186415, 0.009438692796378939, 0.06533638304976548, -90.07321392186392, -0.12128224876489394, -89.93109919548064]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_080119.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -1.2458° | Baseline =  +0.1573° | Diff = 1.4031°
   J1: Calc =  +1.4940° | Baseline =  +2.4059° | Diff = 0.9119°
   J2: Calc =  +1.1060° | Baseline =  +0.0007° | Diff = 1.1053°
   J3: Calc =  +2.2354° | Baseline =  +2.2051° | Diff = 0.0303°
   J4: Calc =  -1.9476° | Baseline =  +0.0020° | Diff = 1.9496°
   J5: Calc =  -0.0500° | Baseline =  +0.0090° | Diff = 0.0590°
   J6: Calc =  -0.0309° | Baseline =  +0.0101° | Diff = 0.0410°
 [LEFT ARM]
   J0: Calc =  -1.2427° | Baseline =  +0.1042° | Diff = 1.3469°
   J1: Calc =  -0.9183° | Baseline =  -1.7528° | Diff = 0.8345°
   J2: Calc =  -1.1203° | Baseline =  -0.0007° | Diff = 1.1196°
   J3: Calc =  +2.0867° | Baseline =  +2.1662° | Diff = 0.0795°
   J4: Calc =  +2.2029° | Baseline =  -0.0002° | Diff = 2.2031°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  -0.0500° | Baseline =  +0.0000° | Diff = 0.0500°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_080119.json
Arm: both
Right move offset (deg): [1.245802521947887, -1.4940272386647138, -1.1059982560663375, -2.2353877473634145, 1.9476451100791383, 0.05000001010902222, 0.030861113415987147]
Left move offset (deg): [1.2427278861973, 0.9182740859720999, 1.1203009204480212, -2.0866949580949967, -2.2028577706875514, 0.04999999884478757, 0.0499999968217344]
Head move offset (deg): [0.006722566723940236, -5.834120448401992]
Preview move complete. Inspect the robot pose before applying.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_080119.json
Arm: both
Right move offset (deg): [1.245802521947887, -1.4940272386647138, -1.1059982560663375, -2.2353877473634145, 1.9476451100791383, 0.05000001010902222, 0.030861113415987147]
Left move offset (deg): [1.2427278861973, 0.9182740859720999, 1.1203009204480212, -2.0866949580949967, -2.2028577706875514, 0.04999999884478757, 0.0499999968217344]
Head move offset (deg): [0.006722566723940236, -5.834120448401992]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.1572894105816832, -2.40589708384901, -0.0006526531559405941, -2.205097462871287, -0.0019775390625, -0.009008789062499998, -0.010107421874999998]
Left move offset (deg): [-0.10420695389851488, 1.7528088258044554, 0.0006526531559405941, -2.166155824566832, 0.00021972656249999998, 0.0015380859374999997, -0.0]
Head move offset (deg): [-0.0, -5.0000976562499995]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_080119.json
Arm: both
Right move offset (deg): [1.245802521947887, -1.4940272386647138, -1.1059982560663375, -2.2353877473634145, 1.9476451100791383, 0.05000001010902222, 0.030861113415987147]
Left move offset (deg): [1.2427278861973, 0.9182740859720999, 1.1203009204480212, -2.0866949580949967, -2.2028577706875514, 0.04999999884478757, 0.0499999968217344]
Head move offset (deg): [0.006722566723940236, -5.834120448401992]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.1572894105816832, -2.40589708384901, -0.0006526531559405941, -2.205097462871287, -0.0019775390625, -0.009008789062499998, -0.010107421874999998]
Left move offset (deg): [-0.10420695389851488, 1.7528088258044554, 0.0006526531559405941, -2.166155824566832, 0.00021972656249999998, 0.0015380859374999997, -0.0]
Head move offset (deg): [-0.0, -5.0000976562499995]
Preview move complete. Inspect the robot pose before applying.
