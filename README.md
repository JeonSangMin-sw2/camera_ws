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

[ITERATION 2/6] Sweeping physically with staged offset -0.4419°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -0.3574°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0106° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.3574°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.3574° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.12°, optimal_offset=0.12°

[ITERATION 2/6] Sweeping physically with staged offset 0.1169°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.02°, optimal_offset=0.15°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0355° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.1169°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.1169°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.1169° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.0522°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0397° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.0522°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.0522° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.1169°
  * Joint 5 Change      : 0.3574°
  * Joint 3 Change      : 2.0522°
  * Bracket Pos Change  : 1.0383 mm
  * Bracket Rot Change  : 6.0507°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.3574°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.3574° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset 0.1169°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.28°, optimal_offset=0.36°

[ITERATION 2/6] Sweeping physically with staged offset 0.3583°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.03°, optimal_offset=0.35°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0119° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.3583°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.3583°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.3583° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.0522°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.0522° (click APPLY OFFSET to save).
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

[ITERATION 2/6] Sweeping physically with staged offset -2.0621°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.5302°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.6208°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0035° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.6208°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -2.6208° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.17°, optimal_offset=-0.13°

[ITERATION 2/6] Sweeping physically with staged offset -0.1322°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.05°, optimal_offset=-0.17°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0338° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1322°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1322°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1322° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.7926°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.9082°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.9941°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0005° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.9941°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9941° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.1322°
  * Joint 5 Change      : 2.6208°
  * Joint 3 Change      : 1.9941°
  * Bracket Pos Change  : 0.1776 mm
  * Bracket Rot Change  : 0.1473°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-2.6208°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -2.6208° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset -0.1322°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.11°, optimal_offset=-0.22°

[ITERATION 2/6] Sweeping physically with staged offset -0.2159°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=0.01°, optimal_offset=-0.20°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0121° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2159°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.2159°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.2159° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-1.9941°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9941° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.5, Z: -1.0 mm
  * Bracket Rot: R: +0.04, P: +0.33, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.36°, Joint 5: -0.36°, Joint 3: -2.05°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.8, Z: -0.3 mm
  * Bracket Rot: R: -1.10, P: -0.08, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.22°, Joint 5: -2.62°, Joint 3: -1.99°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.2159°
    * Joint 5 (Wrist Pitch): -2.6208°
    * Joint 3 (Elbow)      : -1.9941°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.3583°
    * Joint 5 (Wrist Pitch): -0.3574°
    * Joint 3 (Elbow)      : -2.0522°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[ERROR] Failed to save bracket values: 'Marker_Detection' object has no attribute 'make_transform'
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
Auto base head pose (deg): [-0.02  -0.025]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
Marker not detected.
[WARNING] Marker not detected at the initial ready pose. Prompting posture adjustment...
[INFO] Right arm marker not visible. Showing teaching dialog...
[INFO] Preserved user-taught ready pose for right arm (elbow).
[INFO] Re-verifying marker visibility at the new posture...
[Sample 1] Captured marker: R_pos=[ 87.2  10.5 223.7]mm, L_pos=[-55.5  17.4 226. ]mm
[INFO] Posture adjustment successful. Re-building motion plan from current pose...
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 2] Captured marker: R_pos=[ 79.9  11.1 209.5]mm, L_pos=[-46.   18.  213.3]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 3] Captured marker: R_pos=[ 90.9  10.  230.8]mm, L_pos=[-60.5  17.2 232.6]mm
Auto motion done: Joint 0 Offset: 5.0deg
[Sample 4] Captured marker: R_pos=[ 94.9   9.4 238. ]mm, L_pos=[-65.6  16.6 238.7]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[ 86.3   8.9 229.2]mm, L_pos=[-55.5  19.  220.2]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[ 85.4   7.5 234.9]mm, L_pos=[-55.4  20.7 214.4]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 87.7  11.9 218. ]mm, L_pos=[-55.3  15.7 231.7]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 88.2  13.3 212.3]mm, L_pos=[-54.8  14.1 237.4]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[ 92.9   9.1 222. ]mm, L_pos=[-60.4  16.  221.9]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[ 99.1   7.7 220.6]mm, L_pos=[-65.6  14.5 218.1]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 81.8  12.2 225.8]mm, L_pos=[-51.1  18.7 230.3]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 76.9  13.9 228.2]mm, L_pos=[-47.   19.9 234.7]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 89.4   5.2 220.5]mm, L_pos=[-54.3  22.6 229.6]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[ 9.170e+01 -1.000e-01  2.174e+02]mm, L_pos=[-53.5  27.9 233.6]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 85.2  15.8 227.1]mm, L_pos=[-56.8  12.1 222.6]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 83.3  21.1 230.7]mm, L_pos=[-58.3   6.8 219.4]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 84.   23.9 219. ]mm, L_pos=[-57.3   3.5 230.6]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 93.1   2.9 206.6]mm, L_pos=[-53.1  24.6 245.3]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[ 81.9  18.2 242.4]mm, L_pos=[-58.6  10.1 208.2]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[ 89.4  -3.1 228.2]mm, L_pos=[-53.1  31.3 221.6]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 77.7  17.8 216.1]mm, L_pos=[-46.4  15.6 247.1]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[ 97.2   5.7 230.8]mm, L_pos=[-65.6  16.9 207.5]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[ 87.1  10.6 223.7]mm, L_pos=[-55.6  17.4 226. ]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[ 89.7  10.5 221.8]mm, L_pos=[-53.4  17.5 227.7]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[ 92.2  10.5 220. ]mm, L_pos=[-51.5  17.7 229.5]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 84.6  10.6 225.7]mm, L_pos=[-57.7  17.2 224.3]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[ 82.2  10.8 227.7]mm, L_pos=[-59.8  17.1 222.5]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[ 86.5  12.6 223. ]mm, L_pos=[-54.8  19.6 225.3]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[ 85.9  14.8 222.4]mm, L_pos=[-54.1  21.8 224.6]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[ 87.9   8.4 224.2]mm, L_pos=[-56.4  15.2 226.7]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[ 88.7   6.4 224.9]mm, L_pos=[-57.3  13.  227.3]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[ 87.3   8.2 224. ]mm, L_pos=[-55.7  19.7 225.9]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[ 87.3   6.  224.6]mm, L_pos=[-55.8  22.1 226. ]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[ 87.   12.8 223.4]mm, L_pos=[-55.4  15.2 226.2]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[ 86.9  15.2 223.1]mm, L_pos=[-55.3  13.  226.7]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[ 90.2  10.6 227.7]mm, L_pos=[-52.5  17.  221.3]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[ 93.   10.8 232.6]mm, L_pos=[-49.3  16.7 217.6]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[ 83.8  10.6 220.5]mm, L_pos=[-58.5  17.9 231.5]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[ 80.2  10.3 218.2]mm, L_pos=[-61.1  18.3 237.7]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[ 87.6  12.5 221. ]mm, L_pos=[-56.   19.5 223.8]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[ 88.   14.5 218.9]mm, L_pos=[-56.4  21.6 222.4]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[ 86.5   8.5 227.1]mm, L_pos=[-55.   15.4 229.1]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[ 85.9   6.5 231.1]mm, L_pos=[-54.3  13.5 233.2]mm
Auto motion done: Head Pan: -3.50deg
[Sample 44] Captured marker: R_pos=[ 70.8  11.  226.5]mm, L_pos=[-71.7  17.4 220.2]mm
Auto motion done: Head Pan: -1.75deg
[Sample 45] Captured marker: R_pos=[ 78.9  10.7 225.2]mm, L_pos=[-63.7  17.3 223.1]mm
Auto motion done: Head Pan: +1.75deg
[Sample 46] Captured marker: R_pos=[ 95.2  10.4 221.9]mm, L_pos=[-47.3  17.5 228.8]mm
Auto motion done: Head Pan: +3.50deg
[Sample 47] Captured marker: R_pos=[103.4  10.2 219.9]mm, L_pos=[-39.   17.6 231.5]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 48] Captured marker: R_pos=[ 87.4  27.1 225.8]mm, L_pos=[-55.2  34.  227.5]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 49] Captured marker: R_pos=[ 87.2  18.9 224.9]mm, L_pos=[-55.4  25.8 226.9]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 50] Captured marker: R_pos=[ 87.1   2.4 222.1]mm, L_pos=[-55.6   9.2 224.9]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 51] Captured marker: R_pos=[ 87.   -5.8 220.3]mm, L_pos=[-55.7   1.  223.5]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260812_140518.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05451634428664735), np.float64(-0.049046976063608395), np.float64(89.96227079279427), np.float64(-0.3322024967081757), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05476283298062061), np.float64(-0.04832055405431323), np.float64(88.89629508519164), np.float64(-0.07756692271780662), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.0521780049589093, 'joint5': -0.35744427843845744, 'joint6': 0.35829917905157926}, 'left': {'joint3': -1.9940545604582631, 'joint5': -2.620775700247805, 'joint6': -0.2158559917446235}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=2deg, sigma_pos=0.678mm
Right arm joint offset (deg): [ 4.94506086  3.19174174 -2.11257771  2.00217777 -2.38203531  0.30744368
 -0.3082982 ]
Left arm joint offset (deg): [-1.54551918 -6.03371501 -2.64853468  2.0440548  -3.02978159  2.67077626
  0.26585691]
Head joint offset (deg): [ 5.00383053 -1.84533886]
mount_to_cam xi: [0. 0. 0. 0. 0. 0.]
mount_to_cam_new: [0.0495, -0.0115, 0.044, -90.0, -0.0, -90.0]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260812_140518.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +4.9451° | Baseline =  +0.4886° | Diff = 4.4564°
   J1: Calc =  +3.1917° | Baseline =  +4.4061° | Diff = 1.2143°
   J2: Calc =  -2.1126° | Baseline =  +0.0013° | Diff = 2.1139°
   J3: Calc =  +2.0022° | Baseline =  +1.5709° | Diff = 0.4312°
   J4: Calc =  -2.3820° | Baseline =  -0.0048° | Diff = 2.3772°
   J5: Calc =  +0.3074° | Baseline =  +0.3999° | Diff = 0.0925°
   J6: Calc =  -0.3083° | Baseline =  -0.0040° | Diff = 0.3043°
 [LEFT ARM]
   J0: Calc =  -1.5455° | Baseline =  +1.6447° | Diff = 3.1902°
   J1: Calc =  -6.0337° | Baseline =  -4.1243° | Diff = 1.9094°
   J2: Calc =  -2.6485° | Baseline =  -0.0037° | Diff = 2.6448°
   J3: Calc =  +2.0441° | Baseline =  +1.9142° | Diff = 0.1298°
   J4: Calc =  -3.0298° | Baseline =  +0.0011° | Diff = 3.0309°
   J5: Calc =  +2.6708° | Baseline =  +2.6444° | Diff = 0.0264°
   J6: Calc =  +0.2659° | Baseline =  -0.0167° | Diff = 0.2826°
=========================================================

Optimization finished successfully.
