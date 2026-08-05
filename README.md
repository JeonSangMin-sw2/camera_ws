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

[ITERATION 2/6] Sweeping physically with staged offset 2.4582°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset 2.1545°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset 2.0934°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset 2.1987°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0378° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 2.1987°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 2.1987° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.93°, optimal_offset=0.94°

[ITERATION 2/6] Sweeping physically with staged offset 0.9397°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.30°, optimal_offset=0.65°

[ITERATION 3/6] Sweeping physically with staged offset 0.6460°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.14°, optimal_offset=0.79°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0105° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.6460°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.6460°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.6460° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -3.2410°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.1247°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.4638°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0008° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.4638°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.4638° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.6460°
  * Joint 5 Change      : 2.1987°
  * Joint 3 Change      : 2.4638°
  * Bracket Pos Change  : 43.4867 mm
  * Bracket Rot Change  : 0.4734°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (2.1987°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 2.1987° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset 0.6460°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.06°, optimal_offset=0.60°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0499° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.6460°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.6460°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.6460° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.4638°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.4638° (click APPLY OFFSET to save).
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

[ITERATION 2/6] Sweeping physically with staged offset -0.1747°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -0.2630°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -0.4741°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -0.3970°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0090° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.3970°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.3970° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.57°, optimal_offset=-0.56°

[ITERATION 2/6] Sweeping physically with staged offset -0.5552°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.11°, optimal_offset=-0.66°

[ITERATION 3/6] Sweeping physically with staged offset -0.6582°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.01°, optimal_offset=-0.64°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0212° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.6582°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.6582°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.6582° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.6378°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.2664°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0054° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2664°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2664° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.6582°
  * Joint 5 Change      : 0.3970°
  * Joint 3 Change      : 2.2664°
  * Bracket Pos Change  : 45.7666 mm
  * Bracket Rot Change  : 1.9158°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.3970°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.3970° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset -0.6582°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.13°, optimal_offset=-0.51°

[ITERATION 2/6] Sweeping physically with staged offset -0.5148°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.02°, optimal_offset=-0.52°

[ITERATION 3/6] Sweeping physically with staged offset -0.4185°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.28°, optimal_offset=-0.68°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0181° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.4185°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.4185°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.4185° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.2664°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2664° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.2, Z: +45.2 mm
  * Bracket Rot: R: -0.26, P: +0.00, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.65°, Joint 5: +2.20°, Joint 3: -2.46°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.6, Z: +44.0 mm
  * Bracket Rot: R: +2.63, P: -0.14, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.42°, Joint 5: -0.40°, Joint 3: -2.27°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.4185°
    * Joint 5 (Wrist Pitch): -0.3970°
    * Joint 3 (Elbow)      : -2.2664°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.6460°
    * Joint 5 (Wrist Pitch): 2.1987°
    * Joint 3 (Elbow)      : -2.4638°
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
[Sample 1] Captured marker: R_pos=[ 91.4   5.9 196.5]mm, L_pos=[-105.4   18.9  162.3]mm
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 2] Captured marker: R_pos=[ 88.   -4.6 189.6]mm, L_pos=[-101.5    9.2  156.5]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 3] Captured marker: R_pos=[ 84.8 -14.8 182.3]mm, L_pos=[-9.760e+01 -1.000e-01  1.503e+02]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 4] Captured marker: R_pos=[ 94.9  16.6 202.8]mm, L_pos=[-109.4   28.7  167.7]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[ 94.6   7.3 198.8]mm, L_pos=[-102.    17.2  160.2]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[ 97.9   8.6 201.1]mm, L_pos=[-98.7  15.5 158. ]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 88.4   4.3 194. ]mm, L_pos=[-108.9   20.4  164.4]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 85.4   2.6 191.5]mm, L_pos=[-112.5   21.7  166.3]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[ 96.8  -6.1 193.7]mm, L_pos=[-109.7    7.6  159.3]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[102.3 -17.8 190.5]mm, L_pos=[-114.1   -3.4  155.9]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 86.3  18.  198.7]mm, L_pos=[-101.3   30.3  164.8]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 81.5  30.4 200.4]mm, L_pos=[-97.3  41.8 166.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 93.5  -0.  192.4]mm, L_pos=[-104.2   25.2  167.4]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[ 95.7  -6.  188.5]mm, L_pos=[-103.3   31.3  172.8]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 89.6  11.7 200.7]mm, L_pos=[-106.9   12.5  157.4]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 88.   17.4 205.3]mm, L_pos=[-108.6    6.   152.7]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 80.8  14.1 199.8]mm, L_pos=[-114.3    8.8  156.4]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 90.9  -9.1 184.2]mm, L_pos=[-111.8   34.1  177. ]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[ 95.7  20.1 210.3]mm, L_pos=[-103.3    2.7  148.8]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[101.   -3.3 192.7]mm, L_pos=[-95.2  27.9 168.1]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 73.7  26.5 194.3]mm, L_pos=[-106.2   45.1  171.6]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[106.9 -15.6 194.1]mm, L_pos=[-109.2   -6.2  152.4]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[ 91.4   5.9 196.5]mm, L_pos=[-105.4   18.8  162.3]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[ 93.1   6.1 194.6]mm, L_pos=[-103.    19.   164.9]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[ 94.9   6.3 192.8]mm, L_pos=[-100.7   19.1  167.6]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 89.8   5.7 198.4]mm, L_pos=[-107.9   18.8  159.9]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[ 88.2   5.4 200.4]mm, L_pos=[-110.5   18.7  157.6]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[ 90.7   7.9 196. ]mm, L_pos=[-105.4   21.1  162.3]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[ 90.1   9.9 195.5]mm, L_pos=[-105.6   23.4  162.3]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[ 92.2   3.9 196.9]mm, L_pos=[-105.4   16.7  162.3]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[ 93.    2.  197.4]mm, L_pos=[-105.5   14.4  162.3]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[ 91.6   3.7 196.7]mm, L_pos=[-105.3   21.3  162.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[ 91.8   1.6 197.2]mm, L_pos=[-105.3   23.8  162.3]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[ 91.2   8.1 196.3]mm, L_pos=[-105.4   16.4  162.5]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[ 91.   10.3 196.2]mm, L_pos=[-105.4   14.   162.8]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[106.9   6.2 198.2]mm, L_pos=[-89.9  19.1 163.3]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[122.3   6.5 199.9]mm, L_pos=[-74.4  19.4 164.2]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[ 75.9   5.7 194.8]mm, L_pos=[-120.9   18.7  161.4]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[ 60.5   5.6 192.9]mm, L_pos=[-136.3   18.5  160.4]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[ 91.3  21.2 196.9]mm, L_pos=[-105.4   33.5  162.1]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[ 91.1  36.3 197. ]mm, L_pos=[-105.4   48.1  161.9]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[ 91.5  -9.1 196.5]mm, L_pos=[-105.3    4.2  162.5]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[ 91.5 -24.1 196.2]mm, L_pos=[-105.3  -10.5  162.7]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260805_144425.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054239486772378075), np.float64(-0.0027899800576744443), np.float64(90.26165972959966), np.float64(-9.540283552619942e-05), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05457759793319391), np.float64(-0.0040247196086646165), np.float64(92.62608517023781), np.float64(-0.13964829266121878), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.463807165113629, 'joint5': 2.198668580869063, 'joint6': 0.6459591201536343}, 'left': {'joint3': -2.2664350086799248, 'joint5': -0.3969580317953567, 'joint6': -0.4185333802089699}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=2deg, sigma_pos=1mm
Right arm joint offset (deg): [-30.04866364   4.68914741  -8.70325167   2.51380587  14.79469232
  -2.14867138  -0.69595092]
Left arm joint offset (deg): [-36.19719782  -4.83271064  -3.84475603   2.21643656 -12.22612504
   0.4469625    0.46853883]
head_base-to-camera xi: [ 0.69639358  0.10995597  0.01902974  0.02069792 -0.05418933 -0.08475634]
head_base_to_cam_new: [0.004927844287537633, -0.006823675359152974, 0.06518739442949287, -50.206884469650895, 3.1078872729577287, -95.44256657221581]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260805_144425.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc = -30.0487° | Baseline =  -0.3392° | Diff = 29.7095°
   J1: Calc =  +4.6891° | Baseline =  +2.8647° | Diff = 1.8244°
   J2: Calc =  -8.7033° | Baseline =  -0.0011° | Diff = 8.7022°
   J3: Calc =  +2.5138° | Baseline =  +2.5619° | Diff = 0.0481°
   J4: Calc = +14.7947° | Baseline =  +0.0033° | Diff = 14.7914°
   J5: Calc =  -2.1487° | Baseline =  +0.0870° | Diff = 2.2357°
   J6: Calc =  -0.6960° | Baseline =  -0.0088° | Diff = 0.6872°
 [LEFT ARM]
   J0: Calc = -36.1972° | Baseline =  -0.1364° | Diff = 36.0608°
   J1: Calc =  -4.8327° | Baseline =  -2.6189° | Diff = 2.2138°
   J2: Calc =  -3.8448° | Baseline =  -0.0026° | Diff = 3.8421°
   J3: Calc =  +2.2164° | Baseline =  +3.1454° | Diff = 0.9289°
   J4: Calc = -12.2261° | Baseline =  -0.0083° | Diff = 12.2178°
   J5: Calc =  +0.4470° | Baseline =  +0.0059° | Diff = 0.4410°
   J6: Calc =  +0.4685° | Baseline =  +0.0097° | Diff = 0.4589°
=========================================================

Optimization finished successfully.
