[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Marker Bracket Sweeps for right arm (Pass 1/Pass 2)...
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

[FULL AUTO] Calibrating J6 (Wrist Roll)...
[INFO] Moving right arm to wrist_roll_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_roll_v13: J7 nominal ready pose=-0.00°, raw_diff=-0.21°, optimal_offset=-0.17°

[ITERATION 2/6] Sweeping physically with staged offset -0.1705°...
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_roll_v13: J7 nominal ready pose=-0.00°, raw_diff=0.03°, optimal_offset=-0.15°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0179° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1705°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1705°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -0.1705° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J5 (Wrist Pitch)...
[INFO] Moving right arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -0.6449°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset 0.4710°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset 0.3895°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset 0.0615°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 6/6] Sweeping physically with staged offset 0.1533°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[INFO] Joint wrist_pitch_v13 did not meet 0.06° convergence tolerance due to measurement noise floor.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 0.1903° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 & J5 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.1095°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.4094°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0297° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.4094°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.4094° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.1705°
  * Joint 5 Change      : 0.1903°
  * Joint 3 Change      : 2.4094°
  * Bracket Pos Change  : 30.4626 mm
  * Bracket Rot Change  : 4.1086°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Marker Bracket Sweeps for right arm (Pass 2/Pass 2)...
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
[FULL AUTO] J6 (Wrist Roll) converged in Pass 1 (-0.1705°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -0.1705° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J5 (Wrist Pitch)...
[INFO] Moving right arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.1903°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset 2.3508°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset 0.5880°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset 1.5119°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset 1.1604°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 6/6] Sweeping physically with staged offset 0.9612°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[INFO] Joint wrist_pitch_v13 did not meet 0.06° convergence tolerance due to measurement noise floor.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 0.9126° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 & J5 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-2.4094°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.4094° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Marker Bracket Sweeps for left arm (Pass 1/Pass 2)...
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

[FULL AUTO] Calibrating J6 (Wrist Roll)...
[INFO] Moving left arm to wrist_roll_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_roll_v13: J7 nominal ready pose=-0.00°, raw_diff=-0.48°, optimal_offset=-0.38°

[ITERATION 2/6] Sweeping physically with staged offset -0.3844°...
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_roll_v13: J7 nominal ready pose=-0.00°, raw_diff=-0.30°, optimal_offset=-0.63°

[ITERATION 3/6] Sweeping physically with staged offset -0.6263°...
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_roll_v13: J7 nominal ready pose=-0.00°, raw_diff=-0.07°, optimal_offset=-0.68°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0566° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.6263°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_roll_v13_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.6263°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_roll_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: -0.6263° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J5 (Wrist Pitch)...
[INFO] Moving left arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.7512°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0027° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.7512°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_v13_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -1.7512° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 & J5 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.1073°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.3187°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.2258°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0268° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2258°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2258° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.6263°
  * Joint 5 Change      : 1.7512°
  * Joint 3 Change      : 2.2258°
  * Bracket Pos Change  : 31.5601 mm
  * Bracket Rot Change  : 2.4801°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Marker Bracket Sweeps for left arm (Pass 2/Pass 2)...
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
[FULL AUTO] J6 (Wrist Roll) converged in Pass 1 (-0.6263°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: -0.6263° (click APPLY OFFSET to save).
[FULL AUTO] J5 (Wrist Pitch) converged in Pass 1 (-1.7512°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -1.7512° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 & J5 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-2.2258°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.2258° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: -30.0, Y: +0.0, Z: +5.0 mm
  * Bracket Rot: R: +1.63, P: +3.79, Y: +0.70 deg
  * Joint Offsets: Joint 6: -0.17°, Joint 5: +0.91°, Joint 3: -2.41°
  --- LEFT ARM ---
  * Bracket Pos: X: -27.5, Y: -0.0, Z: +5.0 mm
  * Bracket Rot: R: +0.12, P: +1.93, Y: +0.69 deg
  * Joint Offsets: Joint 6: -0.63°, Joint 5: -1.75°, Joint 3: -2.23°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Roll) : -0.6263°
    * Joint 5 (Wrist Pitch): -1.7512°
    * Joint 3 (Elbow)      : -2.2258°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Roll) : -0.1705°
    * Joint 5 (Wrist Pitch): 0.9126°
    * Joint 3 (Elbow)      : -2.4094°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [-0.006 -0.032]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 1] Captured marker: R_pos=[113.6 -39.7 178.9]mm, L_pos=[-101.9  -37.3  184.5]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 2] Captured marker: R_pos=[107.2 -37.1 168.2]mm, L_pos=[-94.5 -34.7 173.9]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 3] Captured marker: R_pos=[117.1 -41.2 184.3]mm, L_pos=[-105.8  -38.7  190. ]mm
Auto motion done: Joint 0 Offset: 5.0deg
[Sample 4] Captured marker: R_pos=[120.8 -42.8 190.1]mm, L_pos=[-109.9  -40.2  195.6]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[112.8 -40.3 182.2]mm, L_pos=[-102.8  -36.7  181.1]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[111.9 -41.  185.5]mm, L_pos=[-103.5  -36.1  177.7]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[114.3 -39.  175.5]mm, L_pos=[-101.1  -37.8  187.8]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[115.  -38.3 172.2]mm, L_pos=[-100.1  -38.3  191.2]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[123.8 -40.  178.8]mm, L_pos=[-112.   -37.7  184.2]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[134.3 -40.3 178.9]mm, L_pos=[-122.4  -38.   183.8]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[103.8 -39.3 178.9]mm, L_pos=[-92.  -36.8 184.8]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 94.3 -38.9 178.9]mm, L_pos=[-82.5 -36.3 185.1]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[116.2 -42.3 174.6]mm, L_pos=[-99.9 -34.6 188.9]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[118.9 -44.9 170.4]mm, L_pos=[-97.9 -31.9 193.6]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[111.3 -37.1 183.3]mm, L_pos=[-104.2  -40.   180.1]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[109.1 -34.5 188. ]mm, L_pos=[-106.7  -42.6  176. ]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[110.1 -33.3 181. ]mm, L_pos=[-104.4  -43.8  182.4]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[120.7 -43.4 164.1]mm, L_pos=[-96.6 -32.8 200.6]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[107.9 -35.6 194.9]mm, L_pos=[-108.8  -41.3  169.7]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[116.7 -46.3 176.6]mm, L_pos=[-99.  -30.9 186.5]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 95.5 -37.5 170.8]mm, L_pos=[-80.7 -37.3 193.2]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[132.5 -41.7 183.8]mm, L_pos=[-124.1  -36.6  178.7]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[113.6 -39.6 178.9]mm, L_pos=[-101.9  -37.3  184.5]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[117.1 -41.  184.4]mm, L_pos=[-105.6  -38.6  189.9]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[117.2 -41.  184.6]mm, L_pos=[-105.2  -38.7  189.7]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[117.1 -41.2 184.2]mm, L_pos=[-106.1  -38.7  190.1]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[117.1 -41.3 184.1]mm, L_pos=[-106.3  -38.6  190.3]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[119.8 -40.8 184.5]mm, L_pos=[-108.3  -38.3  189.9]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[122.3 -40.5 184.6]mm, L_pos=[-110.7  -38.   190. ]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[114.6 -41.5 184.4]mm, L_pos=[-103.5  -39.1  190.2]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[112.  -41.8 184.5]mm, L_pos=[-101.1  -39.4  190.6]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[117.2 -44.1 184.3]mm, L_pos=[-105.8  -35.8  189.8]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[117.4 -47.  184.6]mm, L_pos=[-105.7  -33.   189.7]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[117.  -38.3 184.4]mm, L_pos=[-105.9  -41.6  190.2]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[117.  -35.4 184.7]mm, L_pos=[-106.1  -44.5  190.7]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[120.4 -41.2 191. ]mm, L_pos=[-102.6  -39.   184.5]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[123.2 -41.2 198.4]mm, L_pos=[-99.  -39.4 179.7]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[113.7 -41.5 178.6]mm, L_pos=[-108.8  -38.6  196.5]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[109.7 -41.8 173.8]mm, L_pos=[-111.5  -38.6  203.8]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[117.3 -37.1 179.9]mm, L_pos=[-106.2  -34.9  185.2]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[117.5 -33.2 176.4]mm, L_pos=[-106.3  -31.1  181.3]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[117.  -45.4 189.5]mm, L_pos=[-105.6  -42.6  195.4]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[116.8 -49.6 195.4]mm, L_pos=[-105.3  -46.6  201.7]mm
Auto motion done: Head Pan: -3.50deg
[Sample 44] Captured marker: R_pos=[103.1 -41.3 190.5]mm, L_pos=[-120.   -38.8  182.5]mm
Auto motion done: Head Pan: -1.75deg
[Sample 45] Captured marker: R_pos=[110.2 -41.3 187.5]mm, L_pos=[-113.   -38.7  186.3]mm
Auto motion done: Head Pan: +1.75deg
[Sample 46] Captured marker: R_pos=[124.2 -41.2 180.9]mm, L_pos=[-98.7 -38.7 193.4]mm
Auto motion done: Head Pan: +3.50deg
[Sample 47] Captured marker: R_pos=[131.  -41.2 177.3]mm, L_pos=[-91.4 -38.6 196.6]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 48] Captured marker: R_pos=[117.2 -27.1 189.8]mm, L_pos=[-105.9  -24.2  195.2]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 49] Captured marker: R_pos=[117.2 -34.2 187.2]mm, L_pos=[-106.   -31.5  192.8]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 50] Captured marker: R_pos=[117.3 -48.2 181.1]mm, L_pos=[-105.9  -45.8  187. ]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 51] Captured marker: R_pos=[117.3 -55.1 177.9]mm, L_pos=[-105.8  -52.9  183.7]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260824_090819.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0670000000000002), np.float64(1.5146129380243426e-30), np.float64(3.250103350797004e-05), np.float64(86.20654709719416), np.float64(1.6228736139674564), np.float64(-89.40327111916757)]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.06949019335096633), np.float64(-1.0169433757533666e-16), np.float64(3.163099964570418e-05), np.float64(88.06864595471906), np.float64(0.1198939579220563), np.float64(-89.31154644338253)]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.4094043044548377, 'joint5': 0.9125509208996533, 'joint6': -0.17049103687846928}, 'left': {'joint3': -2.225831250196081, 'joint5': -1.7511664980725214, 'joint6': -0.6262597057707002}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.1239deg, sigma_pos=0.41mm
Right arm joint offset (deg): [ 1.07938428  1.98181249  0.35525639  2.45940425 -4.08696062 -0.96255102
  0.22049109]
Left arm joint offset (deg): [ 0.52726032 -2.66686755 -1.07824504  2.17583131 -1.06703897  1.70116662
  0.57625981]
Head joint offset (deg): [-0.29797387  2.85699734]
mount_to_cam xi: [-0.0001082  -0.00065772  0.00039608  0.00838724 -0.01384929 -0.01395655]
mount_to_cam_new: [0.03304695918567131, 0.0006054244735286818, 0.07084838438045041, -90.00619194220225, 0.022695757109106898, -89.9623166928681]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260824_090819.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +1.0794° | Baseline =  -0.4286° | Diff = 1.5080°
   J1: Calc =  +1.9818° | Baseline =  +2.8390° | Diff = 0.8572°
   J2: Calc =  +0.3553° | Baseline =  -0.0002° | Diff = 0.3555°
   J3: Calc =  +2.4594° | Baseline =  +0.2621° | Diff = 2.1973°
   J4: Calc =  -4.0870° | Baseline =  +0.0664° | Diff = 4.1533°
   J5: Calc =  -0.9626° | Baseline =  +1.3164° | Diff = 2.2789°
   J6: Calc =  +0.2205° | Baseline =  +0.0136° | Diff = 0.2069°
 [LEFT ARM]
   J0: Calc =  +0.5273° | Baseline =  +1.8200° | Diff = 1.2928°
   J1: Calc =  -2.6669° | Baseline =  -1.9092° | Diff = 0.7576°
   J2: Calc =  -1.0782° | Baseline =  -0.0128° | Diff = 1.0654°
   J3: Calc =  +2.1758° | Baseline =  +0.2319° | Diff = 1.9439°
   J4: Calc =  -1.0670° | Baseline =  -0.7561° | Diff = 0.3110°
   J5: Calc =  +1.7012° | Baseline =  +0.5636° | Diff = 1.1376°
   J6: Calc =  +0.5763° | Baseline =  -0.0578° | Diff = 0.6340°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [0.4285755724009901, -2.839041228341585, 0.00021755105198019802, -0.2621490176361386, -0.066357421875, -1.3163818359375, -0.013623046875]
Left move offset (deg): [-1.8200321008663367, 1.909228032178218, 0.012835512066831683, -0.2319094214108911, 0.7560791015624999, -0.5635986328125, 0.0577880859375]
Head move offset (deg): [-0.0, -0.0]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260824_090819.json
Arm: both
Right move offset (deg): [-1.079384283663843, -1.9818124932414398, -0.3552563883360053, -2.459404252934418, 4.086960623236188, 0.9625510168448301, -0.22049108998956696]
Left move offset (deg): [-0.5272603188504041, 2.6668675503726424, 1.07824504151197, -2.175831309743437, 1.0670389676090297, -1.7011666235479064, -0.5762598077260497]
Head move offset (deg): [0.29797387447622403, -2.8569973356739586]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [0.4285755724009901, -2.839041228341585, 0.00021755105198019802, -0.2621490176361386, -0.066357421875, -1.3163818359375, -0.013623046875]
Left move offset (deg): [-1.8200321008663367, 1.909228032178218, 0.012835512066831683, -0.2319094214108911, 0.7560791015624999, -0.5635986328125, 0.0577880859375]
Head move offset (deg): [-0.0, -0.0]
Preview move complete. Inspect the robot pose before applying.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260824_090819.json
Arm: both
Right move offset (deg): [-1.079384283663843, -1.9818124932414398, -0.3552563883360053, -2.459404252934418, 4.086960623236188, 0.9625510168448301, -0.22049108998956696]
Left move offset (deg): [-0.5272603188504041, 2.6668675503726424, 1.07824504151197, -2.175831309743437, 1.0670389676090297, -1.7011666235479064, -0.5762598077260497]
Head move offset (deg): [0.29797387447622403, -2.8569973356739586]
Preview move complete. Inspect the robot pose before applying.
