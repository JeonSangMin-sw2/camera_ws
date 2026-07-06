[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
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

[FULL AUTO] Calibrating J6 (Wrist Roll) first...
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_roll_v13):
       a_cand_t5 = [-0.9639200802153051, 0.249405469140211, 0.093032203675237]
       a_A_t5    = [-0.9639200802153051, 0.249405469140211, 0.093032203675237]
       a_B_t5    = [0.07120901161601469, 0.5783534327647295, -0.8126724945966386]

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_ROLL_V13)
==================================================
  * Camera Circle Normals Angle (Reference): 89.5050 deg
  * Circle Size Error (abs: r_A - r_B)     : 30.0026 mm
  * Estimated Circle Center Distance       : 3.1123 mm
  * Calculated Offset Correction           : 0.904480 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.9045°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: 0.9045° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 0.0006°, Joint 6: 0.9039°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0006°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_pitch_v13):

[ITERATION 2/6] Sweeping physically with staged offset -4.1965°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_pitch_v13):

[ITERATION 3/6] Sweeping physically with staged offset -4.4558°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_pitch_v13):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0067° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -4.4558°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: -4.4558° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 2/6] Sweeping physically with staged offset -2.0070°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 3/6] Sweeping physically with staged offset -1.9022°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0019° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -1.9022°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.9022° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.9039°
  * Joint 5 Change      : 4.4558°
  * Joint 3 Change      : 1.9022°
  * Bracket Pos Change  : 1.5860 mm
  * Bracket Rot Change  : 0.4826°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[INFO] Pass 2: Reusing marker sweep datasets from Pass 1.

[FULL AUTO] Calibrating J6 (Wrist Roll) first...
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_roll_v13):
       a_cand_t5 = [-0.929587847555979, 0.3338855864147468, 0.15616289207338083]
       a_A_t5    = [-0.929587847555979, 0.3338855864147468, 0.15616289207338083]
       a_B_t5    = [0.07121029504852669, 0.5783570715641472, -0.8126697925054466]

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_ROLL_V13)
==================================================
  * Camera Circle Normals Angle (Reference): 89.5050 deg
  * Circle Size Error (abs: r_A - r_B)     : 30.0026 mm
  * Estimated Circle Center Distance       : 3.1123 mm
  * Calculated Offset Correction           : 0.904480 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.9045°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: 0.9045° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: -4.4552°, Joint 6: 0.9039°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -4.4552°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_pitch_v13):

[ITERATION 2/6] Sweeping physically with staged offset -2.9602°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: wrist_pitch_v13):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0224° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -2.9602°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: -2.9602° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.9022°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 2/6] Sweeping physically with staged offset -1.6293°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 3/6] Sweeping physically with staged offset -2.6852°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 4/6] Sweeping physically with staged offset -2.0934°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 5/6] Sweeping physically with staged offset -1.6018°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):

[ITERATION 6/6] Sweeping physically with staged offset -2.3191°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: right, Mode: elbow):
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1783° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for left arm...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Roll) first...
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_roll_v13):
       a_cand_t5 = [-0.9639419653782081, -0.24941449165691554, 0.09278091794286059]
       a_A_t5    = [-0.9639419653782081, -0.24941449165691554, 0.09278091794286059]
       a_B_t5    = [-0.07139654907458584, 0.5782626882029828, 0.812720613872012]

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_ROLL_V13)
==================================================
  * Camera Circle Normals Angle (Reference): 89.4778 deg
  * Circle Size Error (abs: r_A - r_B)     : 30.6713 mm
  * Estimated Circle Center Distance       : 3.2823 mm
  * Calculated Offset Correction           : -3.052459 deg
==================================================
[FULL AUTO] Staging J6 offset: -3.0525°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: -3.0525° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: -0.0006°, Joint 6: -3.0519°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -0.0006°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[ITERATION 2/6] Sweeping physically with staged offset -3.0797°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[ITERATION 3/6] Sweeping physically with staged offset -3.3198°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0386° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -3.3198°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_left_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -3.3198° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 2/6] Sweeping physically with staged offset -2.2056°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 3/6] Sweeping physically with staged offset -2.0108°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0419° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -2.0108°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.0108° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 3.0519°
  * Joint 5 Change      : 3.3198°
  * Joint 3 Change      : 2.0108°
  * Bracket Pos Change  : 1.1722 mm
  * Bracket Rot Change  : 2.5233°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[INFO] Pass 2: Reusing marker sweep datasets from Pass 1.

[FULL AUTO] Calibrating J6 (Wrist Roll) first...
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_roll_v13):
       a_cand_t5 = [-0.9359571917766591, -0.32050133409764775, 0.14581848306431697]
       a_A_t5    = [-0.9359571917766591, -0.32050133409764775, 0.14581848306431697]
       a_B_t5    = [-0.07139596073519362, 0.5782610252627843, 0.812721848760529]

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_ROLL_V13)
==================================================
  * Camera Circle Normals Angle (Reference): 89.4778 deg
  * Circle Size Error (abs: r_A - r_B)     : 30.6713 mm
  * Estimated Circle Center Distance       : 3.2823 mm
  * Calculated Offset Correction           : -3.052459 deg
==================================================
[FULL AUTO] Staging J6 offset: -3.0525°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: -3.0525° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: -3.3192°, Joint 6: -3.0519°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch_v13 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -3.3192°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[ITERATION 2/6] Sweeping physically with staged offset -0.8275°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[ITERATION 3/6] Sweeping physically with staged offset -0.7605°...
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: wrist_pitch_v13):

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0085° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -0.7605°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_left_wrist_pitch_v13_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: -0.7605° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -2.0108°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 2/6] Sweeping physically with staged offset -2.2333°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 3/6] Sweeping physically with staged offset -1.6334°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 4/6] Sweeping physically with staged offset -2.3504°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 5/6] Sweeping physically with staged offset -2.2148°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):

[ITERATION 6/6] Sweeping physically with staged offset -2.2807°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] Dynamically calculated nominal axes from FK (Arm: left, Mode: elbow):
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.6274° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +7.9, Y: -0.0, Z: +1.0 mm
  * Bracket Rot: R: +0.00, P: -2.63, Y: -0.25 deg
  * Joint Offsets: Joint 6: +0.90°, Joint 5: -2.96°, Joint 3: -2.18°
  --- LEFT ARM ---
  * Bracket Pos: X: +7.0, Y: +0.0, Z: +1.1 mm
  * Bracket Rot: R: -0.00, P: -4.16, Y: -0.32 deg
  * Joint Offsets: Joint 6: -3.05°, Joint 5: -0.76°, Joint 3: -1.63°
==================================================

[INFO] Full Auto sequential calibration ended.
