[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial arm & head joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/3 FOR RIGHT ARM
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

[ITERATION 2/6] Sweeping physically with staged offset -0.2208°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0540° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2208°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2208° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) against nominal bracket reference...
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-2.89°, optimal_offset=-2.88°

[ITERATION 2/6] Sweeping physically with staged offset -2.8818°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[ERROR] Marker is not visible.
[INFO] Prompting user for manual teaching due to marker visibility error...
[INFO] Preserved user-taught ready pose for right arm (wrist_yaw2).
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Preserved user-taught ready pose detected for right arm (wrist_yaw2). Using taught posture.
[INFO] Ready Pose Reached.
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=4.24°, optimal_offset=1.38°

[ITERATION 3/6] Sweeping physically with staged offset 0.5239°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=3.00°, optimal_offset=3.53°

[ITERATION 4/6] Sweeping physically with staged offset 2.9285°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-7.07°, optimal_offset=-4.13°

[ITERATION 5/6] Sweeping physically with staged offset -1.5919°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.83°, optimal_offset=1.25°

[ITERATION 6/6] Sweeping physically with staged offset -0.1348°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.55°, optimal_offset=2.42°

[INFO] Joint wrist_yaw2 did not meet 0.06° convergence tolerance; cause is not determined. Fallback remains unconverged.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1838°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.1838° (click APPLY OFFSET to save).
[FULL AUTO] Pending joints: wrist_yaw2. Bracket and elbow deferred.

==================================================
   STARTING PASS 2/3 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) previously converged (-0.2208°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2208° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) against nominal bracket reference...
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Preserved user-taught ready pose detected for right arm (wrist_yaw2). Using taught posture.
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -0.1838°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.63°, optimal_offset=2.46°

[ITERATION 2/6] Sweeping physically with staged offset 2.4644°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-5.94°, optimal_offset=-3.46°

[ITERATION 3/6] Sweeping physically with staged offset -2.2764°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=3.42°, optimal_offset=1.16°

[ITERATION 4/6] Sweeping physically with staged offset -0.0767°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.68°, optimal_offset=2.61°

[ITERATION 5/6] Sweeping physically with staged offset 1.6452°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-6.18°, optimal_offset=-4.53°

[ITERATION 6/6] Sweeping physically with staged offset -1.5141°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=3.46°, optimal_offset=1.95°

[INFO] Joint wrist_yaw2 did not meet 0.06° convergence tolerance; cause is not determined. Fallback remains unconverged.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0124°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0124° (click APPLY OFFSET to save).
[FULL AUTO] Pending joints: wrist_yaw2. Bracket and elbow deferred.

==================================================
   STARTING PASS 3/3 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) previously converged (-0.2208°). Skipping Pass 3 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2208° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) against nominal bracket reference...
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Preserved user-taught ready pose detected for right arm (wrist_yaw2). Using taught posture.
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0124°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=3.36°, optimal_offset=3.39°

[ITERATION 2/6] Sweeping physically with staged offset 3.3852°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-6.31°, optimal_offset=-2.91°

[ITERATION 3/6] Sweeping physically with staged offset -1.6497°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=3.43°, optimal_offset=1.80°

[ITERATION 4/6] Sweeping physically with staged offset 0.5561°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.94°, optimal_offset=3.51°

[ITERATION 5/6] Sweeping physically with staged offset 2.4483°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-6.24°, optimal_offset=-3.79°

[ITERATION 6/6] Sweeping physically with staged offset -0.7433°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=2.95°, optimal_offset=2.21°

[INFO] Joint wrist_yaw2 did not meet 0.06° convergence tolerance; cause is not determined. Fallback remains unconverged.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.7243°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.7243° (click APPLY OFFSET to save).
[FULL AUTO] Pending joints: wrist_yaw2. Bracket and elbow deferred.
[ERROR] Full Auto sequential calibration failed: right: J5/J6 prerequisites did not converge after 3 passes; bracket was not fitted
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 2283, in run
    raise RuntimeError(f"{arm_side}: J5/J6 prerequisites did not converge after 3 passes; bracket was not fitted")
RuntimeError: right: J5/J6 prerequisites did not converge after 3 passes; bracket was not fitted

[INFO] Full Auto sequential calibration ended.
[ERROR] Full Auto Calibration FAILED: right: J5/J6 prerequisites did not converge after 3 passes; bracket was not fitted



t B motion... downsampled to 180 for optimization.
2026-09-08 09:37:48,640 [INFO] Loaded config from setting.yaml successfully.
2026-09-08 09:37:48,651 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
DEBUG SOLVER v1.2: arm_side=right
  L_5_ee = 126.1000
  radius_6 = 54.1764, radius_5 = 174.1961, radius_4 = 0.0000
  x_nom = 0.0000, y_nom = -54.0000, z_nom = -48.0000
  x_e = 0.0000, y_e = -54.1764, z_e = -48.0961
  Initial guess: [0.0, -54.0, -48.0]
  Lower bounds: [-40.0, -94.0, -250.0]
  Upper bounds: [40.0, -14.0, 10.0]
  Optimal residuals: [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(-1.7636509948876265e-08), np.float64(-9.613717374075036e-09)]
[VALIDATION] RIGHT ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):
  * J6 Sweep Radius Err: 0.0000 mm
  * J5 Sweep Radius Err: 0.0000 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.0000 mm < 1.0 mm)
2026-09-08 09:37:51,680 [INFO] 
--- Commencing Continuous Sweep on Joint A (Index 6, duration=12.0s) ---
2026-09-08 09:37:51,684 [INFO] [INFO] Moving Joint A to start sweep position...
2026-09-08 09:37:53,757 [INFO] [INFO] Commencing continuous sweep on Joint A (duration=12.0s)...
2026-09-08 09:38:06,157 [INFO]     -> Swept 176 dense raw coordinate frames during Joint A motion.
2026-09-08 09:38:06,658 [INFO] 
--- Commencing Continuous Sweep on Joint B (Index 5, duration=12.0s) ---
2026-09-08 09:38:06,663 [INFO] [INFO] Moving Joint B to start sweep position...
2026-09-08 09:38:08,737 [INFO] [INFO] Commencing continuous sweep on Joint B (duration=12.0s)...
2026-09-08 09:38:21,148 [INFO]     -> Swept 175 dense raw coordinate frames during Joint B motion.
2026-09-08 09:38:21,149 [INFO] Swept 176 dense raw coordinate frames during Joint A motion... downsampled to 176 for optimization.
2026-09-08 09:38:21,149 [INFO] Swept 175 dense raw coordinate frames during Joint B motion... downsampled to 175 for optimization.
2026-09-08 09:38:22,002 [INFO] Loaded config from setting.yaml successfully.
2026-09-08 09:38:22,013 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
DEBUG SOLVER v1.2: arm_side=right
  L_5_ee = 126.1000
  radius_6 = 54.2234, radius_5 = 174.0092, radius_4 = 0.0000
  x_nom = 0.0000, y_nom = -54.0000, z_nom = -48.0000
  x_e = 0.0000, y_e = -54.2234, z_e = -47.9092
  Initial guess: [0.0, -54.0, -48.0]
  Lower bounds: [-40.0, -94.0, -250.0]
  Upper bounds: [40.0, -14.0, 10.0]
  Optimal residuals: [np.float64(0.0), np.float64(-2.842170943040401e-14), np.float64(0.0), np.float64(-2.2344668689096636e-08), np.float64(9.077381059121591e-09)]
[VALIDATION] RIGHT ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):
  * J6 Sweep Radius Err: 0.0000 mm
  * J5 Sweep Radius Err: 0.0000 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.0000 mm < 1.0 mm)
2026-09-08 09:38:25,062 [INFO] 
--- Commencing Continuous Sweep on Joint A (Index 6, duration=12.0s) ---
2026-09-08 09:38:25,067 [INFO] [INFO] Moving Joint A to start sweep position...
2026-09-08 09:38:27,138 [INFO] [INFO] Commencing continuous sweep on Joint A (duration=12.0s)...
2026-09-08 09:38:39,559 [INFO]     -> Swept 181 dense raw coordinate frames during Joint A motion.
2026-09-08 09:38:40,060 [INFO] 
--- Commencing Continuous Sweep on Joint B (Index 5, duration=12.0s) ---
2026-09-08 09:38:40,065 [INFO] [INFO] Moving Joint B to start sweep position...
2026-09-08 09:38:42,138 [INFO] [INFO] Commencing continuous sweep on Joint B (duration=12.0s)...
2026-09-08 09:38:54,559 [INFO]     -> Swept 177 dense raw coordinate frames during Joint B motion.
2026-09-08 09:38:54,560 [INFO] Swept 181 dense raw coordinate frames during Joint A motion... downsampled to 181 for optimization.
2026-09-08 09:38:54,560 [INFO] Swept 177 dense raw coordinate frames during Joint B motion... downsampled to 177 for optimization.
2026-09-08 09:38:55,400 [INFO] Loaded config from setting.yaml successfully.
2026-09-08 09:38:55,410 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
DEBUG SOLVER v1.2: arm_side=right
  L_5_ee = 126.1000
  radius_6 = 54.1590, radius_5 = 174.1108, radius_4 = 0.0000
  x_nom = 0.0000, y_nom = -54.0000, z_nom = -48.0000
  x_e = 0.0000, y_e = -54.1590, z_e = -48.0108
  Initial guess: [0.0, -54.0, -48.0]
  Lower bounds: [-40.0, -94.0, -250.0]
  Upper bounds: [40.0, -14.0, 10.0]
  Optimal residuals: [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(-1.589549270084518e-08), np.float64(-1.0807047345167575e-09)]
[VALIDATION] RIGHT ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):
  * J6 Sweep Radius Err: 0.0000 mm
  * J5 Sweep Radius Err: 0.0000 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.0000 mm < 1.0 mm)
2026-09-08 09:38:58,494 [INFO] 
--- Commencing Continuous Sweep on Joint A (Index 6, duration=12.0s) ---
2026-09-08 09:38:58,498 [INFO] [INFO] Moving Joint A to start sweep position...
2026-09-08 09:39:00,572 [INFO] [INFO] Commencing continuous sweep on Joint A (duration=12.0s)...
2026-09-08 09:39:13,022 [INFO]     -> Swept 176 dense raw coordinate frames during Joint A motion.
2026-09-08 09:39:13,523 [INFO] 
--- Commencing Continuous Sweep on Joint B (Index 5, duration=12.0s) ---
2026-09-08 09:39:13,527 [INFO] [INFO] Moving Joint B to start sweep position...
2026-09-08 09:39:15,599 [INFO] [INFO] Commencing continuous sweep on Joint B (duration=12.0s)...
2026-09-08 09:39:27,991 [INFO]     -> Swept 179 dense raw coordinate frames during Joint B motion.
2026-09-08 09:39:27,992 [INFO] Swept 176 dense raw coordinate frames during Joint A motion... downsampled to 176 for optimization.
2026-09-08 09:39:27,992 [INFO] Swept 179 dense raw coordinate frames during Joint B motion... downsampled to 179 for optimization.
2026-09-08 09:39:28,728 [INFO] Loaded config from setting.yaml successfully.
2026-09-08 09:39:28,741 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
DEBUG SOLVER v1.2: arm_side=right
  L_5_ee = 126.1000
  radius_6 = 54.1974, radius_5 = 174.2420, radius_4 = 0.0000
  x_nom = 0.0000, y_nom = -54.0000, z_nom = -48.0000
  x_e = 0.0000, y_e = -54.1974, z_e = -48.1420
  Initial guess: [0.0, -54.0, -48.0]
  Lower bounds: [-40.0, -94.0, -250.0]
  Upper bounds: [40.0, -14.0, 10.0]
  Optimal residuals: [np.float64(0.0), np.float64(-2.842170943040401e-14), np.float64(0.0), np.float64(-1.9741022184678767e-08), np.float64(-1.4203352036700777e-08)]
[VALIDATION] RIGHT ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):
  * J6 Sweep Radius Err: 0.0000 mm
  * J5 Sweep Radius Err: 0.0000 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.0000 mm < 1.0 mm)
2026-09-08 09:39:31,798 [INFO] 
--- Commencing Continuous Sweep on Joint A (Index 6, duration=12.0s) ---
2026-09-08 09:39:31,802 [INFO] [INFO] Moving Joint A to start sweep position...
2026-09-08 09:39:33,875 [INFO] [INFO] Commencing continuous sweep on Joint A (duration=12.0s)...
2026-09-08 09:39:46,292 [INFO]     -> Swept 180 dense raw coordinate frames during Joint A motion.
2026-09-08 09:39:46,794 [INFO] 
--- Commencing Continuous Sweep on Joint B (Index 5, duration=12.0s) ---
2026-09-08 09:39:46,798 [INFO] [INFO] Moving Joint B to start sweep position...
2026-09-08 09:39:48,871 [INFO] [INFO] Commencing continuous sweep on Joint B (duration=12.0s)...
2026-09-08 09:40:01,331 [INFO]     -> Swept 180 dense raw coordinate frames during Joint B motion.
2026-09-08 09:40:01,332 [INFO] Swept 180 dense raw coordinate frames during Joint A motion... downsampled to 180 for optimization.
2026-09-08 09:40:01,332 [INFO] Swept 180 dense raw coordinate frames during Joint B motion... downsampled to 180 for optimization.
2026-09-08 09:40:02,465 [INFO] Loaded config from setting.yaml successfully.
2026-09-08 09:40:02,476 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
DEBUG SOLVER v1.2: arm_side=right
  L_5_ee = 126.1000
  radius_6 = 54.2429, radius_5 = 174.0223, radius_4 = 0.0000
  x_nom = 0.0000, y_nom = -54.0000, z_nom = -48.0000
  x_e = 0.0000, y_e = -54.2429, z_e = -47.9223
  Initial guess: [0.0, -54.0, -48.0]
  Lower bounds: [-40.0, -94.0, -250.0]
  Upper bounds: [40.0, -14.0, 10.0]
  Optimal residuals: [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(-2.4290849282528624e-08), np.float64(7.773534686153028e-09)]
[VALIDATION] RIGHT ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):
  * J6 Sweep Radius Err: 0.0000 mm
  * J5 Sweep Radius Err: 0.0000 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.0000 mm < 1.0 mm)




