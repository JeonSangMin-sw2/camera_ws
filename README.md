============================================================
  UNIFIED ROBOT CALIBRATION SUITE LOADED
============================================================
[RECOMMENDED SEQUENCE]
  1. Calibrate camera intrinsics first if needed (Step 1 > Camera tab).
  2. Calibrate joint offsets using Joint subtab.
  3. Perform marker bracket sweeps using Marker subtab.
  4. Control head and verify offsets as a final check.
============================================================
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[Camera] Initialized camera exposure to AUTO mode.
[INFO] Power is already ON.
[INFO] Servos are ON.
[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.2854°, J5=0.0000°, J6=0.0809°] L[J3=-2.1367°, J5=0.0000°, J6=0.0000°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0539, -0.0485, 89.0, 0.22, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0541, -0.0482, 89.58, -0.06, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2853877559237685, 'joint5': 0.0, 'joint6': 0.08086111523160816}, 'left': {'joint3': -2.136694958302961, 'joint5': 0.0, 'joint6': 0.0}}

[INFO] === JOINT-CAMERA UNIFIED CALIBRATION WORKFLOW ===

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.09617deg, sigma_pos=0.4351mm
Right arm joint offset (deg): [-0.50614912  1.96460907  0.94599422  2.21104787 -1.63736875 -0.13224211
 -0.05362075]
Left arm joint offset (deg): [-0.56307438 -1.33526672 -1.00361583  2.06226868  1.91727327 -0.13438464
 -0.02839406]
Head joint offset (deg): [0.01873948 6.37649682]
mount_to_cam xi: [-5.46079455e-04 -5.28060600e-04  8.75371373e-05  4.43611463e-04
 -6.75666021e-03 -4.44744673e-03]
mount_to_cam_new: [0.040030793512076786, 0.009344027158011917, 0.0644625735994593, -90.0813447912942, -0.10494981572696284, -89.91975007209591]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_102324.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.5061° | Baseline =  +0.1573° | Diff = 0.6634°
   J1: Calc =  +1.9646° | Baseline =  +2.4059° | Diff = 0.4413°
   J2: Calc =  +0.9460° | Baseline =  +0.0007° | Diff = 0.9453°
   J3: Calc =  +2.2110° | Baseline =  +2.2051° | Diff = 0.0060°
   J4: Calc =  -1.6374° | Baseline =  +0.0020° | Diff = 1.6393°
   J5: Calc =  -0.1322° | Baseline =  +0.0090° | Diff = 0.1413°
   J6: Calc =  -0.0536° | Baseline =  +0.0101° | Diff = 0.0637°
 [LEFT ARM]
   J0: Calc =  -0.5631° | Baseline =  +0.1042° | Diff = 0.6673°
   J1: Calc =  -1.3353° | Baseline =  -1.7528° | Diff = 0.4175°
   J2: Calc =  -1.0036° | Baseline =  -0.0007° | Diff = 1.0030°
   J3: Calc =  +2.0623° | Baseline =  +2.1662° | Diff = 0.1039°
   J4: Calc =  +1.9173° | Baseline =  -0.0002° | Diff = 1.9175°
   J5: Calc =  -0.1344° | Baseline =  -0.0015° | Diff = 0.1328°
   J6: Calc =  -0.0284° | Baseline =  +0.0000° | Diff = 0.0284°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
