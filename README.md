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
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[WARNING] Auto-centering head encountered a minor issue: list indices must be integers or slices, not tuple
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [ 0.    -0.001]
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0539, -0.0485, 89.0, 0.22, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0541, -0.0482, 89.58, -0.06, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2853877559237685, 'joint5': 0.0, 'joint6': 0.08086111523160816}, 'left': {'joint3': -2.136694958302961, 'joint5': 0.0, 'joint6': 0.0}}

[INFO] === QP JOINT-CAMERA UNIFIED OPTIMIZATION WORKFLOW ===

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.103deg, sigma_pos=0.5046mm
Right arm joint offset (deg): [-0.81027863  1.75844901  1.18856575  2.23538775 -1.95343735 -0.05000001
 -0.03086112]
Left arm joint offset (deg): [-0.8593685  -1.14678704 -1.23840321  2.08669495  2.22200287 -0.05000001
 -0.05      ]
Head joint offset (deg): [0.02077844 6.44233236]
mount_to_cam xi: [-4.66820178e-04 -4.42155310e-04  5.59801815e-06  4.49233873e-04
 -8.02537073e-03 -5.08929379e-03]
mount_to_cam_new: [0.03939006702222422, 0.009340604800347407, 0.06573184339188473, -90.07679538574118, -0.10965122575945299, -89.92466669096318]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_100608.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.8103° | Baseline =  +0.1573° | Diff = 0.9676°
   J1: Calc =  +1.7584° | Baseline =  +2.4059° | Diff = 0.6474°
   J2: Calc =  +1.1886° | Baseline =  +0.0007° | Diff = 1.1879°
   J3: Calc =  +2.2354° | Baseline =  +2.2051° | Diff = 0.0303°
   J4: Calc =  -1.9534° | Baseline =  +0.0020° | Diff = 1.9554°
   J5: Calc =  -0.0500° | Baseline =  +0.0090° | Diff = 0.0590°
   J6: Calc =  -0.0309° | Baseline =  +0.0101° | Diff = 0.0410°
 [LEFT ARM]
   J0: Calc =  -0.8594° | Baseline =  +0.1042° | Diff = 0.9636°
   J1: Calc =  -1.1468° | Baseline =  -1.7528° | Diff = 0.6060°
   J2: Calc =  -1.2384° | Baseline =  -0.0007° | Diff = 1.2378°
   J3: Calc =  +2.0867° | Baseline =  +2.1662° | Diff = 0.0795°
   J4: Calc =  +2.2220° | Baseline =  -0.0002° | Diff = 2.2222°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  -0.0500° | Baseline =  +0.0000° | Diff = 0.0500°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
