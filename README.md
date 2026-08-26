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
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.1517°, J5=-0.0614°, J6=0.0000°] L[J3=-2.1003°, J5=0.0000°, J6=-0.1437°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0538, -0.0487, 89.41, 0.13, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0543, -0.0482, 89.16, -0.06, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.1516934165067894, 'joint5': -0.06137254693253481, 'joint6': 0.0}, 'left': {'joint3': -2.1003027088121917, 'joint5': 0.0, 'joint6': -0.1436684682678275}}

[INFO] === SEQUENTIAL 3-STAGE JOINT-CAMERA CALIBRATION WORKFLOW ===
[STAGE 1/3] Right Arm + Head + Camera Alignment (max_iter=100, eps=1e-7)...
[STAGE 2/3] Left Arm + Head + Camera Alignment (max_iter=100, eps=1e-7)...
[STAGE 3/3] Dual-Arm Unified Fine Integration (Warm Start, max_iter=30, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.2078deg, sigma_pos=0.2466mm
Right arm joint offset (deg): [-0.01134632  1.87487699  0.64468182  2.10169356 -1.38944971  0.01137343
  0.02588356]
Left arm joint offset (deg): [-0.23091831 -1.74071267 -0.68672845  2.05030287  1.36838976 -0.04999903
  0.09993976]
Head joint offset (deg): [0.23020137 6.16813623]
mount_to_cam xi: [-0.00166004 -0.00081803  0.00080306  0.00208147 -0.005      -0.00327724]
mount_to_cam_new: [0.04120393770186954, 0.007701948843319377, 0.06270872754874392, -90.14518408528173, -0.06390835958694518, -89.90320890335573]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_131230.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.0113° | Baseline =  +0.1573° | Diff = 0.1686°
   J1: Calc =  +1.8749° | Baseline =  +2.4059° | Diff = 0.5310°
   J2: Calc =  +0.6447° | Baseline =  +0.0007° | Diff = 0.6440°
   J3: Calc =  +2.1017° | Baseline =  +2.2051° | Diff = 0.1034°
   J4: Calc =  -1.3894° | Baseline =  +0.0020° | Diff = 1.3914°
   J5: Calc =  +0.0114° | Baseline =  +0.0090° | Diff = 0.0024°
   J6: Calc =  +0.0259° | Baseline =  +0.0101° | Diff = 0.0158°
 [LEFT ARM]
   J0: Calc =  -0.2309° | Baseline =  +0.1042° | Diff = 0.3351°
   J1: Calc =  -1.7407° | Baseline =  -1.7528° | Diff = 0.0121°
   J2: Calc =  -0.6867° | Baseline =  -0.0007° | Diff = 0.6861°
   J3: Calc =  +2.0503° | Baseline =  +2.1662° | Diff = 0.1159°
   J4: Calc =  +1.3684° | Baseline =  -0.0002° | Diff = 1.3686°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  +0.0999° | Baseline =  +0.0000° | Diff = 0.0999°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_131230.json
Arm: both
Right move offset (deg): [0.011346319386805947, -1.8748769930852187, -0.6446818179892149, -2.1016935604928975, 1.3894497147844194, -0.011373430463931153, -0.025883564542259083]
Left move offset (deg): [0.2309183104395914, 1.7407126736800123, 0.6867284533687322, -2.0503028705176605, -1.3683897628851533, 0.04999902949294646, -0.09993975661714835]
Head move offset (deg): [-0.23020136741439404, -6.168136232508618]
Preview move complete. Inspect the robot pose before applying.
