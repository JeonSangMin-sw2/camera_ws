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
measurement_noise = sigma_rot=0.06788deg, sigma_pos=0.1111mm
Right arm joint offset (deg): [ 0.3490784   2.99698358  0.01660367  2.61476444 -0.33288481 -0.81171744
  0.68731806]
Left arm joint offset (deg): [ 0.10175353 -2.23247697 -0.21835048  2.45146522  0.56414451 -0.79079222
 -0.62011003]
Head joint offset (deg): [-0.01710249  5.21956912]
mount_to_cam xi: [-4.15770055e-05 -1.93192959e-03  1.39755707e-03  6.71396379e-04
  1.22113788e-03 -5.17031841e-05]
mount_to_cam_new: [0.0444184406866215, 0.009107018360136101, 0.05647972914461327, -90.05251716855635, -0.02982685740901164, -89.83938019565853]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_101345.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +0.3491° | Baseline =  +0.1573° | Diff = 0.1918°
   J1: Calc =  +2.9970° | Baseline =  +2.4059° | Diff = 0.5911°
   J2: Calc =  +0.0166° | Baseline =  +0.0007° | Diff = 0.0160°
   J3: Calc =  +2.6148° | Baseline =  +2.2051° | Diff = 0.4097°
   J4: Calc =  -0.3329° | Baseline =  +0.0020° | Diff = 0.3349°
   J5: Calc =  -0.8117° | Baseline =  +0.0090° | Diff = 0.8207°
   J6: Calc =  +0.6873° | Baseline =  +0.0101° | Diff = 0.6772°
 [LEFT ARM]
   J0: Calc =  +0.1018° | Baseline =  +0.1042° | Diff = 0.0025°
   J1: Calc =  -2.2325° | Baseline =  -1.7528° | Diff = 0.4797°
   J2: Calc =  -0.2184° | Baseline =  -0.0007° | Diff = 0.2177°
   J3: Calc =  +2.4515° | Baseline =  +2.1662° | Diff = 0.2853°
   J4: Calc =  +0.5641° | Baseline =  -0.0002° | Diff = 0.5644°
   J5: Calc =  -0.7908° | Baseline =  -0.0015° | Diff = 0.7893°
   J6: Calc =  -0.6201° | Baseline =  +0.0000° | Diff = 0.6201°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_101345.json
Arm: both
Right move offset (deg): [-0.3490784015178858, -2.9969835837456946, -0.01660366517044512, -2.6147644428258334, 0.3328848086531619, 0.8117174373256428, -0.687318062415413]
Left move offset (deg): [-0.10175352541909115, 2.2324769712566517, 0.21835048038167365, -2.4514652238378773, -0.5641445111361578, 0.790792219286002, 0.6201100277047877]
Head move offset (deg): [0.017102489653405575, -5.219569124324328]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_101345.json
Arm: both
Right move offset (deg): [-0.3490784015178858, -2.9969835837456946, -0.01660366517044512, -2.6147644428258334, 0.3328848086531619, 0.8117174373256428, -0.687318062415413]
Left move offset (deg): [-0.10175352541909115, 2.2324769712566517, 0.21835048038167365, -2.4514652238378773, -0.5641445111361578, 0.790792219286002, 0.6201100277047877]
Head move offset (deg): [0.017102489653405575, -5.219569124324328]
Preview move complete. Inspect the robot pose before applying.
