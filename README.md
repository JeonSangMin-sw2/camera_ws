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
[INFO] Additional camera bracket: YES (Head motion: DISABLED)
[INFO] Power is already ON.
[INFO] Servos are ON.
[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.0'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.5604°, J5=-2.8585°, J6=0.0288°] L[J3=-2.1706°, J5=0.0000°, J6=0.0669°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
[Step2] Calculate requested.
[INFO] Headless mode: Camera extrinsics optimization is DISABLED (Locked to CAD nominal).
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0541, -0.0025, 91.31, 0.08, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0542, -0.0028, 91.67, -0.22, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.5604088871374433, 'joint5': -2.8585310194303686, 'joint6': 0.028842740814684156}, 'left': {'joint3': -2.170631020782591, 'joint5': 0.0, 'joint6': 0.06692409316314804}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.6565deg, sigma_pos=0.105mm
Right arm joint offset (deg): [ 1.39199687  4.23374561  0.90807232  2.6104089  -1.48047775  2.90853105
 -0.07884276]
Left arm joint offset (deg): [ 0.84583021 -4.60310807 -1.05249496  2.22063105  0.705126    0.05000009
 -0.01692401]
head_base-to-camera xi: [0. 0. 0. 0. 0. 0.]
head_base_to_cam_new: [0.102, 0.009, 0.044, -90.0, -0.0, -90.0]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260806_110913.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +1.3920° | Baseline =  +0.1564° | Diff = 1.2356°
   J1: Calc =  +4.2337° | Baseline =  +4.8159° | Diff = 0.5822°
   J2: Calc =  +0.9081° | Baseline =  -0.0176° | Diff = 0.9257°
   J3: Calc =  +2.6104° | Baseline =  +2.7185° | Diff = 0.1081°
   J4: Calc =  -1.4805° | Baseline =  -0.0092° | Diff = 1.4712°
   J5: Calc =  +2.9085° | Baseline =  +2.8938° | Diff = 0.0147°
   J6: Calc =  -0.0788° | Baseline =  -0.0013° | Diff = 0.0775°
 [LEFT ARM]
   J0: Calc =  +0.8458° | Baseline =  -0.0309° | Diff = 0.8767°
   J1: Calc =  -4.6031° | Baseline =  -4.3293° | Diff = 0.2738°
   J2: Calc =  -1.0525° | Baseline =  -0.0028° | Diff = 1.0497°
   J3: Calc =  +2.2206° | Baseline =  +2.1455° | Diff = 0.0751°
   J4: Calc =  +0.7051° | Baseline =  +0.0000° | Diff = 0.7051°
   J5: Calc =  +0.0500° | Baseline =  +0.0360° | Diff = 0.0140°
   J6: Calc =  -0.0169° | Baseline =  -0.0004° | Diff = 0.0165°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260806_110913.json
Arm: both
Right move offset (deg): [-1.3919968686771265, -4.233745608192246, -0.9080723218077364, -2.610408898933499, 1.4804777452269526, -2.9085310530695496, 0.07884276261340183]
Left move offset (deg): [-0.8458302124530305, 4.603108067880928, 1.0524949625702595, -2.2206310515044323, -0.7051259982864566, -0.05000008903378959, 0.016924009449129696]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260806_110913.json
Arm: both
Right move offset (deg): [-1.3919968686771265, -4.233745608192246, -0.9080723218077364, -2.610408898933499, 1.4804777452269526, -2.9085310530695496, 0.07884276261340183]
Left move offset (deg): [-0.8458302124530305, 4.603108067880928, 1.0524949625702595, -2.2206310515044323, -0.7051259982864566, -0.05000008903378959, 0.016924009449129696]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'OPTIMIZED' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260806_110913.json
Arm: both
Right move offset (deg): [-1.3919968686771265, -4.233745608192246, -0.9080723218077364, -2.610408898933499, 1.4804777452269526, -2.9085310530695496, 0.07884276261340183]
Left move offset (deg): [-0.8458302124530305, 4.603108067880928, 1.0524949625702595, -2.2206310515044323, -0.7051259982864566, -0.05000008903378959, 0.016924009449129696]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'OPTIMIZED' Zero Pose. Now resetting and applying home offset...
[APPLY] Saving optimized mount_to_cam to setting.yaml: [0.102, 0.009, 0.044, -90.0, -0.0, -90.0]
[APPLY] Saving optimized head_base_to_cam to setting.yaml: [0.102, 0.009, 0.044, -90.0, -0.0, -90.0]
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.5604°, J5=-2.8585°, J6=0.0288°] L[J3=-2.1706°, J5=0.0000°, J6=0.0669°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (^(?!head_joint_).*$) on...
[INFO] Turning servos (^(?!head_joint_).*$) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.0'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.5604°, J5=-2.8585°, J6=0.0288°] L[J3=-2.1706°, J5=0.0000°, J6=0.0669°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Current pose home offset apply complete.
[SUCCESS] Saved offsets permanently to setting.yaml!
[INFO] Zeroed out applied arm offsets in baseline json: /home/nvidia/camera_ws/config/home_reset_baseline.json
[Step2] Check Calibration State requested.
[Check State] Step 1: Moving to Joint Ready Pose...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
