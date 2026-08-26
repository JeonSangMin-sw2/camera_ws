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

[INFO] === 3-STAGE JOINT-CAMERA DECOUPLED CALIBRATION WORKFLOW ===
[STAGE 1/3] Coarse Multi-DOF Initial Alignment (eps=1e-5)...
[STAGE 2/3] Joint Priority Decoupling (Camera Extrinsics Locked, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.2068deg, sigma_pos=0.2239mm
Right arm joint offset (deg): [-9.74703874e-04  1.86122378e+00  6.69464154e-01  2.10169359e+00
 -1.40289683e+00  1.13735079e-02  2.57946664e-02]
Left arm joint offset (deg): [-0.18440563 -1.76109633 -0.67743403  2.05030286  1.36434131 -0.04999905
  0.10161785]
Head joint offset (deg): [0.22649612 6.20484059]
mount_to_cam xi: [-0.00167218 -0.00074532  0.00061789  0.00204909 -0.005      -0.00316919]
mount_to_cam_new: [0.04131190671229544, 0.007735043536300156, 0.06270870265389582, -90.14587753592645, -0.07452447805488917, -89.90735676775903]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_115154.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.0010° | Baseline =  +0.1573° | Diff = 0.1583°
   J1: Calc =  +1.8612° | Baseline =  +2.4059° | Diff = 0.5447°
   J2: Calc =  +0.6695° | Baseline =  +0.0007° | Diff = 0.6688°
   J3: Calc =  +2.1017° | Baseline =  +2.2051° | Diff = 0.1034°
   J4: Calc =  -1.4029° | Baseline =  +0.0020° | Diff = 1.4049°
   J5: Calc =  +0.0114° | Baseline =  +0.0090° | Diff = 0.0024°
   J6: Calc =  +0.0258° | Baseline =  +0.0101° | Diff = 0.0157°
 [LEFT ARM]
   J0: Calc =  -0.1844° | Baseline =  +0.1042° | Diff = 0.2886°
   J1: Calc =  -1.7611° | Baseline =  -1.7528° | Diff = 0.0083°
   J2: Calc =  -0.6774° | Baseline =  -0.0007° | Diff = 0.6768°
   J3: Calc =  +2.0503° | Baseline =  +2.1662° | Diff = 0.1159°
   J4: Calc =  +1.3643° | Baseline =  -0.0002° | Diff = 1.3646°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  +0.1016° | Baseline =  +0.0000° | Diff = 0.1016°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_115154.json
Arm: both
Right move offset (deg): [0.0009747038742268368, -1.8612237816565398, -0.6694641541609523, -2.1016935858420256, 1.4028968276819993, -0.01137350786870807, -0.025794666425745093]
Left move offset (deg): [0.18440562669273602, 1.7610963287576904, 0.6774340291019244, -2.0503028641310674, -1.3643413119092542, 0.04999905366207348, -0.10161785487709915]
Head move offset (deg): [-0.22649611789790158, -6.204840588725515]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.1572894105816832, -2.40589708384901, -0.0006526531559405941, -2.205097462871287, -0.0019775390625, -0.009008789062499998, -0.010107421874999998]
Left move offset (deg): [-0.10420695389851488, 1.7528088258044554, 0.0006526531559405941, -2.166155824566832, 0.00021972656249999998, 0.0015380859374999997, -0.0]
Head move offset (deg): [-0.0, -5.0000976562499995]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_115154.json
Arm: both
Right move offset (deg): [0.0009747038742268368, -1.8612237816565398, -0.6694641541609523, -2.1016935858420256, 1.4028968276819993, -0.01137350786870807, -0.025794666425745093]
Left move offset (deg): [0.18440562669273602, 1.7610963287576904, 0.6774340291019244, -2.0503028641310674, -1.3643413119092542, 0.04999905366207348, -0.10161785487709915]
Head move offset (deg): [-0.22649611789790158, -6.204840588725515]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.1572894105816832, -2.40589708384901, -0.0006526531559405941, -2.205097462871287, -0.0019775390625, -0.009008789062499998, -0.010107421874999998]
Left move offset (deg): [-0.10420695389851488, 1.7528088258044554, 0.0006526531559405941, -2.166155824566832, 0.00021972656249999998, 0.0015380859374999997, -0.0]
Head move offset (deg): [-0.0, -5.0000976562499995]
Preview move complete. Inspect the robot pose before applying.
