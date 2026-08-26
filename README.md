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
[INFO] Step 2 Mode changed to: 'live'
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
[INFO] Step 2 Mode changed to: 'npz'
[Step2] Calculate requested.

[Step2] Calculate requested (Active Mode: 'npz').
[Step2] Loading NPZ dataset from: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_132834.npz
[Step2] Loaded 63 samples from NPZ dataset.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0538, -0.0487, 89.41, 0.13, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0543, -0.0482, 89.16, -0.06, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.1516934165067894, 'joint5': -0.06137254693253481, 'joint6': 0.0}, 'left': {'joint3': -2.1003027088121917, 'joint5': 0.0, 'joint6': -0.1436684682678275}}

[INFO] === SEQUENTIAL 3-STAGE JOINT-CAMERA CALIBRATION WORKFLOW ===
[STAGE 1/3] Right Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 2/3] Left Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 3/3] Dual-Arm Unified Fine Integration (Warm Start, max_iter=50, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.2058deg, sigma_pos=0.2437mm
Right arm joint offset (deg): [-0.00870621  1.86499316  0.65365147  2.10169357 -1.39959091  0.0113735
  0.02674028]
Left arm joint offset (deg): [-0.21560149 -1.74180624 -0.68443258  2.05030289  1.37392954 -0.04999894
  0.09861403]
Head joint offset (deg): [0.22613396 6.17635681]
mount_to_cam xi: [-0.00172881 -0.00079386  0.00073179  0.00205441 -0.005      -0.003356  ]
mount_to_cam_new: [0.04112529862057629, 0.007729126505678826, 0.06270900854670894, -90.14912417465418, -0.06799272145338513, -89.90458799900259]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_133541.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.0087° | Baseline =  +0.1573° | Diff = 0.1660°
   J1: Calc =  +1.8650° | Baseline =  +2.4059° | Diff = 0.5409°
   J2: Calc =  +0.6537° | Baseline =  +0.0007° | Diff = 0.6530°
   J3: Calc =  +2.1017° | Baseline =  +2.2051° | Diff = 0.1034°
   J4: Calc =  -1.3996° | Baseline =  +0.0020° | Diff = 1.4016°
   J5: Calc =  +0.0114° | Baseline =  +0.0090° | Diff = 0.0024°
   J6: Calc =  +0.0267° | Baseline =  +0.0101° | Diff = 0.0166°
 [LEFT ARM]
   J0: Calc =  -0.2156° | Baseline =  +0.1042° | Diff = 0.3198°
   J1: Calc =  -1.7418° | Baseline =  -1.7528° | Diff = 0.0110°
   J2: Calc =  -0.6844° | Baseline =  -0.0007° | Diff = 0.6838°
   J3: Calc =  +2.0503° | Baseline =  +2.1662° | Diff = 0.1159°
   J4: Calc =  +1.3739° | Baseline =  -0.0002° | Diff = 1.3741°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  +0.0986° | Baseline =  +0.0000° | Diff = 0.0986°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.




Preview moves use the same convention as Apply Home Offset:
the robot moves to zero pose first, then to -joint_offset.

Optimized result: /home/nvidia/camera_ws/result/result_step2/result_20260826_133541.json
Baseline reset: /home/nvidia/camera_ws/config/home_reset_baseline.json

--- RIGHT ARM (deg) ---
  Baseline  : [0.1573, 2.4059, 0.0007, 2.2051, 0.002 , 0.009 , 0.0101]
  Optimized : [-0.0087,  1.865 ,  0.6537,  2.1017, -1.3996,  0.0114,  0.0267]
  Diff (B-O): [ 0.166 ,  0.5409, -0.653 ,  0.1034,  1.4016, -0.0024, -0.0166]

--- LEFT ARM (deg) ---
  Baseline  : [ 0.1042, -1.7528, -0.0007,  2.1662, -0.0002, -0.0015,  0.    ]
  Optimized : [-0.2156, -1.7418, -0.6844,  2.0503,  1.3739, -0.05  ,  0.0986]
  Diff (B-O): [ 0.3198, -0.011 ,  0.6838,  0.1159, -1.3741,  0.0485, -0.0986]

--- HEAD (deg) ---
  Baseline  : [0.    , 5.0001]
  Optimized : [0.2261, 6.1764]
  Diff (B-O): [-0.2261, -1.1763]




(.venv) nvidia@tegra-ubuntu:~/camera_ws$ python3 main_ui.py 
[INFO] Initializing Camera Marker Transform System...
[0] RealSense D405 (Serial: 260322277376)
Resetting Realsense device...
Using camera is :  RealSense D405
depth scale :  9.999999747378752e-05
- Loaded Setting Config from setting.yaml
  * head_base_to_cam: [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
  * mount_to_cam: [0.04447, 0.00978, 0.0577, -90.05, -0.11, -89.95]
[0.0, 0.0543, -0.0482, 89.16, -0.06, 0.0]
Initializing Camera...
Successfully initialized: 1280x720 @ 30fps
Focal Length: fx=654.0186767578125, fy=653.168212890625
Principal Point: 642.6535034179688, 347.9039001464844
Baseline:  0.065
2026-08-26 13:35:23,819 [INFO] Loaded config from setting.yaml successfully.
2026-08-26 13:35:23,838 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
2026-08-26 13:35:23,847 [INFO] Loaded config from setting.yaml successfully.
2026-08-26 13:35:23,861 [INFO] Loaded ready poses from /home/nvidia/camera_ws/config/ready_poses.yaml
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.1517°, J5=-0.0614°, J6=0.0000°] L[J3=-2.1003°, J5=0.0000°, J6=-0.1437°]
[INFO] Connected robot model version string: 'v1.2'
[00] |dx|=7.283e-02, |err|=4.636e+00
[01] |dx|=4.253e-02, |err|=7.533e-01
[02] |dx|=1.078e-01, |err|=5.263e-01
[03] |dx|=3.345e-02, |err|=1.705e-01
[04] |dx|=4.958e-03, |err|=8.968e-02
[05] |dx|=1.476e-03, |err|=8.366e-02
[06] |dx|=2.053e-03, |err|=8.273e-02
[07] |dx|=2.332e-03, |err|=8.170e-02
[08] |dx|=1.886e-03, |err|=8.065e-02
[09] |dx|=1.264e-03, |err|=7.993e-02
[10] |dx|=7.834e-04, |err|=7.955e-02
[11] |dx|=4.720e-04, |err|=7.938e-02
[12] |dx|=2.809e-04, |err|=7.931e-02
[13] |dx|=1.663e-04, |err|=7.929e-02
[14] |dx|=9.834e-05, |err|=7.928e-02
[15] |dx|=5.819e-05, |err|=7.929e-02
[16] |dx|=3.446e-05, |err|=7.929e-02
[17] |dx|=2.043e-05, |err|=7.929e-02
[18] |dx|=1.212e-05, |err|=7.929e-02
[19] |dx|=7.201e-06, |err|=7.929e-02
[20] |dx|=4.280e-06, |err|=7.930e-02
[21] |dx|=2.546e-06, |err|=7.930e-02
[22] |dx|=1.515e-06, |err|=7.930e-02
[23] |dx|=9.019e-07, |err|=7.930e-02
[24] |dx|=5.371e-07, |err|=7.930e-02
[25] |dx|=3.200e-07, |err|=7.930e-02
[26] |dx|=1.907e-07, |err|=7.930e-02
[27] |dx|=1.136e-07, |err|=7.930e-02
[28] |dx|=6.778e-08, |err|=7.930e-02
Converged.
[00] |dx|=7.900e-02, |err|=4.895e+00
[01] |dx|=1.489e-02, |err|=7.271e-01
[02] |dx|=1.024e-01, |err|=5.902e-01
[03] |dx|=4.033e-02, |err|=2.300e-01
[04] |dx|=5.909e-03, |err|=1.377e-01
[05] |dx|=3.914e-03, |err|=1.270e-01
[06] |dx|=3.582e-03, |err|=1.225e-01
[07] |dx|=2.142e-03, |err|=1.193e-01
[08] |dx|=1.278e-03, |err|=1.173e-01
[09] |dx|=7.096e-04, |err|=1.161e-01
[10] |dx|=3.978e-04, |err|=1.154e-01
[11] |dx|=2.187e-04, |err|=1.151e-01
[12] |dx|=9.914e-05, |err|=1.149e-01
[13] |dx|=6.866e-05, |err|=1.148e-01
[14] |dx|=3.037e-05, |err|=1.147e-01
[15] |dx|=2.300e-05, |err|=1.147e-01
[16] |dx|=1.590e-05, |err|=1.147e-01
[17] |dx|=1.066e-05, |err|=1.146e-01
[18] |dx|=7.175e-06, |err|=1.146e-01
[19] |dx|=4.949e-06, |err|=1.146e-01
[20] |dx|=3.545e-06, |err|=1.146e-01
[21] |dx|=2.646e-06, |err|=1.146e-01
[22] |dx|=2.047e-06, |err|=1.146e-01
[23] |dx|=1.626e-06, |err|=1.146e-01
[24] |dx|=1.315e-06, |err|=1.146e-01
[25] |dx|=1.075e-06, |err|=1.146e-01
[26] |dx|=8.846e-07, |err|=1.146e-01
[27] |dx|=7.307e-07, |err|=1.146e-01
[28] |dx|=6.048e-07, |err|=1.146e-01
[29] |dx|=5.011e-07, |err|=1.146e-01
[30] |dx|=4.154e-07, |err|=1.146e-01
[31] |dx|=3.444e-07, |err|=1.146e-01
[32] |dx|=2.855e-07, |err|=1.146e-01
[33] |dx|=2.366e-07, |err|=1.146e-01
[34] |dx|=1.960e-07, |err|=1.146e-01
[35] |dx|=1.623e-07, |err|=1.146e-01
[36] |dx|=1.344e-07, |err|=1.146e-01
[37] |dx|=1.113e-07, |err|=1.146e-01
[38] |dx|=9.206e-08, |err|=1.146e-01
Converged.
[00] |dx|=3.640e-02, |err|=1.317e+00
[01] |dx|=1.588e-02, |err|=1.018e+00
[02] |dx|=5.462e-03, |err|=8.540e-01
[03] |dx|=3.947e-03, |err|=7.185e-01
[04] |dx|=1.018e-03, |err|=7.228e-01
[05] |dx|=7.097e-04, |err|=7.398e-01
[06] |dx|=9.999e-04, |err|=7.384e-01
[07] |dx|=5.119e-04, |err|=7.395e-01
[08] |dx|=2.025e-04, |err|=7.430e-01
[09] |dx|=1.839e-04, |err|=7.461e-01
[10] |dx|=1.615e-04, |err|=7.483e-01
[11] |dx|=1.313e-04, |err|=7.501e-01
[12] |dx|=1.028e-04, |err|=7.516e-01
[13] |dx|=8.041e-05, |err|=7.528e-01
[14] |dx|=6.280e-05, |err|=7.537e-01
[15] |dx|=4.876e-05, |err|=7.545e-01
[16] |dx|=3.772e-05, |err|=7.550e-01
[17] |dx|=2.916e-05, |err|=7.555e-01
[18] |dx|=2.254e-05, |err|=7.558e-01
[19] |dx|=1.741e-05, |err|=7.561e-01
[20] |dx|=1.345e-05, |err|=7.563e-01
[21] |dx|=1.039e-05, |err|=7.565e-01
[22] |dx|=8.032e-06, |err|=7.566e-01
[23] |dx|=6.209e-06, |err|=7.567e-01
[24] |dx|=4.801e-06, |err|=7.568e-01
[25] |dx|=3.713e-06, |err|=7.568e-01
[26] |dx|=2.872e-06, |err|=7.569e-01
[27] |dx|=2.222e-06, |err|=7.569e-01
[28] |dx|=1.720e-06, |err|=7.570e-01
[29] |dx|=1.331e-06, |err|=7.570e-01
[30] |dx|=1.030e-06, |err|=7.570e-01
[31] |dx|=7.977e-07, |err|=7.570e-01
[32] |dx|=6.176e-07, |err|=7.570e-01
[33] |dx|=4.782e-07, |err|=7.570e-01
[34] |dx|=3.703e-07, |err|=7.570e-01
[35] |dx|=2.868e-07, |err|=7.570e-01
[36] |dx|=2.221e-07, |err|=7.570e-01
[37] |dx|=1.720e-07, |err|=7.570e-01
[38] |dx|=1.333e-07, |err|=7.570e-01
[39] |dx|=1.032e-07, |err|=7.570e-01
[40] |dx|=7.995e-08, |err|=7.570e-01
Converged.



