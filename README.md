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
Auto base head pose (deg): [0. 0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (+0.0deg) + Head Tilt (-5.0deg)
[Sample 1] Captured marker: R_pos=[ 89.8 -16.2 189.8]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-5.0deg)
[Sample 2] Captured marker: R_pos=[ 82.4 -17.4 176.2]mm, L_pos=[-64.8 -18.3 175.6]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-10.0deg)
[Sample 3] Captured marker: R_pos=[ 82.5   2.2 181.9]mm, L_pos=[-64.8   1.3 181.3]mm
Auto motion done: J0 (-10.0deg) + Head Tilt (-10.0deg)
[Sample 4] Captured marker: R_pos=[ 75.7 -20.  162.7]mm, L_pos=[-58.2 -20.7 162.1]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+0.0deg)
[Sample 5] Captured marker: R_pos=[ 97.9   5.7 208.7]mm, L_pos=[-80.1   4.8 208.3]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+5.0deg)
[Sample 6] Captured marker: R_pos=[ 97.8 -16.3 203.4]mm, L_pos=[-80.2 -17.1 202.9]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+5.0deg)
[Sample 7] Captured marker: R_pos=[106.4   5.5 222.2]mm, L_pos=[-88.6   4.6 221.8]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+10.0deg)
[Sample 8] Captured marker: R_pos=[106.3 -17.6 216.7]mm, L_pos=[-88.6 -18.4 216.2]mm
Auto motion done: Restore Baseline Pose
[Sample 9] Captured marker: R_pos=[ 89.8 -16.2 189.8]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 10] Captured marker: R_pos=[ 97.2 -17.9 192.4]mm, L_pos=[-65.  -15.4 186.5]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 11] Captured marker: R_pos=[104.6 -19.8 194.6]mm, L_pos=[-57.9 -13.9 183.3]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 12] Captured marker: R_pos=[ 82.6 -14.6 187.1]mm, L_pos=[-79.4 -18.8 191.7]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 13] Captured marker: R_pos=[ 75.4 -13.1 184. ]mm, L_pos=[-86.8 -20.7 194. ]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 14] Captured marker: R_pos=[ 92.4 -23.7 185.5]mm, L_pos=[-74.8 -24.5 184.9]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 15] Captured marker: R_pos=[ 95.2 -31.1 181. ]mm, L_pos=[-77.4 -31.8 180.4]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 16] Captured marker: R_pos=[ 87.4  -8.6 194.1]mm, L_pos=[-69.8  -9.4 193.6]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 17] Captured marker: R_pos=[ 85.   -0.8 198.1]mm, L_pos=[-67.5  -1.7 197.6]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 18] Captured marker: R_pos=[ 91.3 -21.  186. ]mm, L_pos=[-70.9 -12.2 193.3]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 19] Captured marker: R_pos=[ 92.9 -26.  182.4]mm, L_pos=[-69.8  -7.6 197.6]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 20] Captured marker: R_pos=[ 88.6 -11.5 193.9]mm, L_pos=[-73.7 -21.8 185.5]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 21] Captured marker: R_pos=[ 87.5  -6.8 198.2]mm, L_pos=[-75.3 -26.8 181.9]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[ 72.2  -3.6 191.7]mm, L_pos=[-89.  -30.3 186. ]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[ 79.4 -23.  177.1]mm, L_pos=[-85.3 -11.4 202.6]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 24] Captured marker: R_pos=[103.2 -10.7 203.3]mm, L_pos=[-61.9 -23.8 176.7]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 25] Captured marker: R_pos=[106.8 -29.5 186.5]mm, L_pos=[-54.5  -4.3 191.1]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 26] Captured marker: R_pos=[ 69.6   2.3 191.4]mm, L_pos=[-83.2  -5.5 203.1]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
[Sample 27] Captured marker: R_pos=[108.8 -34.6 184.9]mm, L_pos=[-64.2 -28.7 175.4]mm
Auto motion done: Restore Baseline Pose
[Sample 28] Captured marker: R_pos=[ 89.8 -16.2 189.9]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 29] Captured marker: R_pos=[ 95.7 -16.7 197.4]mm, L_pos=[-78.1 -17.5 196.7]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 30] Captured marker: R_pos=[101.9 -17.  204.6]mm, L_pos=[-84.2 -17.9 203.9]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[ 91.5 -11.8 205.1]mm, L_pos=[-71.1  -0.8 213.7]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[ 88.5  -2.3 209.4]mm, L_pos=[-66.2  21.6 226. ]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[ 89.8 -16.2 189.8]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 91.8 -36.2 181. ]mm, L_pos=[-70.3 -37.3 183.8]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 94.  -35.7 179.1]mm, L_pos=[-68.1 -37.7 185.9]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[ 87.6 -36.9 184.6]mm, L_pos=[-74.5 -36.8 180. ]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[ 85.8 -37.2 186.3]mm, L_pos=[-76.4 -36.5 178.3]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[ 89.4 -34.4 182.5]mm, L_pos=[-71.9 -35.  181.8]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[ 89.2 -32.6 182.5]mm, L_pos=[-71.7 -33.3 181.8]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[ 90.3 -38.6 182.7]mm, L_pos=[-72.6 -39.2 182.1]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[ 90.8 -40.4 182.9]mm, L_pos=[-72.9 -40.9 182.1]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[ 89.8 -38.8 182.4]mm, L_pos=[-72.2 -34.8 182.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[ 89.8 -40.7 182.3]mm, L_pos=[-72.1 -32.9 182.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[ 89.7 -34.2 182.8]mm, L_pos=[-72.2 -39.5 181.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[ 89.8 -32.3 183.2]mm, L_pos=[-72.2 -41.4 181.8]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 46] Captured marker: R_pos=[ 92.8 -37.1 187.7]mm, L_pos=[-69.  -36.8 177.9]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 47] Captured marker: R_pos=[ 94.7 -37.7 193.6]mm, L_pos=[-66.5 -36.6 174.6]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 48] Captured marker: R_pos=[ 86.4 -36.1 178.5]mm, L_pos=[-75.2 -37.6 186.9]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 49] Captured marker: R_pos=[ 83.8 -36.  175.2]mm, L_pos=[-76.9 -38.3 192.7]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 50] Captured marker: R_pos=[ 90.  -33.3 179.7]mm, L_pos=[-72.4 -33.7 178.9]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 51] Captured marker: R_pos=[ 90.1 -30.8 177.5]mm, L_pos=[-72.5 -31.3 176.8]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 52] Captured marker: R_pos=[ 89.5 -39.9 186.5]mm, L_pos=[-72.1 -40.7 185.6]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 53] Captured marker: R_pos=[ 89.2 -42.5 190.8]mm, L_pos=[-71.8 -43.4 190.3]mm
Auto motion done: Head Pan: -3.50deg
[Sample 54] Captured marker: R_pos=[ 75.2 -36.9 187.1]mm, L_pos=[-86.3 -36.7 176.6]mm
Auto motion done: Head Pan: -1.75deg
[Sample 55] Captured marker: R_pos=[ 82.6 -36.7 185. ]mm, L_pos=[-79.3 -36.9 179.4]mm
Auto motion done: Head Pan: +1.75deg
[Sample 56] Captured marker: R_pos=[ 96.9 -36.3 180. ]mm, L_pos=[-65.  -37.4 184.4]mm
Auto motion done: Head Pan: +3.50deg
[Sample 57] Captured marker: R_pos=[103.9 -36.  177.2]mm, L_pos=[-57.8 -37.6 186.5]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 58] Captured marker: R_pos=[ 89.8 -22.4 187.9]mm, L_pos=[-72.2 -23.1 187.2]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 59] Captured marker: R_pos=[ 89.7 -29.4 185.3]mm, L_pos=[-72.2 -30.1 184.7]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 60] Captured marker: R_pos=[ 89.7 -43.4 179.7]mm, L_pos=[-72.2 -44.  179. ]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 61] Captured marker: R_pos=[ 89.7 -50.2 176.5]mm, L_pos=[-72.2 -50.8 175.8]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_092834.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0539, -0.0485, 89.0, 0.22, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0541, -0.0482, 89.58, -0.06, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2853877559237685, 'joint5': 0.0, 'joint6': 0.08086111523160816}, 'left': {'joint3': -2.136694958302961, 'joint5': 0.0, 'joint6': 0.0}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.1036deg, sigma_pos=0.4978mm
Right arm joint offset (deg): [-0.8225711   1.75477265  1.17305864  2.23538775 -1.94168599 -0.05000001
 -0.03086112]
Left arm joint offset (deg): [-0.8646172  -1.13244632 -1.21790559  2.08669495  2.21200192 -0.05000001
 -0.05      ]
Head joint offset (deg): [0.02089725 6.40636719]
mount_to_cam xi: [-4.71755557e-04 -4.68132593e-04 -1.00777577e-06  4.28885232e-04
 -7.99239558e-03 -5.02758115e-03]
mount_to_cam_new: [0.03945174627399822, 0.00936091867855066, 0.06569877490806059, -90.07708110416445, -0.11002799586549993, -89.92317788145466]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_092834.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.8226° | Baseline =  +0.1573° | Diff = 0.9799°
   J1: Calc =  +1.7548° | Baseline =  +2.4059° | Diff = 0.6511°
   J2: Calc =  +1.1731° | Baseline =  +0.0007° | Diff = 1.1724°
   J3: Calc =  +2.2354° | Baseline =  +2.2051° | Diff = 0.0303°
   J4: Calc =  -1.9417° | Baseline =  +0.0020° | Diff = 1.9437°
   J5: Calc =  -0.0500° | Baseline =  +0.0090° | Diff = 0.0590°
   J6: Calc =  -0.0309° | Baseline =  +0.0101° | Diff = 0.0410°
 [LEFT ARM]
   J0: Calc =  -0.8646° | Baseline =  +0.1042° | Diff = 0.9688°
   J1: Calc =  -1.1324° | Baseline =  -1.7528° | Diff = 0.6204°
   J2: Calc =  -1.2179° | Baseline =  -0.0007° | Diff = 1.2173°
   J3: Calc =  +2.0867° | Baseline =  +2.1662° | Diff = 0.0795°
   J4: Calc =  +2.2120° | Baseline =  -0.0002° | Diff = 2.2122°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  -0.0500° | Baseline =  +0.0000° | Diff = 0.0500°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
