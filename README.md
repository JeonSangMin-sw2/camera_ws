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
[Step2] Calculate requested.
[ERROR] Robot is not connected!
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
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_132219.json
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
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[WARNING] Auto-centering head encountered a minor issue: list indices must be integers or slices, not tuple
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [0.001 0.   ]
[Step2] Auto Motion requested.
[Step2] Auto motion is only available in live or sim mode.
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
[Sample 1] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.9]mm
Auto motion done: J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]
[Sample 2] Captured marker: R_pos=[ 79.6 -35.  164.4]mm, L_pos=[-62.1 -35.8 163.7]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]
[Sample 3] Captured marker: R_pos=[ 89.8 -36.5 182.8]mm, L_pos=[-72.3 -37.2 182.2]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]
[Sample 4] Captured marker: R_pos=[ 89.8 -26.5 186.7]mm, L_pos=[-72.2 -27.2 186. ]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]
[Sample 5] Captured marker: R_pos=[ 84.6 -18.9 179.8]mm, L_pos=[-67.  -19.7 179.3]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]
[Sample 6] Captured marker: R_pos=[ 84.5 -42.  170.8]mm, L_pos=[-67.1 -42.7 170.1]mm
Auto motion done: Restore Baseline Pose
[Sample 7] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.4 174.9]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 8] Captured marker: R_pos=[ 91.8 -32.5 178.1]mm, L_pos=[-59.9 -29.5 172.1]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 9] Captured marker: R_pos=[ 99.2 -34.5 180.4]mm, L_pos=[-52.8 -27.9 168.9]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 10] Captured marker: R_pos=[ 77.4 -28.8 172.7]mm, L_pos=[-74.2 -33.3 177.4]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 11] Captured marker: R_pos=[ 70.3 -27.2 169.6]mm, L_pos=[-81.5 -35.3 179.7]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 12] Captured marker: R_pos=[ 87.2 -37.9 170.9]mm, L_pos=[-69.7 -38.5 170.1]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 13] Captured marker: R_pos=[ 90. -45. 166.]mm, L_pos=[-72.4 -45.6 165.2]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 14] Captured marker: R_pos=[ 82.  -23.2 180.1]mm, L_pos=[-64.5 -24.  179.6]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 15] Captured marker: R_pos=[ 79.6 -15.7 184.7]mm, L_pos=[-62.2 -16.6 184. ]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 16] Captured marker: R_pos=[ 86.1 -35.2 171.5]mm, L_pos=[-65.7 -26.8 179.2]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 17] Captured marker: R_pos=[ 87.8 -39.9 167.6]mm, L_pos=[-64.4 -22.4 183.6]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 18] Captured marker: R_pos=[ 83.2 -26.1 179.9]mm, L_pos=[-68.6 -36.  170.9]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 19] Captured marker: R_pos=[ 82.  -21.7 184.3]mm, L_pos=[-70.3 -40.7 167. ]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 20] Captured marker: R_pos=[ 66.8 -18.2 177.8]mm, L_pos=[-84.  -44.4 171.2]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 21] Captured marker: R_pos=[ 74.4 -36.6 162.2]mm, L_pos=[-79.8 -26.6 188.8]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[ 97.6 -25.9 189.6]mm, L_pos=[-56.9 -37.4 161.6]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[101.6 -43.7 171.9]mm, L_pos=[-49.3 -18.8 177.1]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 24] Captured marker: R_pos=[ 64.3 -12.2 177.7]mm, L_pos=[-77.7 -20.7 189.6]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
[Sample 25] Captured marker: R_pos=[103.6 -48.7 170. ]mm, L_pos=[-59.3 -42.3 160.1]mm
Auto motion done: Joint 2-4 Decouple (+3.0,-3.0)deg
[Sample 26] Captured marker: R_pos=[ 81.3 -21.5 180. ]mm, L_pos=[-60.7 -10.9 189. ]mm
Auto motion done: Joint 2-4 Decouple (-3.0,+3.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Restore Baseline Pose
[Sample 27] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.9]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 28] Captured marker: R_pos=[ 90.3 -31.5 183.1]mm, L_pos=[-72.8 -32.3 182.4]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 29] Captured marker: R_pos=[ 96.3 -32.3 190.5]mm, L_pos=[-78.8 -33.2 189.7]mm
Auto motion done: Elbow Flexion Low (J3 -3deg, J5 +3deg)
[Sample 30] Captured marker: R_pos=[ 76.4 -29.  164. ]mm, L_pos=[-58.8 -29.5 163.2]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[ 85.9 -27.2 191.2]mm, L_pos=[-65.4 -16.5 200.2]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[ 82.8 -17.9 195.8]mm, L_pos=[-60.2   5.1 213.6]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.9]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 92.  -36.1 180.9]mm, L_pos=[-69.9 -37.4 184.1]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 93.9 -35.8 179.2]mm, L_pos=[-68.2 -37.7 186. ]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[ 87.5 -36.9 184.8]mm, L_pos=[-74.6 -36.8 180.1]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[ 85.8 -37.2 186.6]mm, L_pos=[-76.4 -36.5 178.5]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[ 89.4 -34.4 182.7]mm, L_pos=[-71.9 -35.  182. ]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[ 89.2 -32.6 182.7]mm, L_pos=[-71.8 -33.3 182. ]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[ 90.2 -38.6 182.8]mm, L_pos=[-72.6 -39.3 182.3]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[ 90.7 -40.4 183. ]mm, L_pos=[-72.9 -40.9 182.2]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[ 89.7 -38.8 182.5]mm, L_pos=[-72.2 -34.8 182.3]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[ 89.7 -40.7 182.5]mm, L_pos=[-72.1 -32.9 182.5]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[ 89.8 -34.2 183.1]mm, L_pos=[-72.2 -39.5 182. ]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[ 89.7 -32.3 183.4]mm, L_pos=[-72.3 -41.4 182. ]mm
Auto motion done: Pos: (-0.030,0.000,0.000)
[Sample 46] Captured marker: R_pos=[ 90.9 -30.  155.1]mm, L_pos=[-73.  -30.4 154.5]mm
Auto motion done: Pos: (0.030,0.000,0.000)
[Sample 47] Captured marker: R_pos=[ 88.6 -43.3 210.7]mm, L_pos=[-71.4 -44.2 210. ]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 48] Captured marker: R_pos=[ 92.9 -37.  187.9]mm, L_pos=[-69.  -36.8 178. ]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 49] Captured marker: R_pos=[ 94.7 -37.8 193.7]mm, L_pos=[-66.5 -36.6 174.9]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 50] Captured marker: R_pos=[ 86.5 -36.2 178.6]mm, L_pos=[-75.1 -37.7 187.1]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 51] Captured marker: R_pos=[ 83.8 -36.1 175.4]mm, L_pos=[-77.  -38.3 192.9]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 52] Captured marker: R_pos=[ 89.9 -33.3 179.8]mm, L_pos=[-72.4 -33.7 179.3]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 53] Captured marker: R_pos=[ 90.2 -31.  177.6]mm, L_pos=[-72.5 -31.5 176.8]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 54] Captured marker: R_pos=[ 89.4 -40.  186.5]mm, L_pos=[-72.  -40.7 185.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 55] Captured marker: R_pos=[ 89.1 -42.5 191. ]mm, L_pos=[-71.8 -43.4 190.6]mm
Auto motion done: Head Pan: -3.50deg
[Sample 56] Captured marker: R_pos=[ 75.2 -36.9 187.2]mm, L_pos=[-86.3 -36.6 176.8]mm
Auto motion done: Head Pan: -1.75deg
[Sample 57] Captured marker: R_pos=[ 82.5 -36.7 185.1]mm, L_pos=[-79.3 -36.9 179.5]mm
Auto motion done: Head Pan: +1.75deg
[Sample 58] Captured marker: R_pos=[ 96.9 -36.2 180.1]mm, L_pos=[-64.9 -37.4 184.5]mm
Auto motion done: Head Pan: +3.50deg
[Sample 59] Captured marker: R_pos=[103.9 -36.  177.3]mm, L_pos=[-57.8 -37.5 186.7]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 60] Captured marker: R_pos=[ 89.8 -22.4 188. ]mm, L_pos=[-72.1 -23.1 187.4]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 61] Captured marker: R_pos=[ 89.7 -29.4 185.5]mm, L_pos=[-72.1 -30.1 184.9]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 62] Captured marker: R_pos=[ 89.7 -43.4 179.8]mm, L_pos=[-72.2 -44.  179.2]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 63] Captured marker: R_pos=[ 89.7 -50.2 176.6]mm, L_pos=[-72.2 -50.8 176. ]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_132834.npz
Auto motions sequence completed.
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
measurement_noise = sigma_rot=0.2058deg, sigma_pos=0.2438mm
Right arm joint offset (deg): [-0.00865425  1.86496955  0.65377419  2.10169357 -1.39968392  0.01137349
  0.02675811]
Left arm joint offset (deg): [-0.2155195  -1.7418399  -0.68451305  2.05030289  1.37397741 -0.04999894
  0.09859926]
Head joint offset (deg): [0.22614069 6.17644998]
mount_to_cam xi: [-0.00173028 -0.00079371  0.00073158  0.00205446 -0.005      -0.00335567]
mount_to_cam_new: [0.04112562348613598, 0.007729078386004935, 0.06270901073206937, -90.14920812391085, -0.06800455382682823, -89.9045964246225]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_132834.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.0087° | Baseline =  +0.1573° | Diff = 0.1659°
   J1: Calc =  +1.8650° | Baseline =  +2.4059° | Diff = 0.5409°
   J2: Calc =  +0.6538° | Baseline =  +0.0007° | Diff = 0.6531°
   J3: Calc =  +2.1017° | Baseline =  +2.2051° | Diff = 0.1034°
   J4: Calc =  -1.3997° | Baseline =  +0.0020° | Diff = 1.4017°
   J5: Calc =  +0.0114° | Baseline =  +0.0090° | Diff = 0.0024°
   J6: Calc =  +0.0268° | Baseline =  +0.0101° | Diff = 0.0167°
 [LEFT ARM]
   J0: Calc =  -0.2155° | Baseline =  +0.1042° | Diff = 0.3197°
   J1: Calc =  -1.7418° | Baseline =  -1.7528° | Diff = 0.0110°
   J2: Calc =  -0.6845° | Baseline =  -0.0007° | Diff = 0.6839°
   J3: Calc =  +2.0503° | Baseline =  +2.1662° | Diff = 0.1159°
   J4: Calc =  +1.3740° | Baseline =  -0.0002° | Diff = 1.3742°
   J5: Calc =  -0.0500° | Baseline =  -0.0015° | Diff = 0.0485°
   J6: Calc =  +0.0986° | Baseline =  +0.0000° | Diff = 0.0986°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
