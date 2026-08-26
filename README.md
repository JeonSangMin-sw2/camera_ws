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
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
[Sample 1] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.4 174.9]mm
Auto motion done: J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]
[Sample 2] Captured marker: R_pos=[ 79.6 -35.  164.3]mm, L_pos=[-62.1 -35.7 163.7]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]
[Sample 3] Captured marker: R_pos=[ 89.7 -36.4 182.7]mm, L_pos=[-72.2 -37.1 182. ]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]
[Sample 4] Captured marker: R_pos=[ 89.8 -26.4 186.6]mm, L_pos=[-72.2 -27.2 185.8]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]
[Sample 5] Captured marker: R_pos=[ 84.6 -18.9 179.8]mm, L_pos=[-67.  -19.7 179.2]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]
[Sample 6] Captured marker: R_pos=[ 84.5 -42.  170.7]mm, L_pos=[-67.1 -42.7 170. ]mm
Auto motion done: Restore Baseline Pose
[Sample 7] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 8] Captured marker: R_pos=[ 91.8 -32.5 178. ]mm, L_pos=[-59.8 -29.5 172. ]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 9] Captured marker: R_pos=[ 99.2 -34.5 180.3]mm, L_pos=[-52.8 -27.8 168.8]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 10] Captured marker: R_pos=[ 77.3 -28.8 172.6]mm, L_pos=[-74.2 -33.2 177.3]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 11] Captured marker: R_pos=[ 70.2 -27.1 169.6]mm, L_pos=[-81.5 -35.2 179.6]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 12] Captured marker: R_pos=[ 87.2 -37.8 170.8]mm, L_pos=[-69.6 -38.5 170. ]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 13] Captured marker: R_pos=[ 90.  -44.9 165.9]mm, L_pos=[-72.4 -45.6 165.1]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 14] Captured marker: R_pos=[ 81.9 -23.2 180. ]mm, L_pos=[-64.5 -24.  179.6]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 15] Captured marker: R_pos=[ 79.6 -15.7 184.6]mm, L_pos=[-62.1 -16.5 183.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 16] Captured marker: R_pos=[ 86.1 -35.2 171.4]mm, L_pos=[-65.6 -26.8 179.1]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 17] Captured marker: R_pos=[ 87.8 -39.9 167.5]mm, L_pos=[-64.4 -22.4 183.6]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 18] Captured marker: R_pos=[ 83.2 -26.1 179.9]mm, L_pos=[-68.6 -35.9 170.8]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 19] Captured marker: R_pos=[ 82.  -21.7 184.3]mm, L_pos=[-70.3 -40.6 166.9]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 20] Captured marker: R_pos=[ 66.8 -18.1 177.7]mm, L_pos=[-84.  -44.4 171.2]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 21] Captured marker: R_pos=[ 74.4 -36.6 162.2]mm, L_pos=[-79.8 -26.5 188.7]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[ 97.5 -25.9 189.5]mm, L_pos=[-56.9 -37.3 161.5]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[101.5 -43.6 171.7]mm, L_pos=[-49.3 -18.7 177. ]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 24] Captured marker: R_pos=[ 64.2 -12.1 177.6]mm, L_pos=[-77.7 -20.6 189.5]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
[Sample 25] Captured marker: R_pos=[103.6 -48.6 169.9]mm, L_pos=[-59.3 -42.2 160. ]mm
Auto motion done: Joint 2-4 Decouple (+3.0,-3.0)deg
[Sample 26] Captured marker: R_pos=[ 81.2 -21.5 179.9]mm, L_pos=[-60.7 -10.8 188.8]mm
Auto motion done: Joint 2-4 Decouple (-3.0,+3.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Restore Baseline Pose
[Sample 27] Captured marker: R_pos=[ 84.5 -30.5 175.5]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 28] Captured marker: R_pos=[ 90.2 -31.5 183. ]mm, L_pos=[-72.8 -32.3 182.3]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 29] Captured marker: R_pos=[ 96.2 -32.2 190.4]mm, L_pos=[-78.8 -33.1 189.6]mm
Auto motion done: Elbow Flexion Low (J3 -3deg, J5 +3deg)
[Sample 30] Captured marker: R_pos=[ 76.4 -29.  163.9]mm, L_pos=[-58.8 -29.5 163.1]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[ 85.9 -27.1 191.1]mm, L_pos=[-65.4 -16.5 200.1]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[ 82.8 -17.8 195.8]mm, L_pos=[-60.2   5.1 213.5]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.9]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 92.  -36.1 180.8]mm, L_pos=[-70.  -37.4 184.1]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 93.9 -35.8 179.1]mm, L_pos=[-68.2 -37.7 185.8]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[ 87.5 -36.9 184.7]mm, L_pos=[-74.5 -36.8 180. ]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[ 85.7 -37.2 186.4]mm, L_pos=[-76.4 -36.5 178.4]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[ 89.3 -34.4 182.6]mm, L_pos=[-71.9 -35.  181.8]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[ 89.1 -32.6 182.5]mm, L_pos=[-71.8 -33.3 181.9]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[ 90.2 -38.6 182.7]mm, L_pos=[-72.6 -39.2 182.2]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[ 90.7 -40.4 182.9]mm, L_pos=[-72.9 -40.9 182.1]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[ 89.7 -38.8 182.4]mm, L_pos=[-72.2 -34.8 182.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[ 89.7 -40.7 182.3]mm, L_pos=[-72.1 -32.9 182.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[ 89.7 -34.2 182.9]mm, L_pos=[-72.2 -39.5 181.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[ 89.7 -32.3 183.3]mm, L_pos=[-72.2 -41.4 181.9]mm
Auto motion done: Pos: (-0.030,0.000,0.000)
[Sample 46] Captured marker: R_pos=[ 90.9 -30.  155. ]mm, L_pos=[-73.  -30.3 154.4]mm
Auto motion done: Pos: (0.030,0.000,0.000)
[Sample 47] Captured marker: R_pos=[ 88.6 -43.3 210.6]mm, L_pos=[-71.4 -44.2 209.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 48] Captured marker: R_pos=[ 92.8 -37.  187.8]mm, L_pos=[-69.  -36.8 177.9]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 49] Captured marker: R_pos=[ 94.6 -37.8 193.6]mm, L_pos=[-66.5 -36.5 174.8]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 50] Captured marker: R_pos=[ 86.4 -36.2 178.5]mm, L_pos=[-75.1 -37.6 187. ]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 51] Captured marker: R_pos=[ 83.8 -36.  175.3]mm, L_pos=[-77.  -38.3 192.9]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 52] Captured marker: R_pos=[ 89.9 -33.2 179.7]mm, L_pos=[-72.4 -33.7 179.1]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 53] Captured marker: R_pos=[ 90.1 -31.  177.5]mm, L_pos=[-72.5 -31.4 176.7]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 54] Captured marker: R_pos=[ 89.4 -40.  186.3]mm, L_pos=[-72.  -40.7 185.7]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 55] Captured marker: R_pos=[ 89.1 -42.5 190.9]mm, L_pos=[-71.8 -43.4 190.5]mm
Auto motion done: Head Pan: -3.50deg
[Sample 56] Captured marker: R_pos=[ 75.2 -36.9 187.2]mm, L_pos=[-86.3 -36.6 176.7]mm
Auto motion done: Head Pan: -1.75deg
[Sample 57] Captured marker: R_pos=[ 82.5 -36.7 185. ]mm, L_pos=[-79.2 -36.9 179.4]mm
Auto motion done: Head Pan: +1.75deg
[Sample 58] Captured marker: R_pos=[ 96.8 -36.2 180. ]mm, L_pos=[-65.  -37.4 184.5]mm
Auto motion done: Head Pan: +3.50deg
[Sample 59] Captured marker: R_pos=[103.8 -36.  177.2]mm, L_pos=[-57.7 -37.5 186.5]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 60] Captured marker: R_pos=[ 89.7 -22.4 187.8]mm, L_pos=[-72.1 -23.1 187.3]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 61] Captured marker: R_pos=[ 89.7 -29.4 185.4]mm, L_pos=[-72.1 -30.1 184.8]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 62] Captured marker: R_pos=[ 89.7 -43.4 179.8]mm, L_pos=[-72.2 -44.  179.2]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 63] Captured marker: R_pos=[ 89.6 -50.2 176.6]mm, L_pos=[-72.2 -50.8 176. ]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_122239.npz
Auto motions sequence completed.
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
measurement_noise = sigma_rot=0.2078deg, sigma_pos=0.2466mm
Right arm joint offset (deg): [-0.01136667  1.87488546  0.64463575  2.10169356 -1.38941518  0.01137343
  0.02587709]
Left arm joint offset (deg): [-0.23094996 -1.74069968 -0.6866983   2.05030287  1.36837218 -0.04999903
  0.0999451 ]
Head joint offset (deg): [0.23019875 6.16809948]
mount_to_cam xi: [-0.0016595  -0.00081808  0.00080313  0.00208145 -0.005      -0.00327737]
mount_to_cam_new: [0.04120381437690608, 0.007701966682899155, 0.06270872677764802, -90.14515356248913, -0.06390400066524765, -89.90320560271574]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_122239.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.0114° | Baseline =  +0.1573° | Diff = 0.1687°
   J1: Calc =  +1.8749° | Baseline =  +2.4059° | Diff = 0.5310°
   J2: Calc =  +0.6446° | Baseline =  +0.0007° | Diff = 0.6440°
   J3: Calc =  +2.1017° | Baseline =  +2.2051° | Diff = 0.1034°
   J4: Calc =  -1.3894° | Baseline =  +0.0020° | Diff = 1.3914°
   J5: Calc =  +0.0114° | Baseline =  +0.0090° | Diff = 0.0024°
   J6: Calc =  +0.0259° | Baseline =  +0.0101° | Diff = 0.0158°
 [LEFT ARM]
   J0: Calc =  -0.2309° | Baseline =  +0.1042° | Diff = 0.3352°
   J1: Calc =  -1.7407° | Baseline =  -1.7528° | Diff = 0.0121°
   J2: Calc =  -0.6867° | Baseline =  -0.0007° | Diff = 0.6860°
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
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_122239.json
Arm: both
Right move offset (deg): [0.011366666515193113, -1.874885458144515, -0.6446357502168517, -2.1016935606380613, 1.389415177287269, -0.011373432355733156, -0.025877091139494133]
Left move offset (deg): [0.2309499565884029, 1.7406996829671584, 0.6866983021317855, -2.0503028699203827, -1.368372176838595, 0.04999902989889871, -0.09994509520147238]
Head move offset (deg): [-0.2301987508530976, -6.168099481487844]
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

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260826_122239.json
Arm: both
Right move offset (deg): [0.011366666515193113, -1.874885458144515, -0.6446357502168517, -2.1016935606380613, 1.389415177287269, -0.011373432355733156, -0.025877091139494133]
Left move offset (deg): [0.2309499565884029, 1.7406996829671584, 0.6866983021317855, -2.0503028699203827, -1.368372176838595, 0.04999902989889871, -0.09994509520147238]
Head move offset (deg): [-0.2301987508530976, -6.168099481487844]
Preview move complete. Inspect the robot pose before applying.
