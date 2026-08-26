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
Auto base head pose (deg): [-0.  0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
[Sample 1] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]
[Sample 2] Captured marker: R_pos=[ 79.6 -35.  164.3]mm, L_pos=[-62.1 -35.8 163.6]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]
[Sample 3] Captured marker: R_pos=[ 89.8 -36.5 182.6]mm, L_pos=[-72.2 -37.2 182. ]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]
[Sample 4] Captured marker: R_pos=[ 89.8 -26.4 186.6]mm, L_pos=[-72.2 -27.2 185.8]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]
[Sample 5] Captured marker: R_pos=[ 84.6 -18.9 179.7]mm, L_pos=[-67.  -19.7 179.1]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]
[Sample 6] Captured marker: R_pos=[ 84.5 -42.  170.7]mm, L_pos=[-67.1 -42.7 170. ]mm
Auto motion done: Restore Baseline Pose
[Sample 7] Captured marker: R_pos=[ 84.5 -30.6 175.5]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 8] Captured marker: R_pos=[ 91.8 -32.5 177.9]mm, L_pos=[-59.8 -29.5 171.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 9] Captured marker: R_pos=[ 99.2 -34.5 180.3]mm, L_pos=[-52.8 -27.9 168.8]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 10] Captured marker: R_pos=[ 77.3 -28.8 172.6]mm, L_pos=[-74.2 -33.2 177.3]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 11] Captured marker: R_pos=[ 70.3 -27.2 169.5]mm, L_pos=[-81.5 -35.2 179.5]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 12] Captured marker: R_pos=[ 87.2 -37.8 170.7]mm, L_pos=[-69.6 -38.5 169.9]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 13] Captured marker: R_pos=[ 90.  -44.9 165.9]mm, L_pos=[-72.4 -45.6 165.1]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 14] Captured marker: R_pos=[ 82.  -23.2 180. ]mm, L_pos=[-64.5 -24.  179.5]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 15] Captured marker: R_pos=[ 79.6 -15.7 184.5]mm, L_pos=[-62.1 -16.5 183.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 16] Captured marker: R_pos=[ 86.1 -35.2 171.4]mm, L_pos=[-65.6 -26.8 179.1]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 17] Captured marker: R_pos=[ 87.8 -39.9 167.5]mm, L_pos=[-64.4 -22.4 183.5]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 18] Captured marker: R_pos=[ 83.2 -26.1 179.8]mm, L_pos=[-68.6 -35.9 170.8]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 19] Captured marker: R_pos=[ 82.  -21.7 184.2]mm, L_pos=[-70.3 -40.7 166.9]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 20] Captured marker: R_pos=[ 66.8 -18.1 177.7]mm, L_pos=[-83.9 -44.4 171.1]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 21] Captured marker: R_pos=[ 74.4 -36.6 162.1]mm, L_pos=[-79.8 -26.5 188.7]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[ 97.6 -25.9 189.5]mm, L_pos=[-56.9 -37.3 161.5]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[101.6 -43.7 171.8]mm, L_pos=[-49.3 -18.8 177. ]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 24] Captured marker: R_pos=[ 64.3 -12.2 177.6]mm, L_pos=[-77.7 -20.6 189.4]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
[Sample 25] Captured marker: R_pos=[103.6 -48.7 169.8]mm, L_pos=[-59.3 -42.2 160. ]mm
Auto motion done: Restore Baseline Pose
[Sample 26] Captured marker: R_pos=[ 84.5 -30.6 175.4]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 27] Captured marker: R_pos=[ 90.3 -31.5 182.9]mm, L_pos=[-72.7 -32.3 182.2]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 28] Captured marker: R_pos=[ 96.2 -32.2 190.3]mm, L_pos=[-78.7 -33.1 189.5]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 29] Captured marker: R_pos=[ 85.9 -27.1 191.1]mm, L_pos=[-65.4 -16.5 200.1]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 30] Captured marker: R_pos=[ 82.8 -17.8 195.7]mm, L_pos=[-60.1   5.1 213.4]mm
Auto motion done: Restore Baseline Pose
[Sample 31] Captured marker: R_pos=[ 84.5 -30.6 175.4]mm, L_pos=[-67.  -31.3 174.8]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 32] Captured marker: R_pos=[ 92.  -36.2 180.8]mm, L_pos=[-69.9 -37.4 184. ]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 33] Captured marker: R_pos=[ 93.9 -35.8 179.1]mm, L_pos=[-68.2 -37.7 185.8]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 87.5 -36.9 184.6]mm, L_pos=[-74.5 -36.8 180. ]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 85.7 -37.2 186.4]mm, L_pos=[-76.4 -36.5 178.3]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 36] Captured marker: R_pos=[ 89.4 -34.4 182.6]mm, L_pos=[-71.9 -35.  181.8]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 37] Captured marker: R_pos=[ 89.1 -32.6 182.5]mm, L_pos=[-71.7 -33.3 181.9]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 38] Captured marker: R_pos=[ 90.2 -38.6 182.7]mm, L_pos=[-72.6 -39.2 182.2]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 39] Captured marker: R_pos=[ 90.7 -40.4 182.9]mm, L_pos=[-72.9 -40.9 182.1]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 40] Captured marker: R_pos=[ 89.7 -38.8 182.3]mm, L_pos=[-72.1 -34.8 182.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 41] Captured marker: R_pos=[ 89.7 -40.7 182.3]mm, L_pos=[-72.1 -32.9 182.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 42] Captured marker: R_pos=[ 89.7 -34.2 182.9]mm, L_pos=[-72.2 -39.5 181.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 43] Captured marker: R_pos=[ 89.7 -32.3 183.2]mm, L_pos=[-72.2 -41.4 181.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 44] Captured marker: R_pos=[ 92.8 -37.  187.8]mm, L_pos=[-69.  -36.8 177.9]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 45] Captured marker: R_pos=[ 94.6 -37.8 193.5]mm, L_pos=[-66.5 -36.6 174.7]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 46] Captured marker: R_pos=[ 86.4 -36.2 178.5]mm, L_pos=[-75.1 -37.6 186.9]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 47] Captured marker: R_pos=[ 83.8 -36.  175.3]mm, L_pos=[-76.9 -38.3 192.8]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 48] Captured marker: R_pos=[ 89.9 -33.3 179.7]mm, L_pos=[-72.3 -33.7 179.1]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 49] Captured marker: R_pos=[ 90.  -30.8 177.5]mm, L_pos=[-72.5 -31.2 176.8]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 50] Captured marker: R_pos=[ 89.4 -39.9 186.4]mm, L_pos=[-72.  -40.6 185.8]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 51] Captured marker: R_pos=[ 89.1 -42.5 190.9]mm, L_pos=[-71.8 -43.4 190.5]mm
Auto motion done: Head Pan: -3.50deg
[Sample 52] Captured marker: R_pos=[ 75.2 -36.9 187.2]mm, L_pos=[-86.3 -36.6 176.7]mm
Auto motion done: Head Pan: -1.75deg
[Sample 53] Captured marker: R_pos=[ 82.5 -36.7 184.9]mm, L_pos=[-79.2 -36.9 179.3]mm
Auto motion done: Head Pan: +1.75deg
[Sample 54] Captured marker: R_pos=[ 96.9 -36.3 180. ]mm, L_pos=[-64.9 -37.4 184.4]mm
Auto motion done: Head Pan: +3.50deg
[Sample 55] Captured marker: R_pos=[103.9 -36.  177.2]mm, L_pos=[-57.7 -37.5 186.5]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 56] Captured marker: R_pos=[ 89.8 -22.4 187.9]mm, L_pos=[-72.1 -23.1 187.3]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 57] Captured marker: R_pos=[ 89.7 -29.4 185.4]mm, L_pos=[-72.1 -30.1 184.7]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 58] Captured marker: R_pos=[ 89.7 -43.4 179.7]mm, L_pos=[-72.2 -44.  179.1]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 59] Captured marker: R_pos=[ 89.6 -50.2 176.5]mm, L_pos=[-72.2 -50.8 175.9]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_112701.npz
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
measurement_noise = sigma_rot=0.113deg, sigma_pos=0.3952mm
Right arm joint offset (deg): [-0.97762201  1.19604049  1.48641995  2.15169342 -2.52298123  0.06137255
 -0.        ]
Left arm joint offset (deg): [-1.11656353 -1.20350608 -1.42651657  2.10030271  2.39634459 -0.
  0.14366847]
Head joint offset (deg): [0.23282229 6.71893357]
mount_to_cam xi: [-0.00032068 -0.0005158   0.00037594  0.00212345 -0.00942556 -0.0064307 ]
mount_to_cam_new: [0.038051426461142686, 0.0076656316853924355, 0.06713585869369514, -90.06842449814123, -0.08842956052393554, -89.92046920489366]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_112701.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.9776° | Baseline =  +0.1573° | Diff = 1.1349°
   J1: Calc =  +1.1960° | Baseline =  +2.4059° | Diff = 1.2099°
   J2: Calc =  +1.4864° | Baseline =  +0.0007° | Diff = 1.4858°
   J3: Calc =  +2.1517° | Baseline =  +2.2051° | Diff = 0.0534°
   J4: Calc =  -2.5230° | Baseline =  +0.0020° | Diff = 2.5250°
   J5: Calc =  +0.0614° | Baseline =  +0.0090° | Diff = 0.0524°
   J6: Calc =  -0.0000° | Baseline =  +0.0101° | Diff = 0.0101°
 [LEFT ARM]
   J0: Calc =  -1.1166° | Baseline =  +0.1042° | Diff = 1.2208°
   J1: Calc =  -1.2035° | Baseline =  -1.7528° | Diff = 0.5493°
   J2: Calc =  -1.4265° | Baseline =  -0.0007° | Diff = 1.4259°
   J3: Calc =  +2.1003° | Baseline =  +2.1662° | Diff = 0.0659°
   J4: Calc =  +2.3963° | Baseline =  -0.0002° | Diff = 2.3966°
   J5: Calc =  -0.0000° | Baseline =  -0.0015° | Diff = 0.0015°
   J6: Calc =  +0.1437° | Baseline =  +0.0000° | Diff = 0.1437°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
