pi05_open_door_merge_right_arm 라고 학습했었습니다.
TrainConfig(
        name="pi05_open_door_merge_right_arm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=40,
        ),
        data=LeRobotRBY1OneArmDataConfig(
            repo_id="sanfmin/opendoor_data_merge",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        batch_size=32,
        num_train_steps=50_000,
        log_interval=1000,
        save_interval=10_000,
        fsdp_devices=1,
        freeze_filter=nnx.Nothing,
        seed=1000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
    ),

혹시 얘기주신게 이거 맞을까요?




아래와 같이 실행했어요.

uv run src/openpi/serving/zmq_policy_server.py  \
  --config-name  pi05_open_door_merge_right_arm \
  --checkpoint-dir ~/NAS/openpi/pi05_open_door_merge_right_arm/opendoor_without_mobile/49999  \
  --host 0.0.0.0 --port 5555


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
[WARN] Initial connection attempt failed. Waiting 1.0s before retrying...
[ERROR] Connection failure: Failed to connect robot at 192.168.30.1:50051
[WARNING] Model mismatch! UI selected model: a, but actual robot model is: M. Auto-reconnecting with actual model...
[INFO] Power is already ON.
[INFO] Servos are ON.
[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Auto-updating UI model selection to match robot model: 'M'
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.0686°, J5=0.0000°, J6=0.3089°] L[J3=-1.9436°, J5=0.0000°, J6=-0.0782°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Right arm marker not visible at Init Pose. Showing teaching dialog...
[INFO] Preserved user-taught ready pose for right arm (marker).
[INFO] Re-verifying marker visibility at the new posture...
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [-0.021 -0.022]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 1] Captured marker: R_pos=[ 49.3 -22.3 201. ]mm, L_pos=[-84.2 -14.5 195.5]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 2] Captured marker: R_pos=[ 45.  -21.1 186.4]mm, L_pos=[-73.5 -13.  185.6]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 3] Captured marker: R_pos=[ 51.8 -23.  208.5]mm, L_pos=[-89.7 -15.1 200.7]mm
Auto motion done: Joint 0 Offset: 5.0deg
[Sample 4] Captured marker: R_pos=[ 54.5 -24.1 216.1]mm, L_pos=[-95.3 -15.9 205.9]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[ 50.5 -24.6 206.9]mm, L_pos=[-85.8 -12.8 190.4]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[ 51.5 -26.8 212.8]mm, L_pos=[-87.3 -11.1 185.3]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 48.1 -20.  195.2]mm, L_pos=[-82.4 -16.1 200.6]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 46.6 -17.7 189.3]mm, L_pos=[-80.2 -17.8 205.6]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[ 56.6 -24.6 200. ]mm, L_pos=[-89.2 -15.4 192. ]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[ 64.3 -26.8 199.4]mm, L_pos=[-94.5 -16.5 188.8]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 42.7 -19.8 202.5]mm, L_pos=[-79.5 -13.5 199.2]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 36.4 -17.2 204.4]mm, L_pos=[-75.2 -12.7 203.1]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 51.8 -27.2 196.5]mm, L_pos=[-82.7  -9.6 200.1]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[ 54.2 -32.1 192.4]mm, L_pos=[-81.6  -4.5 204.9]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 47.2 -17.4 205.6]mm, L_pos=[-85.8 -19.5 191.2]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 45.3 -12.4 210.5]mm, L_pos=[-87.6 -24.4 187.1]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 42.2  -7.9 198.4]mm, L_pos=[-83.4 -27.9 196.7]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 51.8 -27.7 181. ]mm, L_pos=[-78.   -8.  215.3]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[ 47.5 -17.  222.5]mm, L_pos=[-91.2 -21.1 177.3]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[ 56.  -36.6 203.7]mm, L_pos=[-84.3  -1.2 194.2]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 33.8 -11.5 192.2]mm, L_pos=[-71.6 -17.2 213.8]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Restore Baseline Pose
[Sample 22] Captured marker: R_pos=[ 49.4 -22.2 201. ]mm, L_pos=[-84.1 -14.4 195.6]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 23] Captured marker: R_pos=[ 54.5 -23.  206.7]mm, L_pos=[-87.6 -15.3 202.9]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 24] Captured marker: R_pos=[ 57.2 -22.9 205.1]mm, L_pos=[-85.5 -15.5 205.2]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 25] Captured marker: R_pos=[ 49.2 -23.1 210.5]mm, L_pos=[-91.8 -15.  198.5]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 46.7 -23.3 212.5]mm, L_pos=[-94.  -14.8 196.5]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 27] Captured marker: R_pos=[ 51.4 -20.9 208.4]mm, L_pos=[-89.  -12.9 200.5]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 28] Captured marker: R_pos=[ 51.1 -18.5 208.1]mm, L_pos=[-88.3 -10.6 200.2]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 29] Captured marker: R_pos=[ 52.4 -25.4 208.7]mm, L_pos=[-90.5 -17.4 200.9]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 30] Captured marker: R_pos=[ 52.9 -27.6 208.9]mm, L_pos=[-91.4 -19.6 201.1]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 31] Captured marker: R_pos=[ 51.8 -25.5 208.6]mm, L_pos=[-89.6 -12.8 200.6]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 32] Captured marker: R_pos=[ 51.8 -27.8 208.7]mm, L_pos=[-89.6 -10.5 200.7]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 33] Captured marker: R_pos=[ 51.8 -20.7 208.6]mm, L_pos=[-89.7 -17.5 200.8]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 34] Captured marker: R_pos=[ 51.8 -18.3 208.7]mm, L_pos=[-89.7 -19.8 201.1]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 35] Captured marker: R_pos=[ 55.1 -23.  211.1]mm, L_pos=[-85.4 -15.1 194.7]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 36] Captured marker: R_pos=[ 58.1 -23.  214.4]mm, L_pos=[-81.  -14.9 189.7]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 37] Captured marker: R_pos=[ 48.1 -23.2 207. ]mm, L_pos=[-93.9 -15.3 207.3]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 38] Captured marker: R_pos=[ 44.2 -23.3 206.4]mm, L_pos=[-97.8 -15.7 214.7]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 39] Captured marker: R_pos=[ 52.4 -20.3 205.1]mm, L_pos=[-90.2 -11.9 197.4]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 40] Captured marker: R_pos=[ 52.8 -17.8 202.3]mm, L_pos=[-90.8  -8.7 195. ]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 41] Captured marker: R_pos=[ 51.2 -25.9 212.8]mm, L_pos=[-88.8 -18.5 204.7]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 42] Captured marker: R_pos=[ 50.5 -28.7 217.6]mm, L_pos=[-88.  -21.8 209.3]mm
Auto motion done: Head Pan: -3.50deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Head Pan: -1.75deg
[Sample 43] Captured marker: R_pos=[ 44.  -23.  209.2]mm, L_pos=[-97.2 -15.1 196.9]mm
Auto motion done: Head Pan: +1.75deg
[Sample 44] Captured marker: R_pos=[ 59.5 -23.2 207.7]mm, L_pos=[-82.1 -15.2 204.2]mm
Auto motion done: Head Pan: +3.50deg
[Sample 45] Captured marker: R_pos=[ 67.1 -23.3 206.6]mm, L_pos=[-74.4 -15.3 207.7]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 46] Captured marker: R_pos=[ 52.1  -7.5 213.2]mm, L_pos=[-89.4   0.  204.8]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 47] Captured marker: R_pos=[ 51.9 -15.2 211. ]mm, L_pos=[-89.6  -7.5 202.9]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 48] Captured marker: R_pos=[ 51.6 -30.7 205.7]mm, L_pos=[-89.7 -22.6 198.3]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 49] Captured marker: R_pos=[ 51.5 -38.4 202.6]mm, L_pos=[-89.8 -30.  195.6]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260812_190908.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.0, -0.0544, -0.049, 89.6, -0.36, 180.0]
[INFO] Using calibrated marker bracket values for left: [0.0, 0.0547, -0.0489, 88.78, -0.11, 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.068647947024635, 'joint5': 0.0, 'joint6': 0.3089493273485196}, 'left': {'joint3': -1.9436406734698237, 'joint5': 0.0, 'joint6': -0.07817763836420505}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.2487deg, sigma_pos=0.1339mm
Right arm joint offset (deg): [ 0.04604022  3.6749295   0.86854863  2.01864796 -2.24647931 -0.0499994
 -0.35894769]
Left arm joint offset (deg): [ 0.21131437 -4.0303271  -0.27839172  1.89364089  0.55125055 -0.04999897
  0.02817722]
Head joint offset (deg): [ -0.80228713 -89.95437384]
mount_to_cam xi: [-1.57667647e+00  1.02047606e-03  2.25838493e-02  2.80821692e-03
  7.08623043e-02  8.79886279e-02]
mount_to_cam_new: [0.06008424540736694, 0.030780932418608305, -0.04409601071372197, 179.65379897669425, 0.7833128599797315, -90.86266929788461]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260812_190908.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +0.0460° | Baseline =  +0.0644° | Diff = 0.0184°
   J1: Calc =  +3.6749° | Baseline =  +4.4074° | Diff = 0.7324°
   J2: Calc =  +0.8685° | Baseline =  +0.0002° | Diff = 0.8683°
   J3: Calc =  +2.0186° | Baseline =  +1.5588° | Diff = 0.4599°
   J4: Calc =  -2.2465° | Baseline =  -0.0053° | Diff = 2.2412°
   J5: Calc =  -0.0500° | Baseline =  +0.0806° | Diff = 0.1306°
   J6: Calc =  -0.3589° | Baseline =  +0.0013° | Diff = 0.3603°
 [LEFT ARM]
   J0: Calc =  +0.2113° | Baseline =  +0.2308° | Diff = 0.0195°
   J1: Calc =  -4.0303° | Baseline =  -4.1041° | Diff = 0.0738°
   J2: Calc =  -0.2784° | Baseline =  +0.0020° | Diff = 0.2803°
   J3: Calc =  +1.8936° | Baseline =  +1.8988° | Diff = 0.0051°
   J4: Calc =  +0.5513° | Baseline =  +0.0308° | Diff = 0.5205°
   J5: Calc =  -0.0500° | Baseline =  +0.0393° | Diff = 0.0893°
   J6: Calc =  +0.0282° | Baseline =  -0.0062° | Diff = 0.0343°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...
