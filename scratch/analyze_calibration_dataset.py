import sys
import os
import glob
import json
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer
import rby1_sdk.dynamics as rd

class OfflineRobot:
    def __init__(self, dyn_robot):
        self.dyn_robot = dyn_robot
        self._model = self._create_model_meta()

    def model(self):
        return self._model

    def get_dynamics(self):
        return self.dyn_robot

    def get_state(self):
        class State:
            def __init__(self, dof):
                self.position = np.zeros(dof)
        return State(self.dyn_robot.get_dof())

    def _create_model_meta(self):
        class ModelMeta:
            def __init__(self, dyn_robot):
                self.robot_joint_names = dyn_robot.get_joint_names()
                self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
                self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
                self.head_idx = [self.robot_joint_names.index(f"head_{i}") for i in range(2)] if "head_0" in self.robot_joint_names else []
        return ModelMeta(self.dyn_robot)

def load_offline_robot():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
    if not os.path.exists(urdf_path):
        urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model.urdf" # Fallback
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def load_sweep_data(path):
    angles = []
    poses = []
    cam_pts = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(x.strip()) for x in line.strip().split(",")]
            angles.append(parts[0])
            cam_pts.append(parts[1:4])
            T = np.array(parts[10:26]).reshape(4, 4)
            poses.append(T)
    return np.array(angles), np.array(poses), np.array(cam_pts)

def analyze_sweep_files(directory):
    print("\n" + "="*70)
    print("1. SWEEP MARKER NOISE & 2. CIRCLE INTEGRITY (DISTORTION CHECK)")
    print("="*70)
    
    sweep_files = glob.glob(os.path.join(directory, "sweep_points_*.txt"))
    if not sweep_files:
        print("[WARN] No sweep_points_*.txt files found in directory.")
        return

    for path in sweep_files:
        fname = os.path.basename(path)
        angles, poses, cam_pts = load_sweep_data(path)
        if len(cam_pts) < 10:
            print(f"[{fname}] Not enough data points ({len(cam_pts)}).")
            continue
            
        # Point-to-point distance noise
        dists = np.linalg.norm(np.diff(cam_pts, axis=0), axis=1)
        noise_std = np.std(dists)
        
        # Circle Fit
        c_fit, R_fit, r_fit, rmse_fit, _, _, _ = BaseCalibrator.fit_circle_3d(cam_pts, robust=True)
        
        print(f"--- File: {fname} ---")
        print(f"  Total Points  : {len(cam_pts)}")
        print(f"  Marker Noise (Point-to-Point Dist StdDev): {noise_std:.4f} mm")
        print(f"  Circle Radius : {r_fit:.2f} mm")
        print(f"  Circle RMSE   : {rmse_fit:.4f} mm")
        if rmse_fit < 2.0:
            print("  >> Result: Circle maintained well (Low Spatial Distortion).")
        else:
            print("  >> WARNING: High RMSE! Circle shape is distorted or noisy. Check Camera Intrinsics.")

def analyze_joint_convergence(directory):
    print("\n" + "="*70)
    print("3. JOINT OFFSET 3, 5, 6 CONVERGENCE & OSCILLATION ANALYSIS")
    print("="*70)
    
    log_files = glob.glob(os.path.join(directory, "joint_calib_debug_*.txt"))
    # Also check flat directory in case they are extracted without subfolders
    if not log_files:
        log_files = glob.glob(os.path.join(directory, "*", "joint_calib_debug_*.txt"))
        
    if not log_files:
        print("[WARN] No joint_calib_debug_*.txt logs found.")
        return
        
    for path in log_files:
        fname = os.path.basename(path)
        print(f"\n--- Log: {fname} ---")
        
        iterations = []
        with open(path, "r") as f:
            for line in f:
                if "Iter" in line and "delta =" in line:
                    try:
                        # [DEBUG] Iter X: delta = Y
                        parts = line.split("delta =")
                        delta = float(parts[1].strip().split()[0])
                        iterations.append(delta)
                    except:
                        pass
        
        if not iterations:
            print("  No iteration delta data found.")
            continue
            
        print(f"  Deltas per Iteration: {iterations}")
        
        # Check oscillation
        signs = np.sign(iterations)
        sign_flips = np.sum(signs[:-1] != signs[1:])
        
        if sign_flips > len(iterations) // 2 and len(iterations) > 3:
            print(f"  >> WARNING: Oscillating convergence detected ({sign_flips} sign flips).")
        elif len(iterations) > 0 and abs(iterations[-1]) > 0.5:
            print(f"  >> WARNING: Did not fully converge (Final Delta: {iterations[-1]} > 0.5)")
        else:
            print("  >> Result: Converged smoothly.")

def analyze_step2_validity(directory):
    print("\n" + "="*70)
    print("4. STEP 2 (QP CALIBRATION) VALIDITY CHECK")
    print("="*70)
    
    npz_files = glob.glob(os.path.join(directory, "dataset_*.npz"))
    json_files = glob.glob(os.path.join(directory, "result_*.json"))
    
    if not npz_files:
        print("[WARN] No dataset_*.npz files found.")
    else:
        latest_npz = max(npz_files, key=os.path.getctime)
        print(f"Found Dataset: {os.path.basename(latest_npz)}")
        try:
            q_arm, q_head, T_meas = load_npz_dataset(latest_npz)
            print(f"  Contains {len(T_meas)} samples.")
        except Exception as e:
            print(f"  [ERROR] Failed to load NPZ: {e}")

    if not json_files:
        print("[WARN] No result_*.json files found.")
    else:
        latest_json = max(json_files, key=os.path.getctime)
        print(f"Found JSON Result: {os.path.basename(latest_json)}")
        try:
            with open(latest_json, "r") as f:
                data = json.load(f)
            
            print("  --- Final Offsets (Degrees) ---")
            for key, val in data.get("joint_offsets_deg", {}).items():
                print(f"    {key}: {val}")
                
            rms_pos = data.get("rms_error_pos_mm", "N/A")
            rms_rot = data.get("rms_error_rot_deg", "N/A")
            print(f"  --- Errors ---")
            print(f"    RMS Position Error: {rms_pos} mm")
            print(f"    RMS Rotation Error: {rms_rot} deg")
            
            if isinstance(rms_pos, (int, float)) and rms_pos > 15.0:
                print("  >> WARNING: Position error is > 15mm. Calibration might be invalid.")
            else:
                print("  >> Result: Errors are within expected range.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to load JSON: {e}")

def analyze_camera_distortion():
    print("\n" + "="*70)
    print("5. CAMERA INTRINSICS DISTORTION CHECK")
    print("="*70)
    
    intrinsics_path = "/home/rainbow/camera_ws/config/camera_intrinsics.yaml"
    if os.path.exists(intrinsics_path):
        try:
            with open(intrinsics_path, "r") as f:
                cfg = yaml.safe_load(f)
            rms = cfg.get("rms_error", -1)
            print(f"Camera Intrinsics RMS Error: {rms:.4f} px")
            if rms > 0.35:
                print(">> WARNING: High intrinsic calibration RMS error. This will distort circle shapes in 3D.")
            else:
                print(">> Result: Intrinsic calibration looks solid.")
        except Exception as e:
            print(f"[ERROR] Could not read intrinsics: {e}")
    else:
        print("[WARN] camera_intrinsics.yaml not found.")

def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "/home/rainbow/camera_ws/result"
        print(f"[INFO] No directory specified. Using default: {directory}")
        
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        sys.exit(1)
        
    print(f"Starting Diagnostic Analysis for Directory: {directory}")
    
    analyze_sweep_files(directory)
    analyze_joint_convergence(directory)
    analyze_step2_validity(directory)
    analyze_camera_distortion()
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
