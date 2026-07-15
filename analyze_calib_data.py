import os
import sys
import numpy as np
import glob
import re

def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    basis1 = Vt[0, :]
    basis2 = Vt[1, :]
    
    # Check ellipse ratio (S[0] / S[1]) to see if it's distorted
    ellipse_ratio = S[0] / S[1] if S[1] > 1e-6 else 1.0
    
    points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A_mat = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    c, resid, rank, s = np.linalg.lstsq(A_mat, b, rcond=None)
    xc, yc = c[0]/2, c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    # Calculate noise (RMSE)
    pred_x = xc + r * np.cos(np.arctan2(y - yc, x - xc))
    pred_y = yc + r * np.sin(np.arctan2(y - yc, x - xc))
    rmse = np.sqrt(np.mean((x - pred_x)**2 + (y - pred_y)**2))
    max_err = np.max(np.sqrt((x - pred_x)**2 + (y - pred_y)**2))
    
    return r, rmse, max_err, ellipse_ratio

def analyze_sweep_files(result_dir):
    print("="*60)
    print(" 1. SWEEP POINTS ANALYSIS (Noise, Shape, Distortion)")
    print("="*60)
    
    # Check both result_txt and result/result_txt depending on folder structure
    txt_dir = os.path.join(result_dir, "result_txt")
    if not os.path.exists(txt_dir):
        txt_dir = os.path.join(result_dir, "result", "result_txt")
        
    sweep_files = glob.glob(os.path.join(txt_dir, "sweep_points_*.txt"))
    if not sweep_files:
        print(f"No sweep_points_*.txt files found in {txt_dir}.")
        return
        
    for fpath in sorted(sweep_files):
        fname = os.path.basename(fpath)
        data = []
        with open(fpath, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split(',')
                if len(parts) > 3:
                    try:
                        pt = [float(p.split(':')[1]) if ':' in p else float(p) for p in parts[1:4]]
                        data.append(pt)
                    except:
                        pass
        if len(data) < 10:
            print(f"[{fname}] Not enough points.")
            continue
            
        data = np.array(data)
        r, rmse, max_err, ellipse_ratio = fit_circle_3d(data)
        
        status = "OK"
        if rmse > 2.0 or max_err > 5.0:
            status = "NOISY (High variance)"
        if ellipse_ratio > 1.05 or ellipse_ratio < 0.95:
            status = "DISTORTED (Camera distortion or non-circular path)"
            
        print(f"[{fname}] Status: {status}")
        print(f"  - Points: {len(data)}")
        print(f"  - Radius: {r:.2f} mm")
        print(f"  - RMSE Noise: {rmse:.3f} mm")
        print(f"  - Max Deviation: {max_err:.3f} mm")
        print(f"  - Ellipse Ratio (Major/Minor): {ellipse_ratio:.3f} (Ideal: 1.0)")
        print()

def analyze_convergence(result_dir):
    print("="*60)
    print(" 2. JOINT OFFSET CONVERGENCE ANALYSIS (Divergence/Vibration)")
    print("="*60)
    
    txt_dir = os.path.join(result_dir, "result_txt")
    if not os.path.exists(txt_dir):
        txt_dir = os.path.join(result_dir, "result", "result_txt")
        
    debug_files = glob.glob(os.path.join(txt_dir, "joint_calib_debug_*.txt"))
    if not debug_files:
        print(f"No joint_calib_debug_*.txt files found in {txt_dir}.")
        return
        
    for fpath in sorted(debug_files):
        fname = os.path.basename(fpath)
        print(f"--- {fname} ---")
        
        iterations = []
        current_staged = None
        current_correction = None
        
        with open(fpath, "r") as f:
            for line in f:
                m = re.search(r"Sweeping physically with staged offset ([-0-9.]+)°", line)
                if m:
                    current_staged = float(m.group(1))
                
                m2 = re.search(r"Calculated Offset Correction\s*:\s*([-0-9.]+)\s*deg", line)
                if m2:
                    current_correction = float(m2.group(1))
                    if current_staged is not None:
                        iterations.append((current_staged, current_correction))
                        current_staged = None
        
        if not iterations:
            print("  No iteration data found.")
            print()
            continue
            
        for i, (staged, corr) in enumerate(iterations):
            print(f"  Iter {i+1}: Staged = {staged:7.4f}°, Correction = {corr:7.4f}°")
            
        if len(iterations) > 2:
            last_corr = iterations[-1][1]
            prev_corr = iterations[-2][1]
            if abs(last_corr) > 3.0 and abs(last_corr) > abs(iterations[0][1]):
                print("  => [WARNING] Divergence detected! The correction magnitude is growing or very large.")
            elif last_corr * prev_corr < 0 and abs(last_corr) > 1.0:
                print("  => [WARNING] Vibration/Oscillation detected! The correction sign is flipping wildly.")
            else:
                print("  => [OK] Convergence looks stable.")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_calib_data.py <path_to_result_folder>")
        sys.exit(1)
        
    result_dir = sys.argv[1]
    if not os.path.isdir(result_dir):
        print(f"Error: {result_dir} is not a valid directory.")
        sys.exit(1)
        
    analyze_sweep_files(result_dir)
    analyze_convergence(result_dir)
    
if __name__ == "__main__":
    main()
