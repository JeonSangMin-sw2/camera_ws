import os
import glob
import json
import numpy as np

res_dir = "/home/rainbow/camera_ws/result/result_real/result_step2"
files = glob.glob(os.path.join(res_dir, "result_*.json"))
files.sort()  # Sort chronologically

print(f"Analyzing {len(files)} result files...")
print(f"{'Time':<20} | {'R_J3':<7} | {'R_J5':<7} | {'R_J6':<7} | {'Head_P':<7} | {'Head_T':<7} | {'Cam_Tx':<7} | {'Cam_Ty':<7} | {'Cam_Tz':<7}")
print("-" * 100)

for fpath in files:
    fname = os.path.basename(fpath)
    time_str = fname.replace("result_", "").replace(".json", "")
    with open(fpath, "r") as f:
        data = json.load(f)
    
    # Extract right arm J3, J5, J6 (indices 3, 5, 6)
    r_joint = data.get("right_arm_joint_offset_deg")
    if r_joint is None:
        r_joint = data.get("joint_offset_deg")[:7] if "joint_offset_deg" in data else [0]*7
    
    rj3 = r_joint[3] if len(r_joint) > 3 else 0.0
    rj5 = r_joint[5] if len(r_joint) > 5 else 0.0
    rj6 = r_joint[6] if len(r_joint) > 6 else 0.0
    
    # Head pan, tilt
    head = data.get("head_joint_offset_deg", [0, 0])
    hp = head[0] if head else 0.0
    ht = head[1] if head else 0.0
    
    # Cam Translation (last 3 of xi_cam)
    xi = data.get("xi_cam", [0]*6)
    ctx = xi[3] * 1000.0 if len(xi) > 3 else 0.0
    cty = xi[4] * 1000.0 if len(xi) > 4 else 0.0
    ctz = xi[5] * 1000.0 if len(xi) > 5 else 0.0
    
    print(f"{time_str:<20} | {rj3:7.3f} | {rj5:7.3f} | {rj6:7.3f} | {hp:7.3f} | {ht:7.3f} | {ctx:7.2f} | {cty:7.2f} | {ctz:7.2f}")

