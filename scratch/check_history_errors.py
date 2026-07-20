import os
import json
import glob
import numpy as np

MOCK_GT_OFFSETS = {
    "right": [0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3],
    "left": [-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5],
    "head": [0.8, -1.5]
}

result_dir = "/home/rainbow/camera_ws/result/result_step2"
json_files = sorted(glob.glob(os.path.join(result_dir, "result_*.json")))

print(f"{'Filename':<30} | {'R_max_err':<9} | {'L_max_err':<9} | {'H_max_err':<9} | Optimize Head?")
print("-" * 80)

for fpath in json_files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            continue
            
    r_est = data.get("right_arm_joint_offset_deg")
    l_est = data.get("left_arm_joint_offset_deg")
    h_est = data.get("head_joint_offset_deg")
    
    r_err = np.max(np.abs(np.array(r_est) - np.array(MOCK_GT_OFFSETS["right"]))) if r_est else None
    l_err = np.max(np.abs(np.array(l_est) - np.array(MOCK_GT_OFFSETS["left"]))) if l_est else None
    h_err = np.max(np.abs(np.array(h_est) - np.array(MOCK_GT_OFFSETS["head"]))) if h_est else None
    
    r_err_str = f"{r_err:.4f}" if r_err is not None else "N/A"
    l_err_str = f"{l_err:.4f}" if l_err is not None else "N/A"
    h_err_str = f"{h_err:.4f}" if h_err is not None else "N/A"
    opt_head = "Yes" if h_est is not None else "No"
    
    print(f"{fname:<30} | {r_err_str:<9} | {l_err_str:<9} | {h_err_str:<9} | {opt_head}")
