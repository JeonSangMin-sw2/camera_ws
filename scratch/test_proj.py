import numpy as np

def angle(v1, v2, n):
    return np.degrees(np.arctan2(np.dot(np.cross(v1, v2), n), np.dot(v1, v2)))

J4_true = np.array([0, 1, 0])
J3 = np.array([0, 0, 1])

for t in [45, 60, 75, 80]:
    tilt = np.radians(t)
    a_cand_cam = np.array([0, np.cos(tilt), np.sin(tilt)])
    
    delta = np.radians(1)
    J5 = np.array([np.sin(delta), 0, np.cos(delta)])
    
    n_A_proj = J3 - np.dot(J3, a_cand_cam) * a_cand_cam
    n_B_proj = J5 - np.dot(J5, a_cand_cam) * a_cand_cam
    
    a = angle(n_A_proj, n_B_proj, a_cand_cam)
    print(f"Tilt: {t}, Proj Angle: {a:.3f}")
