import os
import sys
import numpy as np

# Add workspace path
sys.path.append('/home/rainbow/camera_ws')

from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator

txt_dir = '/home/rainbow/camera_ws/result_1_3/result_txt'

def parse_sweep_file(filepath):
    if not os.path.exists(filepath):
        return None
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
            poses.append(T_cam2marker)
    return poses

def fit_circle_3d(pts):
    c0 = np.mean(pts, axis=0)
    pts_c = pts - c0
    u, s, vt = np.linalg.svd(pts_c)
    normal = vt[2]
    u_vec = vt[0]
    v_vec = vt[1]
    u_coords = pts_c @ u_vec
    v_coords = pts_c @ v_vec
    A = np.column_stack([u_coords, v_coords, np.ones_like(u_coords)])
    b = u_coords**2 + v_coords**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc = sol[0] / 2.0
    vc = sol[1] / 2.0
    r = np.sqrt(sol[2] + uc**2 + vc**2)
    center_3d = c0 + uc * u_vec + vc * v_vec
    dist = np.sqrt((u_coords - uc)**2 + (v_coords - vc)**2)
    rmse = np.sqrt(np.mean((dist - r)**2))
    pts_2d = np.column_stack([u_coords, v_coords])
    return center_3d, normal, r, rmse, pts_2d, uc, vc

def run_pipeline_test():
    print("=================================================================")
    print("   RUNNING V1.3 UNIFIED 3-AXIS SIMULTANEOUS CALIBRATION TEST     ")
    print("=================================================================")
    
    import rby1_sdk
    robot = rby1_sdk.create_robot("127.0.0.1", "m")
    mc = MarkerCalibrator(None, robot)
    mc.robot_version = "1.3"
    
    for arm_side in ["left", "right"]:
        p4 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_4.txt'))
        p5 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_5.txt'))
        p6 = parse_sweep_file(os.path.join(txt_dir, f'sweep_points_{arm_side}_marker_axis_6.txt'))
        
        c4, n4, r4, rmse4, pts4, uc4, vc4 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p4]))
        c5, n5, r5, rmse5, pts5, uc5, vc5 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p5]))
        c6, n6, r6, rmse6, pts6, uc6, vc6 = fit_circle_3d(np.array([T[:3, 3]*1000 for T in p6]))
        
        m4 = {'captured_poses': p4, 'radius': r4, 'rmse': rmse4, 'pts_2d': pts4, 'uc_opt': uc4, 'vc_opt': vc4}
        m5 = {'captured_poses': p5, 'radius': r5, 'rmse': rmse5, 'pts_2d': pts5, 'uc_opt': uc5, 'vc_opt': vc5}
        m6 = {'captured_poses': p6, 'radius': r6, 'rmse': rmse6, 'pts_2d': pts6, 'uc_opt': uc6, 'vc_opt': vc6}
        
        unified_res = mc.compute_unified_bracket_calibration_v1_3(
            m5, m6, arm_side, marker_data_4=m4, calib_roll_deg=0.0, calib_pitch_deg=0.0
        )
        
        print(f"\n[{arm_side.upper()} ARM CALIBRATION RESULT]")
        print(f"  * Recommended Joint 5 (Pitch) Offset : {unified_res['d5_opt_deg']:+.4f}°")
        print(f"  * Recommended Joint 6 (Roll)  Offset : {unified_res['d6_opt_deg']:+.4f}°")
        print(f"  * Bracket Position (X, Y, Z)         : [{unified_res['x_e']:.2f}, {unified_res['y_e']:.2f}, {unified_res['z_e']:.2f}] mm")
        print(f"  * Bracket Orientation (R, P, Y)      : [{unified_res['roll_e']:.2f}, {unified_res['pitch_e']:.2f}, {unified_res['yaw_e']:.2f}]°")
        print(f"  * Orthogonality J4-J5 (Yaw-Pitch)    : {unified_res['ang_45']:.3f}° (Dev: {abs(unified_res['ang_45']-90.0):.3f}°)")
        print(f"  * Orthogonality J5-J6 (Pitch-Roll)   : {unified_res['ang_56']:.3f}° (Dev: {abs(unified_res['ang_56']-90.0):.3f}°)")
        print(f"  * Orthogonality J4-J6 (Yaw-Roll)     : {unified_res['ang_46']:.3f}° (Dev: {abs(unified_res['ang_46']-90.0):.3f}°)")
        print(f"  * Max Radius Residual Error          : {unified_res['max_radius_err']:.3f} mm")
        print(f"  * Converged Status                   : {unified_res['converged']}")
        
        # Test Plot Generation
        plot_path = f"/home/rainbow/camera_ws/scratch/test_plot_{arm_side}.png"
        saved = mc.generate_marker_plot(m5, m6, m4, unified_res, arm_side, is_v13=True, save_path=plot_path)
        print(f"  * Plot Saved Successfully            : {saved} -> {plot_path}")
        
        assert unified_res['converged'] == True
        assert 'd5_opt_deg' in unified_res
        assert 'd6_opt_deg' in unified_res
        assert saved == True
        
    print("\n=================================================================")
    print("   ALL TESTS PASSED SUCCESSFULLY!                                ")
    print("=================================================================")

if __name__ == '__main__':
    run_pipeline_test()
