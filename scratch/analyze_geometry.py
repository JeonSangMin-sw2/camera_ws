"""
새로운 실측 데이터를 기반으로 물리적 기하 분석 수행
- 반지름 관계식으로부터 x_e, y_e, z_e, L_5_ee 추정
- 현재 알고리즘의 경계값 문제 진단
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

sys.path.append("/home/rainbow/camera_ws/core/calibration")
from MarkerCalibrator import MarkerCalibrator

def load_sweep_data(filepath):
    angles = []
    poses = []
    ee_pts = []  # EE_X, EE_Y, EE_Z
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(p.strip()) for p in line.strip().split(",")]
            angles.append(parts[0])
            # cols 6,7,8 = EE_X, EE_Y, EE_Z
            ee_pts.append(parts[6:9])
            # T_cam2marker_flat (cols 10-25)
            T_flat = parts[10:26]
            T = np.array(T_flat).reshape(4, 4)
            poses.append(T)
    return np.array(angles), np.array(poses), np.array(ee_pts)

def main():
    base_dir = "/home/rainbow/camera_ws/core/calibration"
    print("="*60)
    print("실측 데이터 물리 기하 분석")
    print("="*60)

    angles_4, poses_4, ee_4 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_4 (1).txt"))
    angles_5, poses_5, ee_5 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_5 (1).txt"))
    angles_6, poses_6, ee_6 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_6 (1).txt"))

    print(f"\n데이터 포인트: Axis4={len(angles_4)}, Axis5={len(angles_5)}, Axis6={len(angles_6)}")

    # ─── 1) EE frame에서 본 마커 위치 (T_ee2marker 변환) ───────────────────
    # 파일 컬럼 구조: ...EE_X, EE_Y, EE_Z (idx6,7,8)...
    # 이것이 마커의 EE frame 좌표 (mm)
    print("\n[EE frame 마커 좌표 통계]")
    for name, ee_arr in [("Axis4", ee_4), ("Axis5", ee_5), ("Axis6", ee_6)]:
        mean = np.mean(ee_arr, axis=0)
        std  = np.std(ee_arr, axis=0)
        print(f"  {name}: mean=[{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}] mm")
        print(f"          std =[{std[0]:.2f},  {std[1]:.2f},  {std[2]:.2f}] mm")

    # ─── 2) 카메라-마커 변환의 translation 궤적으로 원 피팅 ────────────────
    calibrator = MarkerCalibrator(marker_st=None, robot=None)

    pts_4 = np.array([T[:3, 3] * 1000.0 for T in poses_4])
    pts_5 = np.array([T[:3, 3] * 1000.0 for T in poses_5])
    pts_6 = np.array([T[:3, 3] * 1000.0 for T in poses_6])

    res_4 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_4, angles_4, axis_prior=[0.0, 0.0, 1.0])
    res_5 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_5, angles_5, axis_prior=[0.0, 1.0, 0.0])
    res_6 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_6, angles_6, axis_prior=[1.0, 0.0, 0.0])

    r4 = res_4['radius']
    r5 = res_5['radius']
    r6 = res_6['radius']
    print(f"\n[원 피팅 반지름] R4={r4:.4f} mm, R5={r5:.4f} mm, R6={r6:.4f} mm")

    # ─── 3) 반지름 관계식으로 실제 파라미터 추정 ───────────────────────────
    # v1.3 키네마틱:
    #   J6 (X축 회전, d6): r6 = sqrt(y_e^2 + z_e^2)
    #   J5 (Y축 회전, d5): Z_p = y_e*sin(d6) + z_e*cos(d6) + L
    #                      r5 = sqrt(x_e^2 + Z_p^2)
    #   J4 (Z축 회전):     Y' = y_e*cos(d6) - z_e*sin(d6)
    #                      X_p = x_e*cos(d5) + Z_p*sin(d5)
    #                      r4 = sqrt(X_p^2 + Y'^2)

    # d6, d5 추정 (rotation axis 방법)
    nominal_rpy = [90.0, 0.0, -90.0]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

    def extract_axis(poses, ideal):
        mid = len(poses)//2
        R_ref = poses[mid][:3,:3]
        axes = []
        for i,T in enumerate(poses):
            if i == mid: continue
            R_rel = R_ref.T @ T[:3,:3]
            rv = R_scipy.from_matrix(R_rel).as_rotvec()
            ang = np.linalg.norm(rv)
            if ang > np.radians(1.0):
                ax = rv/ang
                if np.dot(ax, ideal) < 0: ax = -ax
                axes.append(ax)
        if axes:
            avg = np.mean(axes, axis=0)
            return avg / np.linalg.norm(avg)
        return ideal

    n6 = extract_axis(poses_6, x_ee_m_ideal)
    n5 = extract_axis(poses_5, y_ee_m_ideal)
    n4 = extract_axis(poses_4, z_ee_m_ideal)

    # Joint offset 추정
    u5 = R_ee_m_ideal @ n5
    d6_rad = -np.arctan2(u5[2], u5[1])
    u4 = R_ee_m_ideal @ n4
    w4 = R_scipy.from_euler('X', d6_rad).as_matrix() @ u4
    d5_rad = np.arctan2(-w4[0], w4[2])
    print(f"\n[추정된 조인트 오프셋] d5={np.degrees(d5_rad):.4f} deg, d6={np.degrees(d6_rad):.4f} deg")

    # 반지름 방정식 비선형 최솟값 추정 (L_5_ee도 변수로!)
    print("\n[L_5_ee를 변수로 포함한 비선형 최적화]")
    def residuals(params):
        xe, ye, ze, L = params
        r6_p = np.sqrt(ye**2 + ze**2)
        Z_p  = ye * np.sin(d6_rad) + ze * np.cos(d6_rad) + L
        r5_p = np.sqrt(xe**2 + Z_p**2)
        Y_pr = ye * np.cos(d6_rad) - ze * np.sin(d6_rad)
        X_pr = xe * np.cos(d5_rad) + Z_p * np.sin(d5_rad)
        r4_p = np.sqrt(X_pr**2 + Y_pr**2)
        reg = 1e-3
        return [r6_p - r6, r5_p - r5, r4_p - r4,
                reg*(xe - 96.0), reg*(ye - 0.0)]

    # 다양한 초기값으로 시도
    best_cost = np.inf
    best_sol = None
    for xe0 in [80, 90, 96, 100]:
        for ze0 in [-200, -150, -130, -100, -70, -50, -30]:
            for L0 in [230, 260, 280, 300]:
                try:
                    sol = least_squares(residuals, [xe0, 0.0, ze0, L0], loss='huber',
                                        bounds=([50, -10, -300, 150], [150, 10, 50, 400]))
                    if sol.cost < best_cost:
                        best_cost = sol.cost
                        best_sol = sol.x
                except:
                    pass

    xe_s, ye_s, ze_s, L_s = best_sol
    print(f"  xe={xe_s:.4f} mm, ye={ye_s:.4f} mm, ze={ze_s:.4f} mm, L_5_ee={L_s:.4f} mm")
    print(f"  cost={best_cost:.8f}")
    print(f"  → x_e 오차(기준 96mm): {abs(xe_s - 96.0):.4f} mm  (5mm 이내? {abs(xe_s-96.0) < 5.0})")

    # 현재 알고리즘의 경계값 확인
    x_nom, y_nom, z_nom = 96.0, 0.0, -5.0
    print(f"\n[현재 알고리즘의 z_e 경계]")
    print(f"  z_nom={z_nom} mm")
    print(f"  z_min = z_nom - 60 = {z_nom - 60:.1f} mm")
    print(f"  z_max = z_nom + 60 = {z_nom + 60:.1f} mm")
    print(f"  실제 z_e = {ze_s:.1f} mm  → 경계 내? {(z_nom-60) <= ze_s <= (z_nom+60)}")
    print(f"\n  ⚠️  실제 z_e({ze_s:.1f} mm)가 z_min({z_nom-60:.1f} mm) 경계 밖")
    print(f"  ⚠️  최적화기가 z_min에 막혀 x_e가 왜곡됨")
    print(f"\n[필요한 수정]")
    print(f"  z_bounds: z_nom ± 60mm  →  실물 범위 기반 절대값 경계로 변경")
    print(f"  또는 L_5_ee를 최적화 변수로 포함시켜야 함")

if __name__ == "__main__":
    main()
