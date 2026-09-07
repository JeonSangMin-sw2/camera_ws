import time
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def fit_circle_plane_normal(pts):
    """
    3D 점들의 모음에서 원이 놓인 평면의 법선 벡터(Normal Vector)와 중심을 SVD로 추출
    """
    c0 = np.mean(pts, axis=0)
    _, _, vt = np.linalg.svd(pts - c0)
    normal = vt[2]
    # 회전 방향의 일관성을 위해 Z성분이 양수가 되도록 부호 조정
    return normal / np.linalg.norm(normal), c0

def run_simulation_head_sweep_test():
    print("=== [TEST] Step 1.5 Head-Camera Pre-calibration in RBY1 Simulation ===")
    
    robot = rby.create_robot("127.0.0.1:50051", "m")
    if not robot.connect(1):
        print("[ERROR] Cannot connect to simulator at 127.0.0.1:50051")
        return
    print("[SUCCESS] Connected to RBY1 Simulator.")

    dyn = robot.get_dynamics()
    model = robot.model()
    
    # 1. 시뮬레이션에 주입할 '가상의 실제 조립 오차 (Ground Truth Offsets)' 정의
    # 카메라 브라켓 조립 틸트 오차 (Roll +1.5 deg, Pitch -1.0 deg, Yaw 0.0 deg)
    gt_cam_rpy_deg = np.array([1.5, -1.0, 0.0]) # in mount frame
    gt_cam_rot_error = R_scipy.from_euler('xyz', gt_cam_rpy_deg, degrees=True).as_matrix()
    
    # 헤드 관절 영점 오차 (Pan +0.8 deg, Tilt -1.2 deg)
    gt_head_offset_deg = np.array([0.8, -1.2])
    gt_head_offset_rad = gt_head_offset_deg * D2R
    
    # 숄더 관절 영점 오차 (J0 +1.2 deg, J1 -0.8 deg)
    gt_arm_offset_deg = np.array([1.2, -0.8, 0.5, 0.0, 0.0, 0.0, 0.0])
    gt_arm_offset_rad = gt_arm_offset_deg * D2R
    
    print("\n[Ground Truth Injected Offsets]")
    print(f"  Camera Bracket Assembly Tilt : Roll={gt_cam_rpy_deg[0]:+.2f}°, Pitch={gt_cam_rpy_deg[1]:+.2f}°, Yaw={gt_cam_rpy_deg[2]:+.2f}°")
    print(f"  Head Joint Zero Offsets      : Pan={gt_head_offset_deg[0]:+.2f}°, Tilt={gt_head_offset_deg[1]:+.2f}°")
    print(f"  Shoulder Joint Zero Offsets  : R_J0={gt_arm_offset_deg[0]:+.2f}°, R_J1={gt_arm_offset_deg[1]:+.2f}°")

    # 2. 공칭 CAD 파라미터 (Nominal)
    # setting.yaml 기준 mount_to_cam_nom: Euler ZYX [-90, 0, -90]
    R_nom = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    p_nom = np.array([0.047, 0.009, 0.057])
    
    # 실제 장착 회전 (공칭치 @ 가상 조립오차)
    R_cam_actual = R_nom @ gt_cam_rot_error

    # 3. 양팔을 가슴 앞 Ready Pose에 고정 (마커 위치 생성)
    # 팔은 전혀 움직이지 않음
    q_full = np.zeros(dyn.get_dof())
    # torso_angles
    for idx, val in zip(model.torso_idx, np.radians([0, 30, -60, 30, 0, 0])):
        q_full[idx] = val
    # arm ready pose + 실제 관절 오프셋
    q_r_ready = np.radians([-45, -30, 0, -90, 0, 45, 0]) + gt_arm_offset_rad
    for idx, val in zip(model.right_arm_idx, q_r_ready):
        q_full[idx] = val
        
    state = dyn.make_state(["base", "ee_right", "link_head_2"], model.robot_joint_names)
    state.set_q(q_full)
    dyn.compute_forward_kinematics(state)
    
    # 마커의 절대 위치 in base frame (고정된 3D 점)
    T_base_to_ee = dyn.compute_transformation(state, 0, 1)
    # Tf_to_marker (마커가 툴에 약간 기울어져 붙어있다고 가정해도 상관없음!)
    T_f_to_m = np.eye(4)
    T_f_to_m[:3, 3] = [0.0, -0.054, -0.048]
    p_marker_base = (T_base_to_ee @ T_f_to_m)[:3, 3]
    print(f"\nStationary Marker in Base Frame: {np.round(p_marker_base * 1000, 1)} mm (Fixed Anchor)")

    # 4. [Step 1.5-A] Tilt 단독 스윕 (Pan=0 고정, Tilt만 -10° ~ +10° 회전)
    tilt_angles_cmd = np.linspace(-10, 10, 11) * D2R
    pts_tilt_cam = []
    
    for t_cmd in tilt_angles_cmd:
        # 실제 헤드 관절각 = 명령각 + 헤드 오프셋
        q_full[model.head_idx[0]] = 0.0 + gt_head_offset_rad[0]
        q_full[model.head_idx[1]] = t_cmd + gt_head_offset_rad[1]
        
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        T_base_to_mount = dyn.compute_transformation(state, 0, 2)
        
        # 실제 카메라 좌표계 위치
        T_base_to_cam = np.eye(4)
        T_base_to_cam[:3, :3] = T_base_to_mount[:3, :3] @ R_cam_actual
        T_base_to_cam[:3, 3] = T_base_to_mount[:3, 3] + T_base_to_mount[:3, :3] @ p_nom
        
        # 카메라 입장에서 본 마커 좌표
        p_m_cam = np.linalg.inv(T_base_to_cam[:3, :3]) @ (p_marker_base - T_base_to_cam[:3, 3])
        pts_tilt_cam.append(p_m_cam)
        
    pts_tilt_cam = np.array(pts_tilt_cam)
    n_tilt_cam, c_tilt = fit_circle_plane_normal(pts_tilt_cam)

    # 5. [Step 1.5-B] Pan 단독 스윕 (Tilt=0 고정, Pan만 -15° ~ +15° 회전)
    pan_angles_cmd = np.linspace(-15, 15, 11) * D2R
    pts_pan_cam = []
    
    for p_cmd in pan_angles_cmd:
        q_full[model.head_idx[0]] = p_cmd + gt_head_offset_rad[0]
        q_full[model.head_idx[1]] = 0.0 + gt_head_offset_rad[1]
        
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        T_base_to_mount = dyn.compute_transformation(state, 0, 2)
        
        T_base_to_cam = np.eye(4)
        T_base_to_cam[:3, :3] = T_base_to_mount[:3, :3] @ R_cam_actual
        T_base_to_cam[:3, 3] = T_base_to_mount[:3, 3] + T_base_to_mount[:3, :3] @ p_nom
        
        p_m_cam = np.linalg.inv(T_base_to_cam[:3, :3]) @ (p_marker_base - T_base_to_cam[:3, 3])
        pts_pan_cam.append(p_m_cam)
        
    pts_pan_cam = np.array(pts_pan_cam)
    n_pan_cam, c_pan = fit_circle_plane_normal(pts_pan_cam)

    print("\n--- 3D Circle Fitting Results in Camera Frame ---")
    print("Measured Tilt Normal in Camera Frame:", np.round(n_tilt_cam, 4))
    print("Measured Pan  Normal in Camera Frame:", np.round(n_pan_cam, 4))

    # 6. 두 법선 벡터로부터 카메라 회전 행렬 R_cam_est 복원
    # 공칭 축: Tilt = [-1, 0, 0], Pan = [0, -1, 0]
    # 부호 일관성 맞추기
    if n_tilt_cam[0] > 0: n_tilt_cam = -n_tilt_cam
    if n_pan_cam[1] > 0: n_pan_cam = -n_pan_cam
    
    # 직교화 (Gram-Schmidt / SVD)
    z_axis_cam = np.cross(n_tilt_cam, n_pan_cam)
    z_axis_cam /= np.linalg.norm(z_axis_cam)
    
    y_axis_cam = -n_pan_cam
    x_axis_cam = np.cross(y_axis_cam, z_axis_cam)
    
    # 추정된 R_cam_est
    # Mount frame 기준으로의 R_est 복원
    # R_mount_to_cam_est.T @ [0, 1, 0] = n_tilt_cam
    # R_mount_to_cam_est.T @ [0, 0, 1] = n_pan_cam
    V_cam = np.column_stack([np.cross(n_tilt_cam, n_pan_cam), n_tilt_cam, n_pan_cam])
    V_mount = np.eye(3) # [X_cross, Y_tilt, Z_pan]
    
    U_svd, _, Vt_svd = np.linalg.svd(V_cam @ V_mount.T)
    R_cam_est = (U_svd @ Vt_svd).T # R_mount_to_cam_est
    
    # CAD 공칭치 대비 추정된 조립 틸트 오차 각도
    R_error_est = R_nom.T @ R_cam_est
    rpy_error_est_deg = R_scipy.from_matrix(R_error_est).as_euler('xyz', degrees=True)
    
    print("\n=======================================================")
    print(" 7. ESTIMATION vs GROUND TRUTH COMPARISON")
    print("=======================================================")
    print(f" Camera Roll Error  -> Ground Truth: {gt_cam_rpy_deg[0]:+6.2f}°, Estimated: {rpy_error_est_deg[0]:+6.2f}° (Err: {abs(gt_cam_rpy_deg[0]-rpy_error_est_deg[0]):.4f}°)")
    print(f" Camera Pitch Error -> Ground Truth: {gt_cam_rpy_deg[1]:+6.2f}°, Estimated: {rpy_error_est_deg[1]:+6.2f}° (Err: {abs(gt_cam_rpy_deg[1]-rpy_error_est_deg[1]):.4f}°)")
    print(f" Camera Yaw Error   -> Ground Truth: {gt_cam_rpy_deg[2]:+6.2f}°, Estimated: {rpy_error_est_deg[2]:+6.2f}° (Err: {abs(gt_cam_rpy_deg[2]-rpy_error_est_deg[2]):.4f}°)")
    print("=======================================================")

if __name__ == "__main__":
    run_simulation_head_sweep_test()
