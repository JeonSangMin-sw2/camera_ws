import sys
import os
import cv2
import numpy as np
import time
from scipy.optimize import least_squares

# marker_detection.py가 있는 부모 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import Marker_Transform
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

def mat2rpy_zyx(R):
    """
    Extract Roll, Pitch, Yaw from a rotation matrix using ZYX convention (R = Rz * Ry * Rx).
    Returns angles in degrees: [roll, pitch, yaw]
    """
    # R = [[r00, r01, r02],
    #      [r10, r11, r12],
    #      [r20, r21, r22]]
    
    # yaw = atan2(r10, r00)
    # pitch = atan2(-r20, sqrt(r21^2 + r22^2))
    # roll = atan2(r21, r22)
    
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return np.degrees([roll, pitch, yaw])

def fit_circle_3d_robust(points):
    """
    선형 대수적 피팅(초기값) + 비선형 기하학적 피팅(최적화)을 결합한 고정밀 3D 원 피팅 알고리즘
    Returns: center_3d (3D), normal (rotation axis), radius, rmse
    """
    points = np.array(points)
    
    # 1. 평면 피팅 (SVD) 및 데이터 중심화
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    
    _, _, vh = np.linalg.svd(pts_centered)
    normal = vh[2, :]
    if normal[2] < 0:
        normal = -normal

    # 2. 2D 투영 (SVD의 vh[0], vh[1]은 이미 평면상의 완벽한 직교 기저 벡터입니다)
    ex = vh[0, :]
    ey = vh[1, :]
    
    # 내적을 통한 2D 좌표 변환
    pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)

    # 3. 1차 추정: 선형 대수적 피팅 (Algebraic Fit - 기존 코드 방식)
    A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
    b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc_init, vc_init, offset_init = res[0], res[1], res[2]
    
    radius_init = np.sqrt(max(0, offset_init + uc_init**2 + vc_init**2))
    initial_guess = [uc_init, vc_init, radius_init]

    # 4. 2차 최적화: 비선형 기하학적 피팅 (Geometric Fit)
    def residuals(params, xy):
        uc, vc, R = params
        # 각 데이터 포인트에서 중심까지의 거리 계산
        distances = np.sqrt((xy[:, 0] - uc)**2 + (xy[:, 1] - vc)**2)
        # 실제 반지름과의 기하학적 거리 오차 반환
        return distances - R

    # Huber loss를 사용하여 튀는 데이터(Outlier)의 영향력을 감소시킴
    opt_result = least_squares(residuals, initial_guess, args=(pts_2d,), loss='huber')
    uc_opt, vc_opt, radius_opt = opt_result.x

    # 5. 최적화된 2D 중심을 3D로 복원
    center_3d = centroid + uc_opt * ex + vc_opt * ey
    
    # 6. RMSE 계산
    final_residuals = residuals(opt_result.x, pts_2d)
    rmse = np.sqrt(np.mean(final_residuals**2))
    
    return center_3d, normal, radius_opt, rmse

def main():
    print("\n" + "="*50)
    print("   Marker Bracket Calibration Tool")
    print("="*50)
    print("Instructions:")
    print("  1. Position the marker in the camera view.")
    print("  2. Rotate the bracket to different positions.")
    print("  3. Press 'c' to capture the current marker position.")
    print("  4. After 3 points, calibration results will be shown.")
    print("  5. Collect up to 10 points for a refined fit.")
    print("  6. Press 'q' or 'ESC' to quit.")
    print("="*50 + "\n")

    # Initialize Marker_Transform (which handles camera and detector initialization)
    try:
        marker_st = Marker_Transform()
        # Ensure we are looking for the correct marker type
        # Defaulting to plate type as it's common for brackets
        marker_st.marker_detection.set_marker_type("plate")
    except Exception as e:
        print(f"Failed to initialize camera/marker system: {e}")
        return

    captured_poses = [] # List to store captured 4x4 marker poses
    
    try:
        while True:
            # Step 1: Capture and update images
            # Marker_Transform.get_marker_transform with sampling_time=0 captures a single frame
            # internally calling camera.capture_image() if not already monitoring
            results = marker_st.get_marker_transform(sampling_time=0, side="all")
            
            # For visualization, we need the raw image
            color_img = marker_st.camera.get_color_image()
            if color_img is None:
                time.sleep(0.01)
                continue

            display_img = color_img.copy()
            
            # Check if any marker was detected in this frame
            current_pose = None
            if results and len(results) > 0:
                # results is a dict or list depending on side argument and marker_type
                # get_marker_transform(side="all") returns a list of 16-element lists for "plate"
                # Let's take the first one
                if isinstance(results, list):
                    pose_ref = results[0]
                    current_pose = np.array(pose_ref).reshape(4, 4)
                elif isinstance(results, dict):
                    # For other types, results might be a dict
                    first_key = list(results.keys())[0]
                    current_pose = np.array(results[first_key]).reshape(4, 4)
                
                # Unit Conversion: m to mm
                current_pose[:3, 3] *= 1000.0

            # UI overlays
            cv2.putText(display_img, f"Captured Points: {len(captured_poses)}/10", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if current_pose is not None:
                cv2.putText(display_img, "MARKER DETECTED", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_img, "NO MARKER", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Marker Bracket Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF

            # Handle 'c' key for capture
            if key == ord('c'):
                print("\n[INFO] Capturing with LPF (Sampling 1.0s)... Please hold the bracket still.")
                # Call get_marker_transform with sampling_time > 0 for robust averaged result
                lpf_results = marker_st.get_marker_transform(sampling_time=1.0, side="all")
                
                captured_pose = None
                if lpf_results and len(lpf_results) > 0:
                    if isinstance(lpf_results, list):
                        captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                    elif isinstance(lpf_results, dict):
                        first_key = list(lpf_results.keys())[0]
                        captured_pose = np.array(lpf_results[first_key]).reshape(4, 4)

                if captured_pose is not None:
                    # Unit Conversion: m to mm
                    captured_pose[:3, 3] *= 1000.0
                    
                    captured_poses.append(captured_pose.copy())
                    print(f"[{len(captured_poses)}] Captured filtered point (Camera Frame): {np.round(captured_pose[:3, 3], 2)}")
                    
                    if len(captured_poses) >= 3:
                        # Use the first captured pose as the reference frame
                        T_cam_ref = captured_poses[0]
                        T_ref_cam = np.linalg.inv(T_cam_ref)
                        
                        # Transform all captured points relative to the first captured pose
                        relative_poses = [T_ref_cam @ T for T in captured_poses]
                        points = [T[:3, 3] for T in relative_poses]
                        
                        center, axis, radius, rmse = fit_circle_3d_robust(points)
                        
                        # Calculate Fitting Score (0-100%)
                        # RMSE = 0mm -> 100%, RMSE = 4mm -> 0%
                        fitting_score = max(0.0, 100.0 * (1.0 - rmse / 4.0))
                        
                        # Calculate Tilt: angle between marker Z-axis and rotation axis
                        # We use the latest relative pose for tilt calculation
                        current_relative_pose = relative_poses[-1]
                        marker_z = current_relative_pose[:3, 2]
                        # Normalized dot product
                        dot_val = np.dot(marker_z, axis)
                        # Angle is between 0 and 90.
                        tilt_angle = np.degrees(np.arccos(min(1.0, max(-1.0, abs(dot_val)))))
                        
                        # Calculate RPY (ZYX) for the current marker orientation relative to the first one
                        rpy = mat2rpy_zyx(current_relative_pose[:3, :3])
                        
                        print(f"\n--- Calibration Update (N={len(captured_poses)}) ---")
                        print(f"  Reference Point: First Captured Marker")
                        print(f"  Rotation Center (Relative, mm): X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
                        print(f"  Rotation Axis (Normal): [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
                        print(f"  Distance to Axis (Radius, mm): {radius:.2f}")
                        print(f"  Fitting Quality Score (%): {fitting_score:.1f}%")
                        print(f"  Marker Tilt vs Axis (deg): {tilt_angle:.2f}")
                        print(f"  Marker RPY (Relative, deg): Roll={rpy[0]:.2f}, Pitch={rpy[1]:.2f}, Yaw={rpy[2]:.2f}")
                        print("-" * 40)
                        
                        if len(captured_poses) >= 10:
                            print("\n[INFO] Collected 10 points. Calibration complete.")
                else:
                    print("[WARN] No marker detected. Capture failed.")

            elif key == ord('q') or key == 27: # 'q' or ESC
                print("\nExiting calibration.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        marker_st.camera.stream_off()
        cv2.destroyAllWindows()
        print("Camera resource released.")

if __name__ == "__main__":
    main()
