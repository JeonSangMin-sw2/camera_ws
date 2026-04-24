import sys
import os
import cv2
import numpy as np
import time
import random
import argparse
import logging
import rby1_sdk as rby
from scipy.optimize import least_squares

# --- Configuration ---
MAX_POINTS = 10 # Number of points to collect for calibration
# ---------------------

# Robot Helper Functions (Copied from 00_helper.py)
def initialize_robot(address, model, power=".*", servo=".*"):
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        sys.exit(1)
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            sys.exit(1)
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            sys.exit(1)
    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            logging.error(f"Failed to reset control manager")
            sys.exit(1)
    if not robot.enable_control_manager():
        logging.error(f"Failed to enable control manager")
        sys.exit(1)
    return robot

def movej(robot, torso=None, right_arm=None, left_arm=None, minimum_time=0):
    rc = rby.BodyComponentBasedCommandBuilder()
    if torso is not None:
        rc.set_torso_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(torso)
        )
    if right_arm is not None:
        rc.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(right_arm)
        )
    if left_arm is not None:
        rc.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(left_arm)
        )

    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(rc)
        ),
        1,
    ).get()

    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        logging.error("Failed to conduct movej.")
        return False

    return True

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
    parser = argparse.ArgumentParser(description="Marker Bracket Calibration Tool")
    parser.add_argument("--address", type=str, help="Robot IP address (optional for manual mode)")
    parser.add_argument("--model", type=str, default="rby1_a", help="Robot model (default: rby1_a)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("   Marker Bracket Calibration Tool")
    print("="*50)
    print("Instructions:")
    print("  1. Position the marker in the camera view.")
    print("  2. If robot is connected, press 'c' to start auto-calibration.")
    print("  3. Otherwise, move the bracket manually and press 'c' to capture.")
    print(f"  4. After 3 points, calibration results will be shown.")
    print(f"  5. Collect up to {MAX_POINTS} points for a refined fit.")
    print("  6. Press 'q' or 'ESC' to quit.")
    print("="*50 + "\n")

    # Robot Initialization
    robot = None
    initial_joint_pos = None
    if args.address:
        try:
            print(f"[INFO] Connecting to robot at {args.address}...")
            robot = initialize_robot(args.address, args.model)
            print("[INFO] Robot initialized successfully.")
        except Exception as e:
            print(f"[WARN] Robot initialization failed: {e}. Running in manual mode.")
            robot = None
    else:
        print("[INFO] No robot address provided. Running in manual mode.")

    # Initialize Marker_Transform (which handles camera and detector initialization)
    try:
        marker_st = Marker_Transform()
        marker_st.marker_detection.set_marker_type("plate")
        print(f"[INFO] Current Marker Size: {marker_st.marker_detection.marker_size_mm} mm")
        print(f"[TIP] If your physical marker size (outer border) is not {marker_st.marker_detection.marker_size_mm}mm, accuracy will be affected.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera/marker system: {e}")
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
            cv2.putText(display_img, f"Captured Points: {len(captured_poses)}/{MAX_POINTS}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if current_pose is not None:
                cv2.putText(display_img, "MARKER DETECTED", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_img, "NO MARKER", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Marker Bracket Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF

            # Handle 'c' key for capture (now with Auto-Loop if robot is connected)
            if key == ord('c'):
                print("\n" + "="*40)
                print("   STARTING AUTOMATED CALIBRATION")
                print("="*40)
                
                # Get initial robot pose if not already done
                if robot and initial_joint_pos is None:
                    state = robot.get_state()
                    initial_joint_pos = list(state.body.right_arm.position)
                    print(f"[INFO] Recorded Initial Right Arm Pose: {initial_joint_pos}")

                while len(captured_poses) < MAX_POINTS:
                    print(f"\n[STEP {len(captured_poses) + 1}/{MAX_POINTS}]")
                    
                    # 1. Capture Marker
                    print("  - Capturing marker with LPF (2.0s)...")
                    lpf_results = marker_st.get_marker_transform(sampling_time=2.0, side="right")
                    
                    captured_pose = None
                    if lpf_results and len(lpf_results) > 0:
                        if isinstance(lpf_results, list):
                            captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                        elif isinstance(lpf_results, dict):
                            first_key = list(lpf_results.keys())[0]
                            captured_pose = np.array(lpf_results[first_key]).reshape(4, 4)

                    if captured_pose is not None:
                        captured_pose[:3, 3] *= 1000.0 # m to mm
                        captured_poses.append(captured_pose.copy())
                        print(f"  - Captured point at: {np.round(captured_pose[:3, 3], 2)}")
                        
                        # Only print coordinates during collection (no full status update)
                        print(f"  - Pose [{len(captured_poses)}/{MAX_POINTS}] Saved: {np.round(captured_pose[:3, 3], 2)}")
                    else:
                        print("  [WARN] Marker not detected. Retrying this step...")
                        continue # Re-try current point

                    if len(captured_poses) >= MAX_POINTS:
                        break # Done
                        
                    # 2. Move Robot to next position
                    if robot and initial_joint_pos:
                        target_joint_pos = list(initial_joint_pos)
                        
                        if len(captured_poses) == MAX_POINTS - 1:
                            # Final point: Return to initial position
                            print(f"  - [FINAL STEP] Returning to initial position for pure misalignment check...")
                        else:
                            # Random move for right_arm_6 (+- 20 deg)
                            random_offset_deg = random.uniform(-20, 20)
                            # right_arm_6 is index 6 (7th joint)
                            target_joint_pos[6] = initial_joint_pos[6] + np.radians(random_offset_deg)
                            print(f"  - Moving right_arm_6 to offset {random_offset_deg:.2f} deg...")
                        
                        if movej(robot, right_arm=target_joint_pos, minimum_time=1.5):
                            print("  - Robot reached target.")
                            time.sleep(0.5) # Settling time
                        else:
                            print("  [ERROR] Robot movement failed.")
                            break
                    else:
                        print("\n[INFO] Manual mode: Please move the bracket and press 'c' for next point.")
                        break # Break the loop to wait for next manual 'c' if robot not connected
                
                if len(captured_poses) >= MAX_POINTS:
                    print("\n" + "="*40)
                    print("   CALIBRATION RESULTS (FINAL)")
                    print("="*40)
                    
                    # Final Processing
                    T_cam_ref = captured_poses[0]
                    T_ref_cam = np.linalg.inv(T_cam_ref)
                    relative_poses = [T_ref_cam @ T for T in captured_poses]
                    points = [T[:3, 3] for T in relative_poses]
                    
                    center, axis, radius, rmse = fit_circle_3d_robust(points)
                    fitting_score = max(0.0, 100.0 * (1.0 - rmse / 4.0))
                    
                    current_relative_pose = relative_poses[-1]
                    marker_y = current_relative_pose[:3, 1]
                    dot_val = np.dot(marker_y, axis)
                    tilt_angle = np.degrees(np.arccos(min(1.0, max(-1.0, abs(dot_val)))))
                    rpy = mat2rpy_zyx(current_relative_pose[:3, :3])
                    
                    abs_axis = np.abs(axis)
                    axis_labels = ["X", "Y", "Z"]
                    primary_axis = axis_labels[np.argmax(abs_axis)]
                    
                    print(f"  Reference Point: First Captured Marker")
                    print(f"  Rotation Center (Relative, mm): X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
                    print(f"  Rotation Axis (In Ref Frame): X={axis[0]:.4f}, Y={axis[1]:.4f}, Z={axis[2]:.4f} (Mainly {primary_axis})")
                    print(f"  Distance to Axis (Radius, mm): {radius:.2f}")
                    print(f"  Fitting Quality Score (%): {fitting_score:.1f}%")
                    print(f"  Marker Tilt vs Axis (deg): {tilt_angle:.2f}")
                    print(f"  Marker RPY (Relative, deg): Roll={rpy[0]:.2f}, Pitch={rpy[1]:.2f}, Yaw={rpy[2]:.2f}")
                    print("="*40)
                    print("\n[FINISH] Automated calibration loop complete.")

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
