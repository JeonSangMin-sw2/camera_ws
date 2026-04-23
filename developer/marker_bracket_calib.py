import sys
import os
import cv2
import numpy as np
import time

# marker_detection.py가 있는 부모 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import Marker_Transform
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

def fit_circle_3d(points):
    """
    Fit a circle to 3D points.
    Returns: center (3D), normal (rotation axis), radius
    """
    points = np.array(points)
    # 1. Plane fitting (SVD)
    # Subtract mean to center the points
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    
    # SVD
    _, _, vh = np.linalg.svd(pts_centered)
    # The normal to the best-fitting plane is the last column of V (or row of VH)
    normal = vh[2, :]
    
    # Ensure normal points somewhat towards the camera (Z positive) or just keep it consistent
    if normal[2] < 0:
        normal = -normal

    # 2. Project points to the plane coordinate system
    # Create local basis (ex, ey, normal)
    if abs(normal[0]) < 0.9:
        ex = np.cross(normal, [1, 0, 0])
    else:
        ex = np.cross(normal, [0, 1, 0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)

    # Project to 2D
    pts_2d = np.zeros((len(points), 2))
    for i, p in enumerate(pts_centered):
        pts_2d[i, 0] = np.dot(p, ex)
        pts_2d[i, 1] = np.dot(p, ey)

    # 3. Fit circle in 2D
    # Equation: (u - uc)^2 + (v - vc)^2 = R^2
    # Linear form: u^2 + v^2 = 2u*uc + 2v*vc + (R^2 - uc^2 - vc^2)
    # Solve A * x = b for x = [2*uc, 2*vc, C] where C = R^2 - uc^2 - vc^2
    A = np.zeros((len(pts_2d), 3))
    b = np.zeros((len(pts_2d), 1))
    for i, (u, v) in enumerate(pts_2d):
        A[i, 0] = 2 * u
        A[i, 1] = 2 * v
        A[i, 2] = 1
        b[i, 0] = u**2 + v**2

    # Least squares fit
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc, vc, offset = res[0, 0], res[1, 0], res[2, 0]
    
    # Calculate radius
    # R^2 = C + uc^2 + vc^2
    r_sq = offset + uc**2 + vc**2
    radius = np.sqrt(max(0, r_sq))

    # 4. Convert center back to 3D
    center_3d = centroid + uc * ex + vc * ey
    
    return center_3d, normal, radius

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
                    captured_poses.append(captured_pose.copy())
                    print(f"[{len(captured_poses)}] Captured filtered point at: {np.round(captured_pose[:3, 3], 2)}")
                    
                    if len(captured_poses) >= 3:
                        # Perform 3D circle fitting
                        points = [T[:3, 3] for T in captured_poses]
                        center, axis, radius = fit_circle_3d(points)
                        
                        # Calculate Tilt: angle between marker Z-axis and rotation axis
                        # We use the latest captured pose for tilt calculation
                        marker_z = current_pose[:3, 2]
                        # Normalized dot product
                        dot_val = np.dot(marker_z, axis)
                        # We use abs(dot) because the axis direction might be inverted
                        # Angle is between 0 and 90.
                        tilt_angle = np.degrees(np.arccos(min(1.0, max(-1.0, abs(dot_val)))))
                        
                        print(f"\n--- Calibration Update (N={len(captured_poses)}) ---")
                        print(f"  Rotation Center (Camera Frame, mm): X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
                        print(f"  Rotation Axis (Normal): [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
                        print(f"  Distance to Axis (Radius, mm): {radius:.2f}")
                        print(f"  Marker Tilt vs Axis (deg): {tilt_angle:.2f}")
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
