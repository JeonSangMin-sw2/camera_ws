import sys
import os
import cv2
import time
import numpy as np

# marker_detection.py가 있는 부모 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import RealSenseCamera
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

from IntrinsicsCalibrator import IntrinsicsCalibrator

def main():
    output_yaml = os.path.join(os.path.dirname(__file__), "..", "config", "camera_intrinsics.yaml")
    calibrator = IntrinsicsCalibrator()
    
    # Charuco 보드 설정 예시 (8x5 squares, 30mm x 15mm, DICT_5X5_100)
    # 필요한 경우 설정을 변경하세요.
    calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 30.0, 22.0, "DICT_5X5_100")

    cam = RealSenseCamera()
    cam.initialize_camera(1280, 720, 30)

    print("\n--- Single Camera Intrinsics Calibration ---")
    print("Instructions:")
    print("  1) Show the calibration board to the camera.")
    print("  2) Press 'c' to capture a frame in memory.")
    print("  3) Press 'ENTER' to run calibration.")
    print("  4) Press 'q' or 'ESC' to exit.")
    print("-------------------------------------------\n")

    captured_images = []
    
    try:
        while True:
            cam.capture_image()
            img = cam.get_color_image()
            if img is None:
                time.sleep(0.01)
                continue

            display_img = img.copy()
            
            # Simple board detection for preview
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD:
                detector = cv2.aruco.CharucoDetector(calibrator.charuco_board)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
                if charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
            elif calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHESSBOARD:
                ret, corners = cv2.findChessboardCorners(gray, calibrator.board_size, None)
                if ret:
                    cv2.drawChessboardCorners(display_img, calibrator.board_size, corners, ret)

            cv2.putText(display_img, f"Captured: {len(captured_images)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Intrinsics Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                captured_images.append(img.copy())
                print(f"[{len(captured_images)}] Frame captured in memory.")

            elif key == 13: # ENTER key
                if len(captured_images) >= 5:
                    print(f"\nRunning calibration on {len(captured_images)} captured images...")
                    success = calibrator.run_calibration_with_images(captured_images, output_yaml)
                    
                    if success and len(captured_images) > 0:
                        # Get detailed data for verification
                        test_img = captured_images[-1]
                        h, w = test_img.shape[:2]
                        
                        # 1. Calculate reprojection error for the last image
                        rvec = calibrator.rvecs[-1]
                        tvec = calibrator.tvecs[-1]
                        obj_pts = calibrator.all_obj_points[-1]
                        img_pts = calibrator.all_img_points[-1]
                        
                        projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, calibrator.cameraMatrix, calibrator.distCoeffs)
                        err = cv2.norm(img_pts, projected_pts, cv2.NORM_L2) / len(projected_pts)
                        
                        # 2. Points Fit Visualization (Original Image)
                        fit_img = test_img.copy()
                        for i in range(len(img_pts)):
                            # Detected: Green Circle
                            cv2.circle(fit_img, (int(img_pts[i][0][0]), int(img_pts[i][0][1])), 4, (0, 255, 0), 1)
                            # Reprojected: Red Cross
                            p = (int(projected_pts[i][0][0]), int(projected_pts[i][0][1]))
                            cv2.drawMarker(fit_img, p, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

                        cv2.putText(fit_img, f"Total RMS: {calibrator.rms_error:.4f}", (30, 40), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(fit_img, f"Frame Error: {err:.4f}", (30, 80), 1, 1.5, (0, 255, 255), 2)
                        cv2.putText(fit_img, "GREEN:Detected, RED:Reprojected", (30, 120), 1, 1.2, (255, 255, 255), 1)

                        # 3. Undistortion Verification (Grid)
                        new_mtx, _ = cv2.getOptimalNewCameraMatrix(calibrator.cameraMatrix, calibrator.distCoeffs, (w, h), 1, (w, h))
                        undistorted = cv2.undistort(test_img, calibrator.cameraMatrix, calibrator.distCoeffs, None, new_mtx)
                        
                        grid_view = undistorted.copy()
                        for y in range(0, h, 60):
                            cv2.line(grid_view, (0, y), (w, y), (0, 255, 0), 1)
                        for x in range(0, w, 60):
                            cv2.line(grid_view, (x, 0), (x, h), (0, 255, 0), 1)
                        cv2.putText(grid_view, "Undistorted Grid View", (30, 40), 1, 1.5, (0, 255, 0), 2)

                        # 4. Linearity Verification (Board Corners)
                        linearity_img = undistorted.copy()
                        if len(calibrator.all_ids) > 0:
                            ids_flat = calibrator.all_ids[-1].flatten()
                            raw_pts = calibrator.all_img_points[-1]
                            # Undistort the specific corners detected in this frame
                            undist_pts = cv2.undistortPoints(raw_pts, calibrator.cameraMatrix, calibrator.distCoeffs, None, new_mtx).reshape(-1, 2)
                            
                            num_x = calibrator.board_size[0] - 1
                            rows = {}
                            cols = {}
                            
                            for i, cid in enumerate(ids_flat):
                                r, c = divmod(cid, num_x)
                                if r not in rows: rows[r] = []
                                if c not in cols: cols[c] = []
                                rows[r].append(undist_pts[i])
                                cols[c].append(undist_pts[i])
                            
                            # Draw Row Lines (Blue)
                            for r_idx in rows:
                                pts = rows[r_idx]
                                if len(pts) >= 2:
                                    pts = sorted(pts, key=lambda p: p[0])
                                    cv2.line(linearity_img, tuple(pts[0].astype(int)), tuple(pts[-1].astype(int)), (255, 0, 0), 1)
                            
                            # Draw Col Lines (Red)
                            for c_idx in cols:
                                pts = cols[c_idx]
                                if len(pts) >= 2:
                                    pts = sorted(pts, key=lambda p: p[1])
                                    cv2.line(linearity_img, tuple(pts[0].astype(int)), tuple(pts[-1].astype(int)), (0, 0, 255), 1)
                            
                            # Draw Points (Green)
                            for p in undist_pts:
                                cv2.circle(linearity_img, tuple(p.astype(int)), 3, (0, 255, 0), -1)
                        
                        cv2.putText(linearity_img, "Linearity Proof (Rows/Cols)", (30, 40), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(linearity_img, "Dots should align with lines if flat.", (30, 80), 1, 1.0, (255, 255, 255), 1)

                        # 5. Display Combined View
                        top_row = np.hstack((fit_img, grid_view))
                        bottom_row = np.hstack((undistorted, linearity_img))
                        combined = np.vstack((top_row, bottom_row))
                        
                        display_w = 1200
                        display_h = int(display_w * (combined.shape[0] / combined.shape[1]))
                        combined_res = cv2.resize(combined, (display_w, display_h))
                        
                        print(f"\n[Verification] Total RMS: {calibrator.rms_error:.4f}, Last Frame Error: {err:.4f}")
                        cv2.imshow("Calibration Verification (Press any key to continue)", combined_res)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Calibration Verification (Press any key to continue)")
                else:
                    print("Not enough images captured to run calibration. (Minimum 5 required)")

            elif key == ord('q') or key == 27: # q or ESC
                break
    finally:
        cam.stream_off()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
