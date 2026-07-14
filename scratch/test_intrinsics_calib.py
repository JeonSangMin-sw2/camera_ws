import sys
import numpy as np
import cv2

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

from core.calibration.IntrinsicsCalibrator import IntrinsicsCalibrator

def generate_simulated_points():
    # Ground truth camera matrix
    fx_gt, fy_gt = 800.0, 800.0
    cx_gt, cy_gt = 640.0, 360.0
    K_gt = np.array([
        [fx_gt, 0.0, cx_gt],
        [0.0, fy_gt, cy_gt],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Ground truth distortion: no distortion (so tangential is also 0)
    dist_gt = np.zeros((5, 1), dtype=np.float64)
    
    # Generate 3D board corners (8x5 board, square size 0.04m)
    board_w, board_h = 8, 5
    square_size = 0.04
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= square_size
    
    # Generate 10 different camera-to-board poses (diverse tilts and translations)
    poses = [
        # rvec (rotation vector in rad), tvec (translation vector in meters)
        (np.array([0.0, 0.0, 0.0]), np.array([-0.1, -0.1, 0.6])), # Center flat
        (np.array([0.2, 0.0, 0.1]), np.array([-0.05, -0.05, 0.55])), # Tilt X
        (np.array([-0.2, 0.1, -0.1]), np.array([-0.1, 0.05, 0.65])), # Tilt Y
        (np.array([0.0, 0.3, 0.0]), np.array([0.05, -0.1, 0.58])), # Tilt Z
        (np.array([0.1, -0.2, 0.2]), np.array([0.1, 0.1, 0.62])), # Corner shift
        (np.array([-0.15, -0.15, -0.1]), np.array([-0.15, -0.15, 0.5])), # Close
        (np.array([0.25, 0.25, 0.0]), np.array([0.0, 0.0, 0.7])), # Far
        (np.array([-0.05, 0.05, -0.25]), np.array([-0.08, 0.08, 0.6])), # Shifted
        (np.array([0.05, -0.05, 0.15]), np.array([0.08, -0.08, 0.57])),
        (np.array([0.2, -0.1, -0.05]), np.array([-0.02, 0.03, 0.63]))
    ]
    
    all_obj_points = []
    all_img_points = []
    all_ids = []
    
    # Project 3D points to 2D for each pose
    for idx, (rvec, tvec) in enumerate(poses):
        proj_pts, _ = cv2.projectPoints(objp, rvec, tvec, K_gt, dist_gt)
        # Add very tiny random noise to simulate corner subpix detection jitter (0.02 pixels)
        noise = np.random.normal(0, 0.02, proj_pts.shape).astype(np.float32)
        proj_pts += noise
        
        all_obj_points.append(objp.copy())
        all_img_points.append(proj_pts)
        all_ids.append(np.arange(len(objp)))
        
    return all_obj_points, all_img_points, all_ids, K_gt

def test_intrinsics_calib():
    print("Generating simulated noiseless projection data...")
    all_obj_points, all_img_points, all_ids, K_gt = generate_simulated_points()
    img_size = (1280, 720)
    
    calibrator = IntrinsicsCalibrator()
    calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 0.04, 0.03, "DICT_5X5_100")
    
    print("\n--- Test Case 1: Standard Calibration (No Outliers) ---")
    # Make copies to avoid modification during test
    obj_pts_c = [p.copy() for p in all_obj_points]
    img_pts_c = [p.copy() for p in all_img_points]
    ids_c = [ids.copy() for ids in all_ids]
    
    success = calibrator._calibrate_and_validate(obj_pts_c, img_pts_c, img_size, ids_c)
    
    if not success:
        print("FAIL: Standard calibration failed.")
        sys.exit(1)
        
    print(f"SUCCESS: Calibration RMS: {calibrator.rms_error:.6f} px")
    print(f"Calibrated K:\n{calibrator.cameraMatrix}")
    print(f"Calibrated distortion:\n{calibrator.distCoeffs.flatten()[:5]}")
    
    # Check that focal lengths and principal points are very close to ground truth
    assert np.allclose(calibrator.cameraMatrix[0, 0], K_gt[0, 0], rtol=0.01), "fx mismatch"
    assert np.allclose(calibrator.cameraMatrix[1, 1], K_gt[1, 1], rtol=0.01), "fy mismatch"
    assert np.allclose(calibrator.cameraMatrix[0, 2], K_gt[0, 2], rtol=0.02), "cx mismatch"
    assert np.allclose(calibrator.cameraMatrix[1, 2], K_gt[1, 2], rtol=0.02), "cy mismatch"
    
    # Check zero tangential distortion
    p1, p2 = calibrator.distCoeffs[2, 0], calibrator.distCoeffs[3, 0]
    print(f"Tangential distortion (p1, p2): ({p1}, {p2})")
    assert np.isclose(p1, 0.0) and np.isclose(p2, 0.0), f"Tangential distortion not zero: p1={p1}, p2={p2}"
    print("SUCCESS: Zero tangential distortion verified.")

    print("\n--- Test Case 2: Calibration with Outlier Image ---")
    obj_pts_c2 = [p.copy() for p in all_obj_points]
    img_pts_c2 = [p.copy() for p in all_img_points]
    ids_c2 = [ids.copy() for ids in all_ids]
    
    # Inject large outlier non-rigid noise into the 3rd view (index 2) by adding 3.0 pixels random error
    print("Injecting 3.0 pixels standard deviation random noise into View 3...")
    img_pts_c2[2] = img_pts_c2[2] + np.random.normal(0, 3.0, img_pts_c2[2].shape).astype(np.float32)
    
    success_outlier = calibrator._calibrate_and_validate(obj_pts_c2, img_pts_c2, img_size, ids_c2)
    
    if not success_outlier:
        print("FAIL: Outlier calibration failed.")
        sys.exit(1)
        
    print(f"SUCCESS: Outlier calibration finished. Final RMS: {calibrator.rms_error:.6f} px")
    print(f"Final number of frames used: {len(calibrator.all_obj_points)} (out of {len(all_obj_points)})")
    
    # Verify that the outlier view (index 2) was successfully removed, leaving 9 views
    assert len(calibrator.all_obj_points) == len(all_obj_points) - 1, f"Expected {len(all_obj_points) - 1} views, got {len(calibrator.all_obj_points)}"
    print("SUCCESS: Outlier view successfully identified and filtered out.")

if __name__ == "__main__":
    test_intrinsics_calib()
