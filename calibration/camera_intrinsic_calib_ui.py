import sys
import os
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QLineEdit, QDialog)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

# Add parent directory to access marker_detection and config
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import RealSenseCamera
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

from IntrinsicsCalibrator import IntrinsicsCalibrator

class IntrinsicCalibApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Camera Intrinsic Calibration UI")
        self.resize(1000, 700)
        
        self.cam = RealSenseCamera()
        self.cam.initialize_camera(1280, 720, 30)
        
        self.calibrator = IntrinsicsCalibrator()
        # Default: 8x5 squares, 30mm x 22mm, DICT_5X5_100
        self.calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 30.0, 22.0, "DICT_5X5_100")
        
        self.captured_images = []
        self.output_yaml = os.path.join(os.path.dirname(__file__), "..", "config", "camera_intrinsics.yaml")
        
        self.init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33) # ~30 fps
        
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # Left Panel (Video & Stats)
        left_panel = QVBoxLayout()
        self.video_label = QLabel("Camera Feed Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        left_panel.addWidget(self.video_label)
        
        stats_box = QGroupBox("Capture Stats")
        stats_layout = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(self.lbl_captured)
        stats_box.setLayout(stats_layout)
        left_panel.addWidget(stats_box)
        
        # Right Panel (Controls & Log)
        right_panel = QVBoxLayout()
        
        controls_box = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        self.btn_capture = QPushButton("CAPTURE FRAME (C)")
        self.btn_capture.setMinimumHeight(50)
        self.btn_capture.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; font-size: 14px;")
        self.btn_capture.clicked.connect(self.capture_frame)
        controls_layout.addWidget(self.btn_capture)
        
        self.btn_calibrate = QPushButton("RUN CALIBRATION")
        self.btn_calibrate.setMinimumHeight(50)
        self.btn_calibrate.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; font-size: 14px;")
        self.btn_calibrate.clicked.connect(self.run_calibration)
        controls_layout.addWidget(self.btn_calibrate)
        
        self.btn_reset = QPushButton("RESET CAPTURES")
        self.btn_reset.setMinimumHeight(30)
        self.btn_reset.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_captures)
        controls_layout.addWidget(self.btn_reset)
        
        controls_box.setLayout(controls_layout)
        right_panel.addWidget(controls_box)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        right_panel.addWidget(self.log_text)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)
        
        self.log_msg("Camera Intrinsic Calibration UI Started.")
        self.log_msg(f"Target pattern: CharucoBoard (8x5, 30mm/22mm)")
        self.log_msg("Please capture at least 5 frames from different angles.")

    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_frame(self):
        self.cam.capture_image()
        img = self.cam.get_color_image()
        if img is None:
            return
            
        self.current_frame = img.copy()
        display_img = img.copy()
        
        # Simple board detection for preview
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        if self.calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD:
            detector = cv2.aruco.CharucoDetector(self.calibrator.charuco_board)
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
        elif self.calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHESSBOARD:
            ret, corners = cv2.findChessboardCorners(gray, self.calibrator.board_size, None)
            if ret:
                cv2.drawChessboardCorners(display_img, self.calibrator.board_size, corners, ret)

        # Convert to QImage and show
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Keep aspect ratio
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.capture_frame()
        super().keyPressEvent(event)

    def capture_frame(self):
        if hasattr(self, 'current_frame'):
            self.captured_images.append(self.current_frame.copy())
            self.lbl_captured.setText(f"Captured Frames: {len(self.captured_images)}")
            self.log_msg(f"[INFO] Frame {len(self.captured_images)} captured.")

    def reset_captures(self):
        self.captured_images.clear()
        self.lbl_captured.setText(f"Captured Frames: 0")
        self.log_msg("[INFO] Capture memory cleared.")

    def run_calibration(self):
        if len(self.captured_images) < 5:
            self.log_msg("[ERROR] Need at least 5 frames to run calibration!")
            return
            
        self.log_msg(f"\n[INFO] Running calibration on {len(self.captured_images)} images. Please wait...")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        success = self.calibrator.run_calibration_with_images(self.captured_images, self.output_yaml)
        
        QApplication.restoreOverrideCursor()
        
        if success:
            self.log_msg(f"[SUCCESS] Calibration complete! RMS Error: {self.calibrator.rms_error:.4f}")
            self.log_msg(f"Saved to: {self.output_yaml}")
            self.show_verification()
        else:
            self.log_msg("[ERROR] Calibration failed. Check images and pattern settings.")

    def show_verification(self):
        if len(self.captured_images) == 0:
            return
            
        test_img = self.captured_images[-1]
        h, w = test_img.shape[:2]
        
        rvec = self.calibrator.rvecs[-1]
        tvec = self.calibrator.tvecs[-1]
        obj_pts = self.calibrator.all_obj_points[-1]
        img_pts = self.calibrator.all_img_points[-1]
        
        projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.calibrator.cameraMatrix, self.calibrator.distCoeffs)
        err = cv2.norm(img_pts, projected_pts, cv2.NORM_L2) / len(projected_pts)
        
        # Fit Image
        fit_img = test_img.copy()
        for i in range(len(img_pts)):
            cv2.circle(fit_img, (int(img_pts[i][0][0]), int(img_pts[i][0][1])), 4, (0, 255, 0), 1)
            p = (int(projected_pts[i][0][0]), int(projected_pts[i][0][1]))
            cv2.drawMarker(fit_img, p, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

        cv2.putText(fit_img, f"Total RMS: {self.calibrator.rms_error:.4f}", (30, 40), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(fit_img, f"Frame Error: {err:.4f}", (30, 80), 1, 1.5, (0, 255, 255), 2)

        # Undistorted Image
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.calibrator.cameraMatrix, self.calibrator.distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, self.calibrator.cameraMatrix, self.calibrator.distCoeffs, None, new_mtx)
        
        grid_view = undistorted.copy()
        for y in range(0, h, 60):
            cv2.line(grid_view, (0, y), (w, y), (0, 255, 0), 1)
        for x in range(0, w, 60):
            cv2.line(grid_view, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(grid_view, "Undistorted Grid View", (30, 40), 1, 1.5, (0, 255, 0), 2)

        # Show in dialog
        top_row = np.hstack((fit_img, grid_view))
        display_w = 1200
        display_h = int(display_w * (top_row.shape[0] / top_row.shape[1]))
        combined_res = cv2.resize(top_row, (display_w, display_h))
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration Verification")
        dl = QVBoxLayout(dialog)
        img_label = QLabel()
        
        # Convert combined_res to QPixmap
        h, w, ch = combined_res.shape
        bytes_per_line = ch * w
        combined_rgb = cv2.cvtColor(combined_res, cv2.COLOR_BGR2RGB)
        qimg = QImage(combined_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        img_label.setPixmap(QPixmap.fromImage(qimg))
        
        dl.addWidget(img_label)
        dialog.exec()

    def closeEvent(self, event):
        self.cam.stream_off()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ex = IntrinsicCalibApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
