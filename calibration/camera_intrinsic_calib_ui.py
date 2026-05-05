import sys
import os
import cv2
import numpy as np
import argparse

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QLineEdit, QDialog)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

# Add parent directory to access marker_detection and config
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import RealSenseCamera, Marker_Detection
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

from IntrinsicsCalibrator import IntrinsicsCalibrator

class IntrinsicCalibApp(QWidget):
    def __init__(self, ui_only=False):
        super().__init__()
        
        self.setWindowTitle("Camera Intrinsic Calibration UI")
        self.resize(1200, 800)
        
        self.ui_only = ui_only
        self.cam = None
        if not self.ui_only:
            try:
                self.cam = RealSenseCamera()
                self.cam.initialize_camera(1280, 720, 30)
            except Exception as e:
                print(f"Camera Init Error: {e}")
                self.ui_only = True
        
        self.calibrator = IntrinsicsCalibrator()
        # Default: 8x5 squares, 30mm x 22mm, DICT_5X5_100
        self.calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 30.0, 22.0, "DICT_5X5_100")
        
        # Marker Detection for monitoring
        self.marker_detector = Marker_Detection()
        self.marker_detector.set_marker_type("plate") # Default to plate
        self.monitor_enabled = False
        
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
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #555;")
        left_panel.addWidget(self.video_label)
        
        # Instructions Panel (English)
        instr_box = QGroupBox("Calibration Guidelines")
        instr_layout = QVBoxLayout()
        instructions = [
            "1. Ensure the calibration board is recognized correctly.",
            "2. Tilt the board at various angles while capturing.",
            "3. Acquire data covering the entire camera field of view.",
            "4. Keep the board as steady as possible during each capture."
        ]
        for text in instructions:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: black; font-weight: bold;")
            instr_layout.addWidget(lbl)
        instr_box.setLayout(instr_layout)
        left_panel.addWidget(instr_box)

        stats_box = QGroupBox("Capture Stats")
        stats_layout = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(self.lbl_captured)
        stats_box.setLayout(stats_layout)
        left_panel.addWidget(stats_box)
        
        # Right Panel (Controls & Log)
        right_panel = QVBoxLayout()
        
        # 3D Marker Monitor Panel
        monitor_box = QGroupBox("Marker 3D Monitoring")
        monitor_layout = QVBoxLayout()
        
        side_layout = QHBoxLayout()
        side_layout.addWidget(QLabel("Arm Side:"))
        self.side_sel = QComboBox()
        self.side_sel.addItems(["Left", "Right"])
        side_layout.addWidget(self.side_sel)
        monitor_layout.addLayout(side_layout)
        
        self.btn_monitor = QPushButton("ENABLE MONITORING")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
        self.btn_monitor.toggled.connect(self.toggle_monitoring)
        monitor_layout.addWidget(self.btn_monitor)
        
        self.lbl_marker_pos = QLabel("X: 0.0, Y: 0.0, Z: 0.0 (mm)")
        self.lbl_marker_pos.setAlignment(Qt.AlignCenter)
        self.lbl_marker_pos.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d4ff; background-color: #111; padding: 5px;")
        monitor_layout.addWidget(self.lbl_marker_pos)
        
        monitor_box.setLayout(monitor_layout)
        right_panel.addWidget(monitor_box)

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
        if self.ui_only:
            self.log_msg("[DEBUG] UI-Only Mode enabled. Camera is mocked.")
        self.log_msg(f"Target pattern: CharucoBoard (8x5, 30mm/22mm)")
        self.log_msg("Please capture at least 5 frames from different angles.")

    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def toggle_monitoring(self, checked):
        self.monitor_enabled = checked
        if checked:
            self.btn_monitor.setText("STOP MONITORING")
            self.btn_monitor.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        else:
            self.btn_monitor.setText("ENABLE MONITORING")
            self.btn_monitor.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
            self.lbl_marker_pos.setText("X: 0.0, Y: 0.0, Z: 0.0 (mm)")

    def update_frame(self):
        if not self.ui_only:
            self.cam.capture_image()
            img = self.cam.get_color_image()
        else:
            # Mock image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "UI-ONLY MODE", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)

        if img is None:
            return
            
        self.current_frame = img.copy()
        display_img = img.copy()
        
        # Simple board detection for preview
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Intrinsics Board Detection
        if self.calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD:
            detector = cv2.aruco.CharucoDetector(self.calibrator.charuco_board)
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
        elif self.calibrator.pattern == IntrinsicsCalibrator.BoardPattern.CHESSBOARD:
            ret, corners = cv2.findChessboardCorners(gray, self.calibrator.board_size, None)
            if ret:
                cv2.drawChessboardCorners(display_img, self.calibrator.board_size, corners, ret)

        # 2. Marker Monitoring
        if self.monitor_enabled:
            # Set intrinsics to detector if calibrated, else use defaults from cam
            if self.calibrator.cameraMatrix is not None:
                self.marker_detector.fx = self.calibrator.cameraMatrix[0, 0]
                self.marker_detector.fy = self.calibrator.cameraMatrix[1, 1]
                self.marker_detector.principal_point = [self.calibrator.cameraMatrix[0, 2], self.calibrator.cameraMatrix[1, 2]]
                self.marker_detector.dist_coeffs = self.calibrator.distCoeffs
            elif not self.ui_only:
                self.marker_detector.fx = self.cam.fx
                self.marker_detector.fy = self.cam.fy
                self.marker_detector.principal_point = self.cam.principal_point
                self.marker_detector.dist_coeffs = self.cam.dist_coeffs

            side = self.side_sel.currentText().lower()
            # Define marker IDs to look for based on side
            target_ids = [10, 11, 12, 13, 14] if side == "left" else [30, 31, 32, 33, 34]
            self.marker_detector.marker_id = target_ids
            
            res = self.marker_detector.detect(img.copy())
            if res and len(res) > 0:
                # Use the first detected marker pose
                T = res[0]
                x, y, z = T[:3, 3] * 1000.0 # m to mm
                self.lbl_marker_pos.setText(f"X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f} (mm)")
                # Draw marker on preview
                cv2.drawMarker(display_img, (640, 360), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            else:
                self.lbl_marker_pos.setText("Marker Not Detected")

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
        if self.cam:
            self.cam.stream_off()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Camera Intrinsic Calibration GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = IntrinsicCalibApp(ui_only=args.ui)
    ex.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
