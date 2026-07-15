import re
import sys

with open('/home/rainbow/camera_ws/main_ui.py', 'r') as f:
    lines = f.readlines()

def find_line_index(lines, substring):
    for i, line in enumerate(lines):
        if substring in line:
            return i
    return -1

# Task 1 & 4: Intrinsics Verification Dialog Upgrade
# Let's locate the dialog definition in show_intrinsics_verification
idx = find_line_index(lines, 'btn_close = QPushButton("CLOSE")')
if idx != -1:
    # Need to replace the layout code entirely for the dialog
    # We find the start of the layout definition
    start_idx = find_line_index(lines, 'dialog.setStyleSheet(DARK_STYLESHEET)')
    end_idx = find_line_index(lines[start_idx:], 'self.log_msg(f"[INTRINSICS] Verification dialog shown. Image saved to: {save_path}")')
    
    if start_idx != -1 and end_idx != -1:
        end_idx += start_idx
        # We replace from start_idx + 1 to end_idx
        new_block = """
            main_layout = QVBoxLayout(dialog)
            
            content_layout = QHBoxLayout()

            # Left side: Image
            pixmap = QPixmap(save_path)
            scaled_pix = pixmap.scaled(1000, 750, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(scaled_pix)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 2px solid #2d2d2d; border-radius: 6px;")
            content_layout.addWidget(img_label, stretch=3)

            # Right side: Parameters
            param_layout = QVBoxLayout()
            param_title = QLabel("Calibrated Intrinsic Parameters")
            param_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffeb3b;")
            param_layout.addWidget(param_title)
            
            if hasattr(self, 'intrinsics_calibrator') and self.intrinsics_calibrator.camera_matrix is not None:
                mtx = self.intrinsics_calibrator.camera_matrix
                dist = self.intrinsics_calibrator.dist_coeffs
                param_text = (
                    f"Focal Length (fx): {mtx[0,0]:.4f}\\n"
                    f"Focal Length (fy): {mtx[1,1]:.4f}\\n\\n"
                    f"Principal Point (cx): {mtx[0,2]:.4f}\\n"
                    f"Principal Point (cy): {mtx[1,2]:.4f}\\n\\n"
                    f"Distortion Coefficients:\\n"
                    f"k1: {dist[0,0]:.4f}\\n"
                    f"k2: {dist[0,1]:.4f}\\n"
                    f"p1: {dist[0,2]:.4f}\\n"
                    f"p2: {dist[0,3]:.4f}\\n"
                    f"k3: {dist[0,4]:.4f}\\n"
                )
                param_label = QLabel(param_text)
                param_label.setStyleSheet("font-size: 14px; font-family: monospace; color: white; background-color: #2d2d2d; padding: 10px; border-radius: 5px;")
                param_layout.addWidget(param_label)
            else:
                param_layout.addWidget(QLabel("Parameters not available."))
            
            param_layout.addStretch()
            content_layout.addLayout(param_layout, stretch=1)
            
            main_layout.addLayout(content_layout)

            btn_close = QPushButton("SAVE PARAMETERS")
            btn_close.setMinimumHeight(40)
            btn_close.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 13px;")
            btn_close.clicked.connect(dialog.accept)
            main_layout.addWidget(btn_close)

            if dialog.exec() == QDialog.Accepted:
                self.save_intrinsics_calibration()
"""
        # Note: the spaces and indentation need to be exact. I'll just write it and check it.
        # But wait, it's easier to use replace_file_content if we do it in multiple passes, or I can just use a python script and be very careful.
"""
