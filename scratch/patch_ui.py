import sys

with open('/home/rainbow/camera_ws/main_ui.py', 'r') as f:
    content = f.read()

# Patch 1: Remove btn_full_auto_ready
content = content.replace(
    'self.btn_full_auto_ready = QPushButton("MOVE TO READY")',
    '# self.btn_full_auto_ready = QPushButton("MOVE TO READY")'
)
content = content.replace(
    'self.btn_full_auto_ready.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")',
    '# self.btn_full_auto_ready.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")'
)
content = content.replace(
    'self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)',
    '# self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)'
)
content = content.replace(
    'full_auto_sublayout.addWidget(self.btn_full_auto_ready)',
    '# full_auto_sublayout.addWidget(self.btn_full_auto_ready)'
)

# Patch 2: Automate Intrinsic calibration
old_auto = 'self.log_msg(f"[INTRINSICS] All {num_steps} guided frames captured! You can now run calibration.")'
new_auto = 'self.log_msg(f"[INTRINSICS] All {num_steps} guided frames captured! Automatically running calibration...")\n                    self.run_intrinsics_calibration()'
content = content.replace(old_auto, new_auto)

# Patch 3: Verification dialog button and logic
old_btn = 'btn_close = QPushButton("CLOSE")'
new_btn = 'btn_close = QPushButton("SAVE PARAMETERS")'
content = content.replace(old_btn, new_btn)

old_color = 'btn_close.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; font-size: 13px;")'
new_color = 'btn_close.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 13px;")'
content = content.replace(old_color, new_color)

old_exec = """            dialog.exec()
            self.log_msg(f"[INTRINSICS] Verification dialog shown. Image saved to: {save_path}")"""
new_exec = """            if dialog.exec() == QDialog.Accepted:
                self.save_intrinsics_calibration()
            self.log_msg(f"[INTRINSICS] Verification dialog shown. Image saved to: {save_path}")"""
content = content.replace(old_exec, new_exec)

with open('/home/rainbow/camera_ws/main_ui.py', 'w') as f:
    f.write(content)

print("Patch applied successfully.")
