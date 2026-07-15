import sys

with open('/home/rainbow/camera_ws/main_ui.py', 'r') as f:
    content = f.read()

old_zero_pose_action = '''        # Top row: Zero Pose Check, Stop (Home Offset Reset and Camera Feed excluded — already in Step 1)
        top_action_row = QHBoxLayout()
        self.btn_step2_zero_pose = QPushButton("Zero Pose Check")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        top_action_row.addWidget(self.btn_step2_zero_pose)

        self.btn_step2_stop = QPushButton("Stop")'''

new_zero_pose_action = '''        # Top row: Stop
        top_action_row = QHBoxLayout()

        self.btn_step2_stop = QPushButton("Stop")'''

if old_zero_pose_action in content:
    content = content.replace(old_zero_pose_action, new_zero_pose_action)
else:
    print("Warning: Task 6 zero pose action pattern not found.")

with open('/home/rainbow/camera_ws/main_ui.py', 'w') as f:
    f.write(content)

print("Patch zero pose applied.")
