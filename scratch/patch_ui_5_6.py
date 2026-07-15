import sys

with open('/home/rainbow/camera_ws/main_ui.py', 'r') as f:
    content = f.read()

# Task 5: Auto calculate
old_auto_finish = '            if success:\n                self.log_msg("Auto motions sequence completed.")\n            else:'
new_auto_finish = '            if success:\n                self.log_msg("Auto motions sequence completed.")\n                self.step2_calculate()\n            else:'
if old_auto_finish in content:
    content = content.replace(old_auto_finish, new_auto_finish)
else:
    print("Warning: Task 5 pattern not found.")

# Task 6: Zero pose button
old_home_btn = '''        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #d84315; color: white; font-weight: bold;")
        self.btn_home_reset.clicked.connect(self.home_offset_reset)
        self.btn_home_reset.setFixedHeight(28)
        home_offset_layout.addWidget(self.btn_home_reset)'''

new_home_btn = '''        btn_row = QHBoxLayout()
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #d84315; color: white; font-weight: bold;")
        self.btn_home_reset.clicked.connect(self.home_offset_reset)
        self.btn_home_reset.setFixedHeight(28)
        btn_row.addWidget(self.btn_home_reset)

        self.btn_step2_zero_pose = QPushButton("Zero Pose")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        btn_row.addWidget(self.btn_step2_zero_pose)
        
        home_offset_layout.addLayout(btn_row)'''

if old_home_btn in content:
    content = content.replace(old_home_btn, new_home_btn)
else:
    print("Warning: Task 6 home btn pattern not found.")

# Now remove the old step2_zero_pose button from actions_layout
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

print("Patch 5_6 applied.")
