import sys

with open('/home/rainbow/camera_ws/main_ui.py', 'r') as f:
    content = f.read()

old_add_tab = 'self.left_tabs.addTab(step1_tab, "Step 1")'
new_add_tab = 'self.left_tabs.addTab(overview_tab, "Overview")\n        self.left_tabs.addTab(step1_tab, "Step 1")'

content = content.replace(old_add_tab, new_add_tab, 1)

with open('/home/rainbow/camera_ws/main_ui.py', 'w') as f:
    f.write(content)

print("Patch overview applied successfully.")
