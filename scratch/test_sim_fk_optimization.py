import sys
import os
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform

app = QApplication(sys.argv)

class MockRobot:
    pass

mw = UnifiedCalibrationApp(marker_st=SimulatedMarkerTransform(MockRobot(), {}, "1.2"), robot=None, ui_only=True)
mw.step2_mode_sel.setCurrentText("sim")

# Intercept log_msg to print optimization results
orig_log = mw.log_msg
def log_print(msg):
    orig_log(msg)
    if "Result saved" in msg or "RESULT" in msg or "Right arm joint offset" in msg or "Left arm joint offset" in msg or "mount_to_cam_new" in msg or "BASE LINE COMPARISON" in msg or "Diff =" in msg:
        print(msg)

mw.log_msg = log_print

# Perform Step 2 Auto Motions directly
mw.move_to_all_auto_motions()

calculated = False

def check_status():
    global calculated
    samples = len(mw.shared_arm_q_list)
    print(f"[PROGRESS] Collected: {samples} / 76")
    if samples >= 76 and not calculated:
        calculated = True
        print(f"\n[AUTO MOTIONS COMPLETE] Collected {samples} samples!")
        mw.step2_calculate()
        QTimer.singleShot(4000, app.quit)

timer = QTimer()
timer.setInterval(800)
timer.timeout.connect(check_status)
timer.start()

QTimer.singleShot(25000, app.quit)

app.exec_()
print("[TEST FINISHED]")
