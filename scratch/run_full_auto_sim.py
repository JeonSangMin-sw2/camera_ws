import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication

app = QApplication(sys.argv)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform
class MockRobot:
    pass

mw = UnifiedCalibrationApp(marker_st=SimulatedMarkerTransform(MockRobot(), {}, "1.2"), robot=MockRobot(), ui_only=True)

def log_print(msg):
    if "FINAL RESULTS" in msg or "Wrist Yaw 2" in msg or "Wrist Pitch" in msg or "optimal_offset_deg" in msg or "Elbow" in msg or "diff_angle" in msg or "Extracted" in msg:
        print(msg)
    elif "42.4" in msg or "2.3" in msg:
        print(msg)
    else:
        if "STARTING PASS" in msg or "[FULL AUTO" in msg:
            print(msg)

mw.log_msg = log_print
mw.robot_version = "1.2"
mw.start_full_auto()

if mw.active_worker:
    mw.active_worker.finished_signal.connect(app.quit)
    app.exec_()
