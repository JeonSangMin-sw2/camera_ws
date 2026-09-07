import os
import sys

# Set offscreen Qt platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

from PySide6.QtWidgets import QApplication
from main_ui import UnifiedCalibrationApp

def test_ui_tabs():
    app = QApplication.instance() or QApplication(sys.argv)
    
    print("[TEST] Initializing UnifiedCalibrationApp with ui_only=True...")
    win = UnifiedCalibrationApp(marker_st=None, robot=None, ui_only=True)
    
    # Check tabs
    tab_count = win.left_tabs.count()
    tab_titles = [win.left_tabs.tabText(i) for i in range(tab_count)]
    print(f"[TEST] Tab count: {tab_count}, Titles: {tab_titles}")
    
    assert "Step 1" in tab_titles, "Step 1 tab missing!"
    assert "Step 1.5" in tab_titles, "Step 1.5 tab missing!"
    assert "Step 2" in tab_titles, "Step 2 tab missing!"
    
    step1_idx = tab_titles.index("Step 1")
    step1_5_idx = tab_titles.index("Step 1.5")
    step2_idx = tab_titles.index("Step 2")
    
    print(f"[TEST] Step 1 idx: {step1_idx}, Step 1.5 idx: {step1_5_idx}, Step 2 idx: {step2_idx}")
    assert step1_idx < step1_5_idx < step2_idx, "Tabs are not in correct sequence [Step 1, Step 1.5, Step 2]!"
    
    # Test switching between tabs to verify _reparent_shared_widgets
    print("[TEST] Switching to Step 1...")
    win.left_tabs.setCurrentIndex(step1_idx)
    
    print("[TEST] Switching to Step 1.5...")
    win.left_tabs.setCurrentIndex(step1_5_idx)
    
    print("[TEST] Switching to Step 2...")
    win.left_tabs.setCurrentIndex(step2_idx)
    
    print("[TEST] Switching back to Step 1.5...")
    win.left_tabs.setCurrentIndex(step1_5_idx)
    
    print("[TEST] Checking Step 1.5 widgets...")
    assert hasattr(win, "tbl_step1_5_head_monitor"), "tbl_step1_5_head_monitor missing!"
    assert hasattr(win, "tbl_step1_5_cam_monitor"), "tbl_step1_5_cam_monitor missing!"
    assert hasattr(win, "btn_step1_5_ready"), "btn_step1_5_ready missing!"
    assert hasattr(win, "btn_step1_5_start"), "btn_step1_5_start missing!"
    assert hasattr(win, "btn_step1_5_apply"), "btn_step1_5_apply missing!"
    assert hasattr(win, "head_camera_calibrator"), "head_camera_calibrator missing!"

    # Test mock head sweep
    print("[TEST] Testing mock head sweep in Step 1.5...")
    res = win.head_camera_calibrator.perform_head_sweep(
        arm_side="right",
        pan_range_deg=15.0,
        tilt_range_deg=10.0,
        num_steps=11,
        log_callback=print
    )
    assert res and res["success"], "Mock head sweep failed!"
    win._update_step1_5_tables(res)
    print("Table updated successfully.")
    
    # Check table values
    head_pan_item = win.tbl_step1_5_head_monitor.item(0, 1)
    assert head_pan_item is not None, "Head pan table item is empty!"
    print(f"Table Head Pan offset cell: {head_pan_item.text()}")

    cam_roll_item = win.tbl_step1_5_cam_monitor.item(0, 1)
    assert cam_roll_item is not None, "Camera roll table item is empty!"
    print(f"Table Camera Roll calib cell: {cam_roll_item.text()}")

    print("\n>>> ALL UI TESTS PASSED SUCCESSFULLY! <<<")
    return True

if __name__ == "__main__":
    success = test_ui_tabs()
    if not success:
        sys.exit(1)
