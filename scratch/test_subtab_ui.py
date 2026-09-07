import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtWidgets import QApplication

from main_ui import UnifiedCalibrationApp

def test_ui_structure():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    print("Initializing UnifiedCalibrationApp(None, None, ui_only=True)...")
    window = UnifiedCalibrationApp(None, None, ui_only=True)

    # 1. Check top tabs
    top_tab_count = window.left_tabs.count()
    top_tab_names = [window.left_tabs.tabText(i) for i in range(top_tab_count)]
    print(f"Top tabs ({top_tab_count}): {top_tab_names}")
    assert top_tab_count == 3, f"Expected 3 top tabs, got {top_tab_count}"
    assert "Overview" in top_tab_names[0] or "Wizard" in top_tab_names[0]
    assert "Step 1" in top_tab_names[1]
    assert "Step 2" in top_tab_names[2]

    # 2. Check workflow subtabs in Step 1
    wf_count = window.workflow_tabs.count()
    wf_names = [window.workflow_tabs.tabText(i) for i in range(wf_count)]
    print(f"Workflow subtabs ({wf_count}): {wf_names}")
    assert wf_count == 4, f"Expected 4 workflow subtabs, got {wf_count}: {wf_names}"
    assert wf_names == ["Auto", "Joint", "Marker", "Head & Cam"]

    # 3. Check Dash Stack dynamic switching
    assert hasattr(window, 'dash_stack')
    # Default is page 0 (Arm & Marker)
    window.workflow_tabs.setCurrentIndex(1)
    app.processEvents()
    assert window.dash_stack.currentIndex() == 0, f"Expected dash_stack 0 for Joint tab, got {window.dash_stack.currentIndex()}"

    # Switch to Head & Cam (index 3)
    window.workflow_tabs.setCurrentIndex(3)
    app.processEvents()
    assert window.dash_stack.currentIndex() == 1, f"Expected dash_stack 1 for Head & Cam tab, got {window.dash_stack.currentIndex()}"

    # Switch back to Auto (index 0)
    window.workflow_tabs.setCurrentIndex(0)
    app.processEvents()
    assert window.dash_stack.currentIndex() == 0, f"Expected dash_stack 0 for Auto tab, got {window.dash_stack.currentIndex()}"

    # 4. Check Table column count (should be 2 columns only, no delta)
    assert window.tbl_step1_5_head_monitor.columnCount() == 2, f"Expected 2 columns in head table, got {window.tbl_step1_5_head_monitor.columnCount()}"
    assert window.tbl_step1_5_cam_monitor.columnCount() == 2, f"Expected 2 columns in cam table, got {window.tbl_step1_5_cam_monitor.columnCount()}"

    # 5. Check table update
    dummy_results = {
        "success": True,
        "head_offsets_deg": {"pan": -0.123, "tilt": 0.456},
        "nominal_mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
        "calibrated_mount_to_cam": [0.0475, 0.0091, 0.0568, -89.8, 0.2, -89.7],
        "quality": {
            "rmse_tilt_plane_mm": 0.045,
            "rmse_pan_plane_mm": 0.038,
            "ortho_error_deg": 0.012
        }
    }
    window._update_step1_5_tables(dummy_results)
    assert window.tbl_step1_5_head_monitor.item(0, 1).text() == "-0.123°"
    assert window.tbl_step1_5_head_monitor.item(1, 1).text() == "+0.456°"
    assert window.tbl_step1_5_cam_monitor.item(0, 1).text() == "-89.800"
    print("Table values populated successfully with 2 columns.")

    # 6. Check Ready Pose Move in mock/ui_only
    print("Testing move_to_ready_pose_step1_5...")
    window.move_to_ready_pose_step1_5()
    # Wait for thread
    if hasattr(window, 'step1_5_ready_worker') and window.step1_5_ready_worker is not None:
        window.step1_5_ready_worker.wait(3000)
    app.processEvents()
    assert window.btn_step1_5_ready.isEnabled() == True
    assert window.btn_step1_5_start.isEnabled() == True
    print("Ready pose worker completed and buttons re-enabled properly!")

    print("\n[SUCCESS] All Subtab UI, Dashboard Switching, Table Columns, and Ready Pose tests PASSED!")

if __name__ == "__main__":
    test_ui_structure()
