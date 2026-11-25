"""Capture every wizard slide (main_ui --ui, maximized like the app) for the user manual.

usage: python capture_wizard_ui.py <repo root> <out dir>
Writes <out>/<lang>/<name>.png for ko and en, and <out>/<lang>/rects.json: per capture the window size
and the rectangles (logical px, window coordinates) of the widgets the manual points at. Per language,
because the widgets move with the text length.
"""
import json
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = ["main_ui.py", "--ui"]
sys.path.insert(0, str(ROOT))
import main_ui  # noqa: E402
from PySide6.QtCore import QTimer, QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

orig_exec = main_ui.QApplication.exec
rects = {}


def settle():
    for _ in range(15):
        QApplication.processEvents()


def rect_of(gui, widget):
    top_left = widget.mapTo(gui, QPoint(0, 0))
    return [top_left.x(), top_left.y(), widget.width(), widget.height()]


def shoot(gui, lang, name, widgets=()):
    settle()
    path = OUT / lang / f"{name}.png"
    gui.grab().save(str(path))
    rects.setdefault(name, {"window": [gui.width(), gui.height()]})
    rects[name].update({key: rect_of(gui, w) for key, w in widgets})


def run():
    gui = next(x for x in QApplication.topLevelWidgets() if isinstance(x, main_ui.UnifiedCalibrationApp))
    wz = gui.wizard_widget
    W = type(wz)
    for lang, index in (("ko", 1), ("en", 0)):
        rects.clear()
        (OUT / lang).mkdir(parents=True, exist_ok=True)
        gui.combo_lang.setCurrentIndex(index)
        wz.setVisible(False)
        gui.overview_container.setVisible(True) if getattr(gui, "overview_container", None) else None
        gui.btn_start_wizard.setVisible(True)
        shoot(gui, lang, "00_start", [("start", gui.btn_start_wizard)])
        gui.btn_start_wizard.click()
        plain = [("01_intrinsics_prep", W.SLIDE_INTRINSICS_PREP, [("open_board", wz.btn_open_board)]),
                 ("02_intrinsics_calib", W.SLIDE_INTRINSICS_CALIB, []),
                 ("03_robot_connect", W.SLIDE_ROBOT_CONNECT, [("connect_box", wz.conn_box)]),
                 ("04_gripper_pose", W.SLIDE_GRIPPER_POSE, [("move", wz.btn_gripper_pose)]),
                 ("05_camera_mount", W.SLIDE_CAMERA_MOUNT, []),
                 ("06_gripper_removal", W.SLIDE_GRIPPER_REMOVAL, []),
                 ("07_marker_bracket", W.SLIDE_MARKER_BRACKET, []),
                 ("09_home_offset", W.SLIDE_HOME_OFFSET, [("reset", wz.btn_step3_reset), ("teach", wz.btn_how_to_move)]),
                 ("12_apply", W.SLIDE_APPLY, [])]
        for name, idx, widgets in plain:
            wz.stacked_widget.setCurrentIndex(idx)
            shoot(gui, lang, name, widgets)

        # The external-PC window of the robot connection slide (its own window size).
        from ui.wizard_widget import RobotIpDialog
        dlg = RobotIpDialog(wz.wizard_ip_input.text(), wz)
        dlg.show()
        settle()
        dlg.grab().save(str(OUT / lang / "03b_robot_ip_dialog.png"))
        rects["03b_robot_ip_dialog"] = {"window": [dlg.width(), dlg.height()]}
        dlg.close()

        # Exposure: after 1) found a value, box 2 asks whether to keep it; then the hand controls.
        wz.stacked_widget.setCurrentIndex(W.SLIDE_EXPOSURE)
        wz.chk_exp_manual.setChecked(False)
        wz.set_exposure_motion_status("done", "auto")
        wz.show_auto_exposure_result(True, {"applied": "manual", "exposure": 4000, "camera_model": "D405", "trials": []})
        shoot(gui, lang, "08_exposure", [("auto", wz.btn_exp_auto), ("choice_box", wz.exp_ctrl_box),
                                         ("accept", wz.btn_exp_accept)])
        wz.chk_exp_manual.setChecked(True)
        shoot(gui, lang, "08b_exposure_manual", [("choice_box", wz.exp_ctrl_box), ("confirm", wz.chk_wiz_exp_confirmed),
                                                 ("zero", wz.btn_exp_zero)])
        wz.chk_exp_manual.setChecked(False)

        # Calibration: a run in Step 2 collection, then the finished run.
        wz.stacked_widget.setCurrentIndex(W.SLIDE_CALIBRATION)
        wz.set_status(wz.lbl_step4_status, "wizard.st.calib_progress",
                      "color: #1e88e5; font-weight: bold; font-size: 16px;", time="12:34")
        wz.btn_start_unified.setEnabled(False)
        wz.run_phase, wz.run_detail, wz.run_skipped, wz.run_outcome = 2, (20, 44), set(), None
        wz.refresh_run_progress()
        shoot(gui, lang, "10_calibration_running", [("start", wz.btn_start_unified), ("run_line", wz.run_progress),
                                                    ("feed", wz.btn_feed4), ("stop", wz.btn_stop4)])
        wz.unified_elapsed = 1315
        wz.stop_step5(True)
        shoot(gui, lang, "11_calibration_done", [("run_line", wz.run_progress)])
        wz.run_phase, wz.run_detail, wz.run_outcome = -1, None, None
        wz.refresh_run_progress()
        (OUT / lang / "rects.json").write_text(json.dumps(rects, indent=1), encoding="utf-8")
        print(lang, "captured", sorted(rects))
    QApplication.quit()


def exec_with_check(*_):
    QTimer.singleShot(1500, run)
    return orig_exec()


main_ui.QApplication.exec = exec_with_check
try:
    main_ui.main()
except SystemExit:
    pass
