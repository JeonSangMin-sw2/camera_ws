"""What a run leaves under result/: GUI log, exposure, every sweep's points (all redirected to a temp folder)."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.storage import CONFIG_PATHS, SessionLog


class DummyRobotModel:
    left_arm_idx = list(range(7))
    right_arm_idx = list(range(7, 14))
    head_idx = [14, 15]


class DummyRobot:
    def model(self):
        return DummyRobotModel()

    def get_state(self):
        return SimpleNamespace(position=[0.0] * 16)

    def get_dynamics(self):
        return None


def real_camera_source(auto=False, setpoint=5000.0, last_frame_us=4980.0):
    return SimpleNamespace(sim=False, camera=object(), get_camera_exposure=lambda: (auto, setpoint),
                           get_camera_exposure_unit_us=lambda: 1.0, get_actual_exposure=lambda: last_frame_us)


class RecordsTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        patcher = patch.dict(CONFIG_PATHS, txt_dir=str(self.folder), result_dir=str(self.folder),
                             plot_dir=str(self.folder), log_dir=str(self.folder / "result_log"))
        patcher.start()
        self.addCleanup(patcher.stop)


class TestSessionLog(RecordsTestCase):
    def test_lines_are_appended_with_the_time_to_a_file_per_day(self):
        SessionLog.append("[INFO] first", now=datetime(2026, 10, 6, 21, 24, 54))
        SessionLog.append("두 번째 줄", now=datetime(2026, 10, 6, 21, 30, 1))
        SessionLog.append("next day", now=datetime(2026, 10, 7, 9, 0, 0))
        log_dir = self.folder / "result_log"
        self.assertEqual(sorted(p.name for p in log_dir.iterdir()), ["gui_log_20261006.txt", "gui_log_20261007.txt"])
        self.assertEqual((log_dir / "gui_log_20261006.txt").read_text(encoding="utf-8"),
                         "[21:24:54] [INFO] first\n[21:30:01] 두 번째 줄\n")


class TestGuiLogIsSaved(RecordsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from main_ui import UnifiedCalibrationApp
        self.window = UnifiedCalibrationApp(ui_only=True)
        self.addCleanup(self.window.close)

    def saved_text(self):
        files = list((self.folder / "result_log").glob("gui_log_*.txt"))
        return files[0].read_text(encoding="utf-8") if files else ""

    def test_ui_only_mode_keeps_no_run_record(self):
        self.assertFalse(self.window.save_gui_log)
        self.window.log_msg("[INFO] not a run")
        self.assertEqual(self.saved_text(), "")

    def test_every_gui_log_line_goes_to_the_result_folder(self):
        self.window.save_gui_log = True
        self.window.log_msg("[Camera] Wizard applied exposure (auto=False, exposure=4000 = 4.0 ms)")
        self.window.log_msg("[full] failed: right_elbow: joint calibration did not converge")
        lines = self.saved_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"^\[\d\d:\d\d:\d\d\] \[Camera\] Wizard applied exposure \(auto=False, exposure=4000 = 4\.0 ms\)$")
        self.assertTrue(lines[1].endswith("right_elbow: joint calibration did not converge"))

    def test_a_write_failure_is_shown_once_and_does_not_stop_the_gui_log(self):
        self.window.save_gui_log = True
        self.window.log_text.clear()
        with patch.object(SessionLog, "append", side_effect=OSError("disk full")) as append:
            self.window.log_msg("line 1")
            self.window.log_msg("line 2")
        self.assertEqual(append.call_count, 1)
        shown = self.window.log_text.toPlainText()
        self.assertIn("line 1", shown)
        self.assertIn("line 2", shown)
        self.assertEqual(shown.count("no longer being saved"), 1)
        self.assertIn("disk full", shown)


class TestExposureRecord(RecordsTestCase):
    def test_exposure_of_a_real_camera_is_reported_as_set_and_as_measured(self):
        calibrator = JointCalibrator(marker_st=real_camera_source(), robot=DummyRobot())
        self.assertEqual(calibrator.camera_exposure_record(),
                         {"auto": False, "setpoint": 5000.0, "setpoint_unit_us": 1.0, "last_frame_us": 4980.0})

    def test_nothing_is_made_up_without_a_real_camera(self):
        for source in (None, SimpleNamespace(sim=True, camera=object()), SimpleNamespace(sim=False, camera=None)):
            self.assertIsNone(JointCalibrator(marker_st=source, robot=DummyRobot()).camera_exposure_record())


class TestEverySweepIsSaved(RecordsTestCase):
    def setUp(self):
        super().setUp()
        self.calibrator = JointCalibrator(marker_st=real_camera_source(setpoint=4000.0, last_frame_us=3990.0), robot=DummyRobot())
        self.calibrator.joint_offsets = {"right": {"elbow": 0.0, "wrist_pitch": 0.0, "wrist_yaw2": 0.0, "wrist_roll": 0.0}}

    def test_each_sweep_of_a_joint_calibration_asks_for_its_points_and_the_log_names_the_exposure(self):
        calls = []

        def sweep(arm_side, mode, **kwargs):
            calls.append(kwargs)
            return {"angle_between_normals": 0.02, "center_dist": 0.3, "r_A": 75.0, "r_B": 75.1,
                    "optimal_offset": 0.01, "sign": 1.0}

        self.calibrator.perform_calibration_sweep_continuous = sweep
        logs = []
        result = self.calibrator.perform_joint_calibration("right", "wrist_pitch", log_callback=logs.append,
                                                           sweep_duration=1.0, save_debug=True, pass_idx=2)
        self.assertTrue(result["converged"])
        self.assertGreaterEqual(len(calls), 2)
        # The plot is still made for the first sweep only; the points are asked for every time.
        self.assertEqual([c["save_debug"] for c in calls], [True] + [False] * (len(calls) - 1))
        for number, call in enumerate(calls, 1):
            self.assertEqual(call["debug_points_note"],
                             f"pass 2, sweep {number}, staged offset {call['current_offset_deg']:.4f} deg")
        self.assertTrue(any("Camera exposure for these sweeps: manual, setpoint 4000" in line
                            and "last frame 3990 us" in line for line in logs), logs)
        saved = (self.folder / "joint_calib_debug_right_wrist_pitch.txt").read_text(encoding="utf-8")
        self.assertIn("Camera exposure for these sweeps: manual, setpoint 4000", saved)

    def test_no_points_are_asked_for_when_debug_saving_is_off(self):
        calls = []

        def sweep(arm_side, mode, **kwargs):
            calls.append(kwargs)
            return {"angle_between_normals": 0.02, "center_dist": 0.3, "r_A": 75.0, "r_B": 75.1,
                    "optimal_offset": 0.01, "sign": 1.0}

        self.calibrator.perform_calibration_sweep_continuous = sweep
        self.calibrator.perform_joint_calibration("right", "wrist_pitch", log_callback=lambda line: None, sweep_duration=1.0)
        self.assertTrue(all(c["debug_points_note"] is None and not c["save_debug"] for c in calls))

    def test_saved_points_carry_the_note_and_stay_readable_block_by_block(self):
        q_full = np.zeros(16)
        dataset = [(q_full, np.eye(4)), (q_full, np.eye(4))]
        with patch.object(BaseCalibrator, "compute_fk", return_value=np.eye(4)), \
                patch.object(self.calibrator, "is_head_active", return_value=False):
            for number in (1, 2):
                self.calibrator.save_debug_points("right", 2, dataset, [0.0] * 7, "ee_right", None, np.eye(4), "joint_A",
                                                  note=f"pass 1, sweep {number}, staged offset -2.1{number}00 deg")
        text = (self.folder / "sweep_points_right_joint_A_axis_2.txt").read_text(encoding="utf-8")
        self.assertEqual(text.count("=== NEW ITERATION ==="), 1)
        self.assertIn("# pass 1, sweep 1, staged offset -2.1100 deg", text)
        self.assertIn("# pass 1, sweep 2, staged offset -2.1200 deg", text)
        data_lines = [line for line in text.splitlines() if line and not line.startswith("#") and "NEW ITERATION" not in line]
        self.assertEqual(len(data_lines), 4)
        self.assertTrue(all(len(line.split(",")) == 58 for line in data_lines))

    def test_two_modes_sweeping_the_same_axis_write_separate_files(self):
        dataset = [(np.zeros(16), np.eye(4))]
        with patch.object(BaseCalibrator, "compute_fk", return_value=np.eye(4)), \
                patch.object(self.calibrator, "is_head_active", return_value=False):
            for mode in ("wrist_roll_v13", "wrist_pitch_v13"):
                self.calibrator.save_debug_points("right", 6, dataset, [0.0] * 7, "ee_right", None, np.eye(4), "joint_A",
                                                  file_tag=JointCalibrator.sweep_points_tag(mode, "joint_A"))
        for mode in ("wrist_roll_v13", "wrist_pitch_v13"):
            text = (self.folder / f"sweep_points_right_{mode}_joint_A_axis_6.txt").read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Joint_A_Angle(deg)"), text[:40])
            self.assertNotIn("NEW ITERATION", text)

    def test_a_new_joint_calibration_clears_only_its_own_sweep_files(self):
        other = self.folder / "sweep_points_right_wrist_roll_v13_joint_A_axis_6.txt"
        own = [self.folder / "sweep_points_right_wrist_pitch_v13_joint_A_axis_6.txt",
               self.folder / "sweep_points_right_wrist_pitch_v13_joint_B_axis_4.txt"]
        for path in [other] + own:
            path.write_text("earlier sweep\n", encoding="utf-8")
        self.calibrator.perform_calibration_sweep_continuous = lambda arm_side, mode, **kwargs: {
            "angle_between_normals": 90.0, "center_dist": 0.3, "r_A": 127.0, "r_B": 68.0, "optimal_offset": 0.0, "sign": 1.0}
        self.calibrator.perform_joint_calibration("right", "wrist_pitch_v13", log_callback=lambda line: None,
                                                  sweep_duration=1.0, save_debug=True, pass_idx=1)
        self.assertEqual(other.read_text(encoding="utf-8"), "earlier sweep\n")
        self.assertFalse(any(path.exists() for path in own))


if __name__ == "__main__":
    unittest.main()
