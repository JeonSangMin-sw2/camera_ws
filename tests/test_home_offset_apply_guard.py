"""A Step 2 result is applied as home offset at most once, and only when it was solved in this run
from samples collected at the current robot zero. Home Offset Reset/Apply move the zero
(CalibrationCore.home_epoch); a result from an npz dataset, older samples or a previous run of the
program is analysis only. Rollback to the reset baseline stays available. Offline, no robot.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from core.calibration import CalibrationCore
from core.calibration.sequences import collection


class GuardFixture(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.result = Path(folder.name, "result_20260929_120000.json")
        self.result.write_text(json.dumps({"joint_offset_deg": [0.1] * 14}))
        self.baseline = Path(folder.name, "home_reset_baseline.json")
        # Same layout as save_home_reset_baseline_json: the type sits under "metadata" (2026-09-29: a
        # top-level "type" here hid that the real baseline was refused as a Step 2 result).
        self.baseline.write_text(json.dumps({"joint_offset_deg": [0.0] * 14,
                                             "metadata": {"type": "home_reset_baseline"}}))
        self.core = CalibrationCore()
        self.core.robot = MagicMock()
        self.controller = MagicMock()
        self.controller.move_home_offset_candidate_path.return_value = {"status": "success", "arm": "both"}
        self.controller.move_to_check_position_candidate_path.return_value = {"status": "success", "arm": "both"}
        self.controller.apply_current_pose_home_offset.return_value = {"success": True, "needs_reconnect": True}
        for p in (patch("core.robot.home_offset.HomeOffsetController", return_value=self.controller),
                  patch("core.robot.home_offset.reset_current_pose_home_offsets", return_value={"success": True}),
                  patch("core.robot.home_offset.save_home_reset_baseline_json",
                        return_value=(str(self.baseline), {}))):
            p.start()
            self.addCleanup(p.stop)

    def solve(self, data_home_epoch):
        """Step 2 Calculate: optimize_step2 writes self.result and records it as last_result_path."""
        def fake_optimize(core, **kwargs):
            core.last_result_path = str(self.result)
            return {"offset": [0.1]}
        with patch("core.calibration.sequences.step2.optimize_step2", side_effect=fake_optimize):
            result = self.core.run("optimize", kwargs={}, data_home_epoch=data_home_epoch)
        self.assertTrue(result.success, result.error)

    def home(self, task, **options):
        return self.core.run("home", task_type=task, **options)


class TestCoreApplyGuard(GuardFixture):
    def test_nothing_is_applicable_before_a_calculation(self):
        self.assertIsNone(self.core.applicable_result_path)
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))
        result = self.home("apply", arm="both", include_head=True, json_path=str(self.result))
        self.assertEqual(result.status, "failed")
        self.assertIn("cannot be applied", result.error)
        self.controller.apply_current_pose_home_offset.assert_not_called()

    def test_result_from_current_zero_is_applied_once(self):
        self.solve(data_home_epoch=0)
        self.assertTrue(self.core.result_apply_allowed(str(self.result)))
        self.assertTrue(self.home("move_zero", json_path=str(self.result), label="opt", arm="both",
                                  include_head=True).success)
        first = self.home("apply", arm="both", include_head=True, json_path=str(self.result))
        self.assertTrue(first.success, first.error)
        self.assertEqual(self.core.home_epoch, 1)
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))
        # Second time: the preview motion still runs (comparison, with a warning), the application does not.
        again = self.home("move_zero", json_path=str(self.result), label="opt", arm="both", include_head=True)
        self.assertTrue(again.success, again.error)
        second = self.home("apply", arm="both", include_head=True, json_path=str(self.result))
        self.assertEqual(second.status, "failed")
        self.assertEqual(self.controller.move_home_offset_candidate_path.call_count, 2)
        self.assertEqual(self.controller.apply_current_pose_home_offset.call_count, 1)

    def test_baseline_written_by_the_reset_is_recognised(self):
        # 2026-09-29: the real baseline (type under "metadata") was refused for the rollback preview.
        from types import SimpleNamespace
        from core.robot.home_offset import build_home_reset_baseline_data
        robot = SimpleNamespace(get_state=lambda: SimpleNamespace(position=[0.0] * 24))
        model = SimpleNamespace(right_arm_idx=list(range(2, 9)), left_arm_idx=list(range(9, 16)), head_idx=[0, 1])
        self.baseline.write_text(json.dumps(build_home_reset_baseline_data(robot, model, model_name="a")))
        self.assertTrue(self.core._is_home_reset_baseline(str(self.baseline)))
        self.assertFalse(self.core._is_home_reset_baseline(str(self.result)))
        for task in ("move_zero", "move_check"):
            options = dict(json_path=str(self.baseline), label="baseline", arm="both", include_head=True)
            self.assertTrue(self.home(task, **options).success)

    def test_locked_result_can_still_be_previewed(self):
        # 2026-09-29: the user could not compare the optimized and baseline check poses; only the
        # application may be locked.
        logs = []
        self.core.log_msg = logs.append
        for task in ("move_zero", "move_check"):
            options = dict(json_path=str(self.result), label="opt", arm="both", include_head=True)
            self.assertTrue(self.home(task, **options).success)
        self.controller.move_home_offset_candidate_path.assert_called_once()
        self.controller.move_to_check_position_candidate_path.assert_called_once()
        self.assertEqual(sum("Preview only" in line for line in logs), 2)
        self.controller.apply_current_pose_home_offset.assert_not_called()

    def test_npz_or_stale_samples_are_analysis_only(self):
        self.solve(data_home_epoch=None)          # npz dataset
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))
        self.core.home_epoch = 2
        self.solve(data_home_epoch=1)             # samples from before the last zero move
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))
        self.solve(data_home_epoch=2)
        self.assertTrue(self.core.result_apply_allowed(str(self.result)))

    def test_rollback_to_the_baseline_is_always_possible(self):
        for task in ("move_zero", "move_check"):
            options = dict(json_path=str(self.baseline), label="baseline", arm="both", include_head=True)
            self.assertTrue(self.home(task, **options).success)
        rollback = self.home("apply", arm="both", include_head=True, json_path=None)
        self.assertTrue(rollback.success, rollback.error)
        self.controller.apply_current_pose_home_offset.assert_called_once()

    def test_any_zero_move_locks_the_result(self):
        self.solve(data_home_epoch=0)
        self.assertTrue(self.home("reset", model_name="a", include_head=True).success)
        self.assertEqual(self.core.home_epoch, 1)
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))
        self.solve(data_home_epoch=1)
        self.assertTrue(self.home("apply", arm="both", include_head=True, json_path=None).success)  # rollback
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))

    def test_failed_apply_still_locks(self):
        self.solve(data_home_epoch=0)
        self.controller.apply_current_pose_home_offset.side_effect = RuntimeError("reset failed part way")
        self.assertEqual(self.home("apply", arm="both", include_head=True, json_path=str(self.result)).status, "failed")
        self.assertFalse(self.core.result_apply_allowed(str(self.result)))

    def test_step2_samples_carry_their_zero(self):
        def fake_optimize(core, **kwargs):
            core.last_result_path = str(self.result)
            return {"offset": [0.1]}
        samples = [{"q_arm": np.zeros(14), "q_head": None, "marker": np.eye(4), "home_epoch": 0}]
        self.core.include_head_motion = False
        with patch("core.calibration.sequences.step2.optimize_step2", side_effect=fake_optimize):
            self.assertTrue(self.core.run("step2", samples=samples, verify=False).success)
            self.assertTrue(self.core.result_apply_allowed(str(self.result)))
            self.core.home_epoch = 1
            self.assertTrue(self.core.run("step2", samples=samples, verify=False).success)
            self.assertFalse(self.core.result_apply_allowed(str(self.result)))
            untagged = [dict(samples[0], home_epoch=None)]
            self.assertTrue(self.core.run("step2", samples=untagged, verify=False).success)
            self.assertFalse(self.core.result_apply_allowed(str(self.result)))

    def test_collected_samples_are_tagged_with_the_current_zero(self):
        self.core.home_epoch = 3
        self.core.observer = SimpleNamespace(sim=True, last_frame={})
        sample = (np.zeros(14), np.zeros(2), np.stack([np.eye(4)] * 2))
        with patch.object(collection, "get_both_arm_config", return_value={"arm_idx": list(range(14))}), \
                patch.object(collection, "get_head_config", return_value={"head_idx": [14, 15]}), \
                patch.object(collection, "execute_auto_motion_step"), \
                patch.object(collection, "capture_one_sample", return_value=sample):
            result = self.core.run("collect", plan=[{}])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.completed["samples"][0]["home_epoch"], 3)


class TestApplyButtons(GuardFixture):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from main_ui import UnifiedCalibrationApp
        cls.window = UnifiedCalibrationApp(ui_only=True)
        cls.wiz = cls.window.wizard_widget

    def setUp(self):
        super().setUp()
        self.core = self.window.core
        self.core.applicable_result_path = None
        self.addCleanup(setattr, self.core, "applicable_result_path", None)

    def dialog(self):
        from main_ui import ApplyHomeOffsetDialog
        return ApplyHomeOffsetDialog(self.window, str(self.result), str(self.baseline), "both",
                                     include_head=True, compare_summary="")

    def test_dialog_previews_a_locked_result_but_offers_rollback_only(self):
        dlg = self.dialog()
        self.assertTrue(dlg.btn_opt.isEnabled())              # preview for comparison
        self.assertTrue(dlg.lbl_opt_locked.isVisibleTo(dlg))
        self.assertTrue(dlg.btn_apply.isEnabled())            # rollback to the baseline
        self.core.applicable_result_path = str(self.result)
        dlg = self.dialog()
        self.assertTrue(dlg.btn_opt.isEnabled())
        self.assertFalse(dlg.lbl_opt_locked.isVisibleTo(dlg))

    def test_dialog_refuses_only_the_locked_application(self):
        dlg = self.dialog()
        dlg.btn_opt.setChecked(True)
        with patch("main_ui.show_warning_dialog") as warn, patch("main_ui.Step2ApplyHomeOffsetWorker") as worker:
            dlg.on_apply("optimized")
            worker.assert_not_called()
            dlg._execute_preview_motion("move_zero", "Zero", "zero", "Preview Error")
        warn.assert_called_once()
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[1], "move_zero")

    def test_wizard_locks_only_the_optimized_application(self):
        with patch.object(self.wiz, "get_apply_paths", return_value=(str(self.result), str(self.baseline))):
            self.wiz.set_wizard_buttons_enabled(True)
            self.assertFalse(self.wiz.btn_apply_new_offset.isEnabled())
            for btn in (self.wiz.btn_new_offset_zero, self.wiz.btn_new_offset_preview,
                        self.wiz.btn_rollback_zero, self.wiz.btn_rollback_preview, self.wiz.btn_rollback_joint):
                self.assertTrue(btn.isEnabled())
            self.assertFalse(self.wiz.lbl_wiz_opt_locked.isHidden())
            self.core.applicable_result_path = str(self.result)
            self.wiz.set_wizard_buttons_enabled(True)
            self.assertTrue(self.wiz.btn_apply_new_offset.isEnabled())
            self.assertTrue(self.wiz.lbl_wiz_opt_locked.isHidden())
            self.core.applicable_result_path = None
            with patch("ui.wizard_widget.QMessageBox") as box, \
                    patch("ui.core_bridge.Step2ApplyHomeOffsetWorker") as worker:
                self.wiz.wizard_apply_offset("optimized")
            box.warning.assert_called_once()
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
