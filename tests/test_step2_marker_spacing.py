"""Step 2 init pose: markers too close together are widened by the operator before any motion.

The Step 2 plan moves both arms at once; markers closer than step2.min_marker_x_gap_m along the
camera x axis (image horizontal, the arms' side-by-side direction) risk a collision. At the init
pose, once both markers are visible, the gap is checked; when it is too small the operator widens
the arms in a dialog that shows the gap live (measured by the core), and confirms. After that
Step 2 runs as usual with no further spacing checks. Offline: no robot, no camera.
"""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.calibration import CalibrationCore
from core.calibration.data import (DEFAULT_MIN_MARKER_X_GAP_M, MarkersTooCloseError,
                                   load_min_marker_x_gap_m, marker_x_gap_m)
from core.calibration.sequences import collection
from core.storage import ConfigStorage

ROOT = Path(__file__).resolve().parents[1]
MIN_GAP = 0.11


def marker(x, y=0.0, z=0.35):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T.ravel()


class FakeObserver:
    """Right marker at +gap/2, left at -gap/2 on the camera x axis; `gap` can change (arms widened)."""
    sim = True

    def __init__(self, gap_m, visible=("right", "left")):
        self.gap_m, self.visible, self.last_frame = gap_m, set(visible), {}

    def get_marker_transform(self, sampling_time=0, side="left", **kwargs):
        if side not in self.visible:
            return None
        return [marker(self.gap_m / 2, y=0.004) if side == "right" else marker(-self.gap_m / 2, y=-0.003)]


class FakeOperator:
    """Stands in for the UI dialog: after `widen_after` gap updates it widens the arms to
    `widened_gap_m`, then confirms (or cancels) once it sees a gap at or above the minimum."""

    def __init__(self, observer, widened_gap_m=0.13, widen_after=1, accept=True):
        self.observer, self.widened_gap_m, self.widen_after, self.accept = observer, widened_gap_m, widen_after, accept
        self.prompts, self.updates, self.session = [], [], None

    def prompt(self, gap_m, min_gap_m):
        self.prompts.append((gap_m, min_gap_m))
        self.session = {"done": threading.Event(), "accepted": False, "gap_m": None, "closed": False}
        return self.session

    def on_event(self, kind, value):
        if kind != "marker_gap" or self.session is None or self.session["done"].is_set():
            return
        self.updates.append(value)
        if len(self.updates) >= self.widen_after:
            self.observer.gap_m = self.widened_gap_m
        if value["gap_m"] is not None and value["gap_m"] >= value["min_gap_m"]:
            self.session["gap_m"] = value["gap_m"]
            self.session["accepted"] = self.accept
            self.session["done"].set()


class TestMarkerGap(unittest.TestCase):
    def test_gap_is_the_camera_x_distance_only(self):
        self.assertAlmostEqual(marker_x_gap_m(marker(0.06, y=0.02), marker(-0.055, y=-0.03, z=0.4)), 0.115)
        self.assertAlmostEqual(marker_x_gap_m(marker(-0.055), marker(0.06)), 0.115)

    def test_non_finite_positions_are_rejected(self):
        with self.assertRaises(ValueError):
            marker_x_gap_m(marker(np.nan), marker(0.0))

    def test_error_names_the_measured_gap(self):
        error = MarkersTooCloseError(0.1034, MIN_GAP)
        self.assertEqual((error.gap_m, error.min_gap_m), (0.1034, MIN_GAP))
        self.assertIn("10.3 cm", str(error))


class TestMinGapConfig(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "setting.yaml"

    def test_value_comes_from_setting_yaml(self):
        ConfigStorage.save(self.path, {"step2": {"min_marker_x_gap_m": 0.13}})
        self.assertEqual(load_min_marker_x_gap_m(self.path), (0.13, True))

    def test_missing_key_uses_the_default_and_says_so(self):
        ConfigStorage.save(self.path, {"camera": {}})
        self.assertEqual(load_min_marker_x_gap_m(self.path), (DEFAULT_MIN_MARKER_X_GAP_M, False))

    def test_non_positive_value_is_an_error(self):
        ConfigStorage.save(self.path, {"step2": {"min_marker_x_gap_m": 0.0}})
        with self.assertRaises(ValueError):
            load_min_marker_x_gap_m(self.path)

    def test_shipped_setting_and_default_are_11_cm(self):
        shipped = ConfigStorage.load(ROOT / "config" / "setting.yaml")["step2"]["min_marker_x_gap_m"]
        self.assertEqual(shipped, MIN_GAP)
        self.assertEqual(DEFAULT_MIN_MARKER_X_GAP_M, MIN_GAP)


class TestInitPoseSpacing(unittest.TestCase):
    """`step2_ready` (GUI Init Pose): visibility check -> spacing check -> widen dialog if needed."""

    def setUp(self):
        self.core = CalibrationCore()
        self.core.robot = MagicMock()
        self.verify = MagicMock(return_value=True)
        for p in (patch("core.robot.motion.move_to_auto_ready_pose"),
                  patch("core.robot.motion.verify_and_align_head_at_ready_pose", self.verify),
                  patch.object(collection, "load_min_marker_x_gap_m", return_value=(MIN_GAP, True)),
                  patch.object(collection, "SPACING_MONITOR_PERIOD_S", 0.001)):
            p.start()
            self.addCleanup(p.stop)

    def use(self, observer, operator=None):
        self.core.observer = observer
        if operator is not None:
            self.core.prompt_marker_spacing = operator.prompt
            self.core.on_event = operator.on_event

    def test_wide_enough_needs_no_dialog(self):
        operator = FakeOperator(FakeObserver(0.111))
        self.use(operator.observer, operator)
        result = self.core.run("step2_ready")
        self.assertTrue(result.success, result.error)
        self.assertEqual(operator.prompts, [])

    def test_too_close_opens_the_monitor_and_continues_once_widened(self):
        operator = FakeOperator(FakeObserver(0.09), widened_gap_m=0.125, widen_after=2)
        self.use(operator.observer, operator)
        result = self.core.run("step2_ready")
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(operator.prompts), 1)
        self.assertAlmostEqual(operator.prompts[0][0], 0.09)
        # Live updates: red (too close) first, then green once widened.
        self.assertAlmostEqual(operator.updates[0]["gap_m"], 0.09)
        self.assertAlmostEqual(operator.updates[-1]["gap_m"], 0.125)
        self.assertTrue(operator.session["closed"])
        self.assertTrue(result.partial["marker_x_gap"]["widened"])
        self.assertAlmostEqual(result.partial["marker_x_gap"]["gap_m"], 0.125)
        # The spacing is checked after the visibility check (teaching dialog) and only then.
        self.verify.assert_called_once()

    def test_hidden_marker_while_widening_shows_as_no_gap(self):
        observer = FakeObserver(0.09)
        operator = FakeOperator(observer, widened_gap_m=0.13, widen_after=3)
        self.use(observer, operator)
        original = operator.on_event

        def on_event(kind, value):
            if kind == "marker_gap" and len(operator.updates) == 0:
                observer.visible = {"right"}          # an arm passes in front of a marker
            elif kind == "marker_gap" and len(operator.updates) == 1:
                observer.visible = {"right", "left"}
            original(kind, value)
        self.core.on_event = on_event
        result = self.core.run("step2_ready")
        self.assertTrue(result.success, result.error)
        self.assertIn(None, [u["gap_m"] for u in operator.updates])

    def test_cancel_fails_the_init_pose(self):
        operator = FakeOperator(FakeObserver(0.09), accept=False)
        self.use(operator.observer, operator)
        result = self.core.run("step2_ready")
        self.assertEqual(result.status, "failed")
        self.assertIn("cancelled by user", result.error)

    def test_stop_closes_the_monitor(self):
        observer = FakeObserver(0.09)
        operator = FakeOperator(observer)
        self.use(observer, operator)

        def on_event(kind, value):
            if kind == "marker_gap":
                self.core.stop_event.set()          # operator presses Stop instead of widening
        self.core.on_event = on_event
        result = self.core.run("step2_ready")
        self.assertEqual(result.status, "cancelled")
        self.assertTrue(operator.session["closed"])
        self.assertFalse(operator.session["done"].is_set())

    def test_without_a_dialog_too_close_is_an_error(self):
        self.use(FakeObserver(0.09))
        result = self.core.run("step2_ready")
        self.assertEqual(result.status, "failed")
        self.assertIn("Markers too close", result.error)

    def test_marker_missing_at_the_check_is_an_error(self):
        self.use(FakeObserver(0.2, visible=("right",)), FakeOperator(FakeObserver(0.2)))
        result = self.core.run("step2_ready")
        self.assertEqual(result.status, "failed")
        self.assertIn("Both markers must be in view", result.error)


class TestCollectionSpacing(unittest.TestCase):
    def setUp(self):
        self.core = CalibrationCore()
        self.core.robot = MagicMock()
        move = patch.object(collection, "execute_auto_motion_step")
        patches = [
            patch.object(collection, "load_min_marker_x_gap_m", return_value=(MIN_GAP, True)),
            patch.object(collection, "SPACING_MONITOR_PERIOD_S", 0.001),
            patch.object(collection, "get_both_arm_config", return_value={"arm_idx": list(range(14))}),
            patch.object(collection, "get_head_config", return_value={"head_idx": [14, 15]}),
            patch.object(collection, "capture_one_sample",
                         return_value=(np.zeros(14), np.zeros(2), np.stack([np.eye(4)] * 2))),
            patch.object(collection, "build_incremental_motion_plan", return_value=[{}]),
            patch.object(collection, "move_to_auto_ready_pose"),
            patch.object(collection, "verify_and_align_head_at_ready_pose", return_value=True),
            move,
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.move = collection.execute_auto_motion_step

    def test_auto_motion_after_the_init_pose_does_not_check_again(self):
        # GUI "2) Auto Motion" (prepare=False) runs as before, whatever the spacing now is.
        operator = FakeOperator(FakeObserver(0.05))
        self.core.observer, self.core.prompt_marker_spacing = operator.observer, operator.prompt
        result = self.core.run("collect", plan=[{}])
        self.assertTrue(result.success, result.error)
        self.assertEqual(operator.prompts, [])
        self.move.assert_called_once()

    def test_full_auto_widens_before_the_first_motion(self):
        # Full Auto / Step 2 run the init pose inside the collection (prepare=True).
        operator = FakeOperator(FakeObserver(0.09))
        self.core.observer, self.core.prompt_marker_spacing = operator.observer, operator.prompt
        order = []
        self.move.side_effect = lambda *a, **k: order.append(("move", operator.session["done"].is_set()))
        self.core.on_event = operator.on_event
        result = self.core.run("collect", prepare=True)
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(operator.prompts), 1)
        self.assertEqual(order, [("move", True)])

    def test_full_auto_cancelled_widening_moves_nothing(self):
        operator = FakeOperator(FakeObserver(0.09), accept=False)
        self.core.observer, self.core.prompt_marker_spacing = operator.observer, operator.prompt
        self.core.on_event = operator.on_event
        result = self.core.run("collect", prepare=True)
        self.assertEqual(result.status, "failed")
        self.move.assert_not_called()


class TestMarkerSpacingMessages(unittest.TestCase):
    def test_both_languages_have_every_text(self):
        texts = ConfigStorage.load(ROOT / "config" / "ui_config" / "i18n.yaml")["dialogs"]["markers_too_close"]
        keys = ("title", "notice_msg", "monitor_title", "monitor_header", "monitor_desc",
                "gap_value", "gap_hidden", "btn_done", "btn_cancel")
        for key in keys:
            for lang in ("en", "ko"):
                text = texts[key][lang].format(gap="10.3", min_gap="11.0")
                self.assertTrue(text.strip(), (key, lang))
        for lang in ("en", "ko"):
            self.assertIn("10.3", texts["notice_msg"][lang].format(gap="10.3", min_gap="11.0"))
            self.assertIn("11.0", texts["gap_value"][lang].format(gap="10.3", min_gap="11.0"))


if __name__ == "__main__":
    unittest.main()
