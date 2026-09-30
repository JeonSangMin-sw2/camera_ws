"""The connected robot's version decides which marker bracket the detector uses.

`Tf_to_marker_<side>_v12` / `_v13` are the CAD nominals for the two brackets; `Tf_to_marker_<side>`
is the live value `calc_cam_to_tool` turns a marker pose into a tool pose with. Nothing tied the
live value to a robot version, so carrying a setting.yaml between a v1.2 and a v1.3 robot left a
bracket ~99 mm and 90 deg wrong, and the UI's own `abs(x) > 0.05` guess ran off the "1.2" default
whenever the robot had not been connected yet.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import core.storage as storage
from core.storage import ConfigStorage
from core.marker_detection import (BRACKET_VERSION_MARGIN_DEG, BRACKET_VERSION_MARGIN_M,
                                   Marker_Transform, bracket_pose_delta, bracket_version_of,
                                   marker_bracket_nominals, resolve_marker_brackets)

ROOT = Path(__file__).resolve().parents[1]

V12_LEFT = [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
V12_RIGHT = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
V13_LEFT = [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
V13_RIGHT = [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
# What Step 1 produces on top of the v1.2 nominal: sub-millimetre, a fraction of a degree.
V12_LEFT_CALIBRATED = [0.0, 0.05408, -0.04885, 90.0, -0.24, 0.0]
V12_RIGHT_CALIBRATED = [0.0, -0.05408, -0.04885, 90.0, -0.24, 180.0]

NOMINALS = {"1.2": V12_LEFT, "1.3": V13_LEFT}


def markers_config(**overrides):
    config = {
        "Tf_to_marker_left": list(V12_LEFT_CALIBRATED),
        "Tf_to_marker_right": list(V12_RIGHT_CALIBRATED),
        "Tf_to_marker_left_v12": list(V12_LEFT),
        "Tf_to_marker_right_v12": list(V12_RIGHT),
        "Tf_to_marker_left_v13": list(V13_LEFT),
        "Tf_to_marker_right_v13": list(V13_RIGHT),
    }
    config.update(overrides)
    return config


def silent(*_args, **_kwargs):
    pass


class TestBracketClassification(unittest.TestCase):
    def test_the_two_brackets_are_far_enough_apart_to_tell_apart(self):
        for side, v12, v13 in (("left", V12_LEFT, V13_LEFT), ("right", V12_RIGHT, V13_RIGHT)):
            distance, angle = bracket_pose_delta(v12, v13)
            self.assertGreater(distance, 10 * BRACKET_VERSION_MARGIN_M, side)
            self.assertGreater(angle, 10 * BRACKET_VERSION_MARGIN_DEG, side)

    def test_a_calibration_does_not_move_a_bracket_into_the_other_version(self):
        self.assertEqual(bracket_version_of(V12_LEFT_CALIBRATED, NOMINALS), "1.2")
        self.assertEqual(bracket_version_of(V13_LEFT, NOMINALS), "1.3")

    def test_a_pose_between_the_two_is_reported_as_unknown(self):
        midpoint = [(a + b) / 2 for a, b in zip(V12_LEFT[:3], V13_LEFT[:3])] + [90.0, 0.0, -45.0]
        self.assertIsNone(bracket_version_of(midpoint, NOMINALS))

    def test_nominals_are_read_off_the_version_suffixed_keys(self):
        self.assertEqual(marker_bracket_nominals(markers_config(), "left"),
                         {"1.2": V12_LEFT, "1.3": V13_LEFT})


class TestBracketResolution(unittest.TestCase):
    def test_matching_version_keeps_the_calibrated_value(self):
        self.assertEqual(resolve_marker_brackets(markers_config(), "1.2", log=silent), {})

    def test_other_version_is_replaced_by_that_versions_nominal(self):
        changes = resolve_marker_brackets(markers_config(), "1.3", log=silent)
        self.assertEqual(changes, {"left": V13_LEFT, "right": V13_RIGHT})

    def test_a_missing_live_bracket_is_filled_from_the_nominal(self):
        config = markers_config()
        del config["Tf_to_marker_left"]
        self.assertEqual(resolve_marker_brackets(config, "1.2", log=silent)["left"], V12_LEFT)

    def test_an_indistinguishable_bracket_is_left_alone(self):
        midpoint = [(a + b) / 2 for a, b in zip(V12_LEFT[:3], V13_LEFT[:3])] + [90.0, 0.0, -45.0]
        changes = resolve_marker_brackets(markers_config(Tf_to_marker_left=midpoint), "1.3", log=silent)
        self.assertNotIn("left", changes)

    def test_a_version_with_no_nominal_changes_nothing(self):
        self.assertEqual(resolve_marker_brackets(markers_config(), "1.4", log=silent), {})


class TestBindRobotAppliesTheVersion(unittest.TestCase):
    """Runs against a throwaway copy of config/, never the real one."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        shutil.copytree(ROOT / "config", os.path.join(self.folder, "config"))
        saved = storage.CONFIG_PATHS["setting_yaml"]
        self.addCleanup(storage.CONFIG_PATHS.__setitem__, "setting_yaml", saved)
        storage.CONFIG_PATHS["setting_yaml"] = os.path.join(self.folder, "config", "setting.yaml")
        ConfigStorage.update_values(storage.CONFIG_PATHS["setting_yaml"],
                                    {("marker", key): value for key, value in markers_config().items()})

    def observer(self):
        engine = Marker_Transform.__new__(Marker_Transform)
        engine.sim = False
        engine.camera = None
        engine.robot = None
        engine.robot_version = "1.2"
        engine.markers_config = markers_config()
        engine.marker_detection = SimpleNamespace(markers_config=None)
        engine.Tf_to_marker_tf_left = engine.make_transform(V12_LEFT_CALIBRATED)
        engine.Tf_to_marker_tf_right = engine.make_transform(V12_RIGHT_CALIBRATED)
        return engine

    def stored_brackets(self):
        marker = ConfigStorage.load(storage.CONFIG_PATHS["setting_yaml"])["marker"]
        return marker["Tf_to_marker_left"], marker["Tf_to_marker_right"]

    def test_binding_a_v13_robot_reloads_both_brackets_and_saves_them(self):
        engine = self.observer()

        engine.bind_robot(SimpleNamespace(), "1.3")

        self.assertEqual(engine.markers_config["Tf_to_marker_left"], V13_LEFT)
        self.assertEqual(engine.markers_config["Tf_to_marker_right"], V13_RIGHT)
        self.assertEqual(self.stored_brackets(), (V13_LEFT, V13_RIGHT))

    def test_the_runtime_transform_is_rebuilt_not_just_the_config(self):
        engine = self.observer()

        engine.bind_robot(SimpleNamespace(), "1.3")

        # calc_cam_to_tool reads these, so a config-only update would change nothing that matters.
        np.testing.assert_allclose(engine.Tf_to_marker_tf_left,
                                   engine.make_transform(V13_LEFT), atol=1e-6)
        np.testing.assert_allclose(engine.Tf_to_marker_tf_right,
                                   engine.make_transform(V13_RIGHT), atol=1e-6)

    def test_binding_the_same_version_preserves_the_calibrated_bracket(self):
        engine = self.observer()

        engine.bind_robot(SimpleNamespace(), "1.2")

        self.assertEqual(engine.markers_config["Tf_to_marker_left"], V12_LEFT_CALIBRATED)
        self.assertEqual(self.stored_brackets(), (V12_LEFT_CALIBRATED, V12_RIGHT_CALIBRATED))

    def test_disconnecting_does_not_reset_a_v13_setup_to_the_default_version(self):
        # bind_robot(None, "1.2") is what disconnect does; "1.2" there is a default, not a reading.
        engine = self.observer()
        engine.markers_config["Tf_to_marker_left"] = list(V13_LEFT)
        engine.markers_config["Tf_to_marker_right"] = list(V13_RIGHT)
        ConfigStorage.update_values(storage.CONFIG_PATHS["setting_yaml"],
                                    {("marker", "Tf_to_marker_left"): list(V13_LEFT),
                                     ("marker", "Tf_to_marker_right"): list(V13_RIGHT)})

        engine.bind_robot(None, "1.2")

        self.assertEqual(engine.markers_config["Tf_to_marker_left"], V13_LEFT)
        self.assertEqual(self.stored_brackets(), (V13_LEFT, V13_RIGHT))

    def test_a_v_prefixed_version_string_is_accepted(self):
        engine = self.observer()

        engine.bind_robot(SimpleNamespace(), "v1.3")

        self.assertEqual(engine.markers_config["Tf_to_marker_left"], V13_LEFT)


class TestShippedSettingFile(unittest.TestCase):
    def test_setting_yaml_carries_a_nominal_for_both_versions(self):
        marker = ConfigStorage.load(ROOT / "config" / "setting.yaml")["marker"]
        for side in ("left", "right"):
            nominals = marker_bracket_nominals(marker, side)
            self.assertEqual(set(nominals), {"1.2", "1.3"}, side)

    def test_the_live_bracket_matches_one_of_them(self):
        marker = ConfigStorage.load(ROOT / "config" / "setting.yaml")["marker"]
        for side in ("left", "right"):
            version = bracket_version_of(marker[f"Tf_to_marker_{side}"],
                                         marker_bracket_nominals(marker, side))
            self.assertIn(version, ("1.2", "1.3"),
                          f"Tf_to_marker_{side} matches neither bracket; it cannot be assigned "
                          f"to a robot version")


if __name__ == "__main__":
    unittest.main()
