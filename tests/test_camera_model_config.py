"""The connected camera model decides the bracket extrinsics and the intrinsics.

Regression cover for a swapped camera being ignored: `_load_all_configs` used to run before the
camera was opened, so no model was ever detected -- setting.yaml kept the previous camera's
bracket transform and camera_intrinsics.yaml kept its focal length. A D405 file left in place for
a D435 (fx 660 against a true 916) makes every solvePnP range read ~28% short.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import core.storage as storage
from core.storage import ConfigStorage
from core.marker_detection import Marker_Transform, camera_model_family

ROOT = Path(__file__).resolve().parents[1]

D405_MOUNT = [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]
D405_HEAD_BASE = [0.098, 0.009, 0.012, -90.0, 0.0, -90.0]
D435_MOUNT = [0.0495, 0.0325, 0.057, -90.0, 0.0, -90.0]
D435_HEAD_BASE = [0.1045, -0.0115, 0.044, -90.0, 0.0, -90.0]


class CameraModelConfigCase(unittest.TestCase):
    """Runs every resolution against a throwaway copy of config/, never the real one."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        shutil.copytree(ROOT / "config", os.path.join(self.folder, "config"))
        self.saved = {key: storage.CONFIG_PATHS[key]
                      for key in ("setting_yaml", "camera_info", "camera_intrinsics")}
        self.addCleanup(storage.CONFIG_PATHS.update, self.saved)
        for key, name in (("setting_yaml", "setting.yaml"), ("camera_info", "camera_info.yaml"),
                          ("camera_intrinsics", "camera_intrinsics.yaml")):
            storage.CONFIG_PATHS[key] = os.path.join(self.folder, "config", name)

    def config_path(self, name):
        return os.path.join(self.folder, "config", name)

    def resolve(self, device_name):
        """Run the camera-dependent config resolution as if `device_name` were plugged in."""
        observer = Marker_Transform.__new__(Marker_Transform)
        observer.camera = SimpleNamespace(device_name=device_name) if device_name else None
        observer.marker_detection = SimpleNamespace(markers_config=None)
        observer._load_all_configs()
        return observer

    def camera_settings(self):
        return ConfigStorage.load(storage.CONFIG_PATHS["setting_yaml"])["camera"]

    def write_camera_settings(self, **values):
        ConfigStorage.update_values(storage.CONFIG_PATHS["setting_yaml"],
                                    {("camera", key): value for key, value in values.items()})

    def focal_length(self, path):
        return ConfigStorage.load(path)["camera_matrix"][0][0]


class TestModelFamily(CameraModelConfigCase):
    def test_variant_suffix_does_not_create_a_second_model(self):
        for reported in ("Intel RealSense D435", "Intel RealSense D435i", "Intel RealSense D435f"):
            self.assertEqual(camera_model_family(reported), "D435", reported)
        self.assertEqual(camera_model_family("Intel RealSense D405"), "D405")
        self.assertIsNone(camera_model_family(None))

    def test_a_d435i_is_resolved_as_a_d435_without_rewriting_anything(self):
        self.resolve("Intel RealSense D435")
        before = self.camera_settings()
        self.resolve("Intel RealSense D435i")
        self.assertEqual(self.camera_settings(), before)


class TestExtrinsicsFollowTheCamera(CameraModelConfigCase):
    def test_swapped_camera_loads_its_own_bracket_extrinsics(self):
        self.write_camera_settings(device_name="D405", head_base_to_cam=D405_HEAD_BASE,
                                   mount_to_cam=D405_MOUNT, mount_to_cam_nominal=D405_MOUNT,
                                   head_base_to_cam_nominal=D405_HEAD_BASE)

        self.resolve("Intel RealSense D435")

        camera = self.camera_settings()
        self.assertEqual(camera["device_name"], "D435")
        self.assertEqual(camera["head_base_to_cam"], D435_HEAD_BASE)
        self.assertEqual(camera["mount_to_cam"], D435_MOUNT)

    def test_stale_extrinsics_are_reloaded_even_when_device_name_was_hand_edited(self):
        # device_name already says D435 while every number is still the D405's: comparing the
        # name alone saw no change and left the wrong bracket transform in place.
        self.write_camera_settings(device_name="D435", head_base_to_cam=D405_HEAD_BASE,
                                   mount_to_cam=D405_MOUNT, mount_to_cam_nominal=D405_MOUNT,
                                   head_base_to_cam_nominal=D405_HEAD_BASE)

        self.resolve("Intel RealSense D435")

        camera = self.camera_settings()
        self.assertEqual(camera["head_base_to_cam"], D435_HEAD_BASE)
        self.assertEqual(camera["mount_to_cam"], D435_MOUNT)

    def test_nominals_track_the_connected_model(self):
        # Step 1.5 and Step 2 anchor their baseline to the nominals; nothing refreshed them, so a
        # D435 was being calibrated against the D405's CAD pose.
        self.write_camera_settings(device_name="D405", head_base_to_cam=D405_HEAD_BASE,
                                   mount_to_cam=D405_MOUNT, mount_to_cam_nominal=D405_MOUNT,
                                   head_base_to_cam_nominal=D405_HEAD_BASE)

        self.resolve("Intel RealSense D435")

        camera = self.camera_settings()
        self.assertEqual(camera["mount_to_cam_nominal"], D435_MOUNT)
        self.assertEqual(camera["head_base_to_cam_nominal"], D435_HEAD_BASE)

    def test_calibrated_extrinsics_survive_a_restart_on_the_same_model(self):
        self.resolve("Intel RealSense D435")
        calibrated = [0.0495, 0.0325, 0.057, -90.0, -0.4, -90.0]
        self.write_camera_settings(mount_to_cam=calibrated)

        self.resolve("Intel RealSense D435")

        camera = self.camera_settings()
        self.assertEqual(camera["mount_to_cam"], calibrated)
        self.assertEqual(camera["mount_to_cam_nominal"], D435_MOUNT)

    def test_unknown_model_reports_the_gap_and_changes_nothing(self):
        self.resolve("Intel RealSense D435")
        before = self.camera_settings()

        observer = self.resolve("Intel RealSense D455")

        self.assertTrue(observer.extrinsics_missing)
        self.assertEqual(self.camera_settings(), before)


class TestIntrinsicsFollowTheCamera(CameraModelConfigCase):
    """Each model's focal length is read from its own store rather than written in here: the
    stores hold real calibrations and get replaced whenever a camera is recalibrated."""

    def stored_focal_length(self, family):
        return self.focal_length(ROOT / "config" / f"camera_intrinsics_{family}.yaml")

    def test_another_models_intrinsics_are_replaced_not_applied(self):
        ConfigStorage.save(storage.CONFIG_PATHS["camera_intrinsics"],
                           ConfigStorage.load(self.config_path("camera_intrinsics_d405.yaml")))

        observer = self.resolve("Intel RealSense D435")

        self.assertFalse(observer.intrinsics_missing)
        self.assertFalse(observer.intrinsics_mismatch)
        active = ConfigStorage.load(observer.active_intrinsics_path)
        self.assertEqual(active["device_name"], "D435")
        self.assertAlmostEqual(active["camera_matrix"][0][0], self.stored_focal_length("d435"))
        self.assertNotAlmostEqual(active["camera_matrix"][0][0], self.stored_focal_length("d405"))

    def test_the_outgoing_models_calibration_is_archived_before_the_swap(self):
        self.resolve("Intel RealSense D435")

        self.resolve("Intel RealSense D405")

        self.assertAlmostEqual(self.focal_length(self.config_path("camera_intrinsics_d435.yaml")),
                               self.stored_focal_length("d435"))
        self.assertAlmostEqual(self.focal_length(self.config_path("camera_intrinsics.yaml")),
                               self.stored_focal_length("d405"))

    def test_no_stored_calibration_falls_back_to_factory_rather_than_a_wrong_file(self):
        observer = self.resolve("Intel RealSense D455")

        self.assertTrue(observer.intrinsics_missing)
        self.assertIsNone(observer.active_intrinsics_path,
                          "a D455 must not be given another model's focal length")

    def test_simulation_keeps_whatever_is_on_file(self):
        observer = self.resolve(None)

        self.assertIsNone(observer.camera_model)
        self.assertFalse(observer.intrinsics_missing)
        self.assertFalse(observer.extrinsics_missing)

    def test_an_unidentified_working_file_is_archived_before_being_replaced(self):
        # Versions before the per-model stores wrote no device_name, and the shipped
        # camera_intrinsics_d435i.yaml had none either. Skipping the archive for those -- the
        # only files that cannot be re-derived from their name -- threw the calibration away.
        mine = {"device_name": None,
                "camera_matrix": [[111.0, 0.0, 640.0], [0.0, 222.0, 360.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0] * 5, "width": 1280, "height": 720, "rms_error": 0.11}
        del mine["device_name"]
        ConfigStorage.save(storage.CONFIG_PATHS["camera_intrinsics"], mine)

        self.resolve("Intel RealSense D435")

        archived = [path for path in Path(self.folder, "config").glob("camera_intrinsics*.yaml")
                    if abs(self.focal_length(path) - 111.0) < 1e-6]
        self.assertTrue(archived, "the outgoing calibration was replaced without being archived")


class TestFrozenBuild(CameraModelConfigCase):
    """A packaged build reads and writes config/ next to the executable, seeded from the bundle."""

    def freeze(self, bundle_config):
        sys.frozen = True
        sys._MEIPASS = str(bundle_config.parent)
        self.addCleanup(lambda: [delattr(sys, name) for name in ("frozen", "_MEIPASS")
                                 if hasattr(sys, name)])

    def test_the_store_is_read_from_the_bundle_when_config_was_not_seeded(self):
        # initialize_defaults() copies the bundle's config/ next to the executable on first run.
        # When that copy has not happened the model's store is still in the bundle; camera_info
        # already falls back there, and without the same fallback a packaged build quietly drops
        # to factory intrinsics on a camera it does ship a calibration for.
        bundle = Path(self.folder, "bundle", "config")
        shutil.copytree(ROOT / "config", bundle)
        for name in Path(self.folder, "config").glob("camera_intrinsics*.yaml"):
            name.unlink()
        self.freeze(bundle)

        observer = self.resolve("Intel RealSense D435f")

        self.assertFalse(observer.intrinsics_missing)
        self.assertEqual(ConfigStorage.load(observer.active_intrinsics_path)["device_name"], "D435")

    def test_a_d435f_is_configured_as_a_d435_end_to_end(self):
        # Start from a D405 setup: the shipped setting.yaml holds whatever the last camera was
        # (after a calibration its mount_to_cam is no longer the nominal), so the test sets it.
        self.write_camera_settings(device_name="D405", mount_to_cam=D405_MOUNT, mount_to_cam_nominal=D405_MOUNT)
        observer = self.resolve("Intel RealSense D435f")

        camera = self.camera_settings()
        self.assertEqual(observer.camera_model, "D435")
        self.assertEqual(camera["device_name"], "D435")
        self.assertEqual(camera["mount_to_cam"], D435_MOUNT)
        self.assertEqual(ConfigStorage.load(observer.active_intrinsics_path)["device_name"], "D435")
        self.assertFalse(observer.extrinsics_missing)
        self.assertFalse(observer.intrinsics_missing)


class TestShippedConfigFiles(unittest.TestCase):
    def test_camera_info_keys_are_families_with_real_measurements(self):
        info = ConfigStorage.load(ROOT / "config" / "camera_info.yaml")
        for key, entry in info.items():
            self.assertEqual(key, camera_model_family(key),
                             f"{key} is a variant; camera_info.yaml is keyed by model family")
            for field in ("head_base_to_cam", "mount_to_cam"):
                self.assertTrue(any(value for value in entry[field][:3]),
                                f"{key}.{field} is all zeros -- a placeholder entry is applied as "
                                f"if it were measured, so leave the model out instead")

    def test_every_stored_intrinsics_file_names_its_model(self):
        for path in (ROOT / "config").glob("camera_intrinsics_*.yaml"):
            family = path.stem.rsplit("_", 1)[-1].upper()
            data = ConfigStorage.load(path)
            self.assertEqual(data.get("device_name"), family, path.name)


if __name__ == "__main__":
    unittest.main()


class TestSavedIntrinsicsApplyImmediately(unittest.TestCase):
    """Saving an intrinsics calibration has to change what the running detector uses.

    The detector only took intrinsics at camera start-up, so calibrating, saving and carrying on
    in the same session ran every sweep on the old values while the file held the new ones.
    """

    FACTORY = [640.0, 360.0, 900.0, 900.0]            # ppx, ppy, fx, fy

    def setUp(self):
        # These check the apply-on-save mechanics, so every calibrated value has to show: run
        # them in "full" mode whatever the module default is.
        from unittest.mock import patch
        patcher = patch("core.marker_detection.calib_intrinsics_mode", "full")
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self):
        from core.marker_detection import Marker_Detection
        engine = Marker_Transform.__new__(Marker_Transform)
        engine.marker_detection = Marker_Detection()
        engine.width, engine.height = 1280, 720
        engine.camera_model = "D435"
        engine.active_intrinsics_path = None
        engine._factory_intrinsics = list(self.FACTORY)
        engine._factory_dist_coeffs = [0.0] * 5
        return engine

    def saved_file(self, fx=911.0, ppy=384.5):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "camera_intrinsics.yaml")
        ConfigStorage.save(path, {
            "device_name": "D435", "width": 1280, "height": 720, "rms_error": 0.2,
            "camera_matrix": [[fx, 0.0, 639.0], [0.0, fx, ppy], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.1, -0.3, 0.0, 0.0, 0.2]})
        return path

    def test_a_saved_file_is_in_use_straight_away(self):
        engine = self.engine()
        path = self.saved_file()

        self.assertTrue(engine.apply_calibrated_intrinsics(path))

        detector = engine.marker_detection
        self.assertAlmostEqual(detector.fx, 911.0)
        self.assertAlmostEqual(detector.principal_point[1], 384.5)
        self.assertAlmostEqual(float(detector.dist_coeffs[0]), 0.1)
        self.assertEqual(engine.active_intrinsics_path, path)

    def test_saving_again_replaces_the_previous_values(self):
        engine = self.engine()
        engine.apply_calibrated_intrinsics(self.saved_file(fx=911.0))

        engine.apply_calibrated_intrinsics(self.saved_file(fx=905.0, ppy=380.0))

        self.assertAlmostEqual(engine.marker_detection.fx, 905.0)
        self.assertAlmostEqual(engine.marker_detection.principal_point[1], 380.0)

    def test_no_file_falls_back_to_factory(self):
        engine = self.engine()
        engine.apply_calibrated_intrinsics(self.saved_file())

        self.assertFalse(engine.apply_calibrated_intrinsics(None))

        self.assertAlmostEqual(engine.marker_detection.fx, self.FACTORY[2])
        self.assertAlmostEqual(engine.marker_detection.principal_point[1], self.FACTORY[1])

    def test_the_core_applies_it_to_the_live_observer(self):
        import threading
        from core.calibration import CalibrationCore
        engine = self.engine()
        core = CalibrationCore()
        self.addCleanup(core.close)
        core.observer = SimpleNamespace(engine=engine, lock=threading.RLock(), close=lambda: None)

        self.assertTrue(core.apply_camera_intrinsics(self.saved_file(fx=907.0)))
        self.assertAlmostEqual(engine.marker_detection.fx, 907.0)

    def test_the_core_reports_no_camera_instead_of_failing(self):
        from core.calibration import CalibrationCore
        core = CalibrationCore()
        self.addCleanup(core.close)
        self.assertFalse(core.apply_camera_intrinsics(self.saved_file()))


class TestFrozenTranslationsFillNewKeys(unittest.TestCase):
    def test_old_external_i18n_gets_new_keys_from_the_bundle(self):
        import sys
        from unittest.mock import patch
        from core.language import LanguageManager
        from core.storage import ConfigStorage
        with tempfile.TemporaryDirectory() as folder:
            bundle = os.path.join(folder, "bundle")
            os.makedirs(os.path.join(bundle, "config", "ui_config"))
            ConfigStorage.save(os.path.join(bundle, "config", "ui_config", "i18n.yaml"),
                               {"a": {"old": {"en": "bundled old"}, "new": {"en": "bundled new"}}})
            external = os.path.join(folder, "i18n.yaml")
            ConfigStorage.save(external, {"a": {"old": {"en": "external old"}}})
            manager = LanguageManager.__new__(LanguageManager)
            manager.translations = {}
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", bundle, create=True):
                manager.load_translations(external)
        self.assertEqual(manager.translations["a"]["old"]["en"], "external old")
        self.assertEqual(manager.translations["a"]["new"]["en"], "bundled new")


class TestPrincipalPointMode(unittest.TestCase):
    """"principal_point" mode: the calibrated principal point with the camera's own focal length
    and distortion."""

    def test_only_the_principal_point_comes_from_the_file(self):
        from unittest.mock import patch
        patcher = patch("core.marker_detection.calib_intrinsics_mode", "principal_point")
        patcher.start()
        self.addCleanup(patcher.stop)
        case = TestSavedIntrinsicsApplyImmediately()
        engine = case.engine()
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "camera_intrinsics.yaml")
        ConfigStorage.save(path, {
            "device_name": "D435", "width": 1280, "height": 720, "rms_error": 0.2,
            "camera_matrix": [[911.0, 0.0, 639.0], [0.0, 911.0, 384.5], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.1, -0.3, 0.0, 0.0, 0.2]})

        self.assertTrue(engine.apply_calibrated_intrinsics(path))

        detector = engine.marker_detection
        self.assertAlmostEqual(detector.principal_point[0], 639.0)
        self.assertAlmostEqual(detector.principal_point[1], 384.5)
        self.assertAlmostEqual(detector.fx, case.FACTORY[2])
        self.assertTrue(all(abs(float(v)) < 1e-12 for v in detector.dist_coeffs))
