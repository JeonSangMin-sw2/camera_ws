"""Static architecture contracts: imports are dependencies, not runtime call graphs."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path):
    return {n.module or "" for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)}


class TestModuleBoundaries(unittest.TestCase):
    def test_language_is_the_only_i18n_entrypoint(self):
        self.assertIn("core.language", imported_modules(ROOT / "main_ui.py"))
        self.assertFalse((ROOT / "core/i18n.py").exists())

    def test_robot_does_not_depend_on_calibration_or_ui(self):
        for path in (ROOT / "core/robot").glob("*.py"):
            self.assertFalse(any(m.startswith(("core.calibration", "ui", "PySide6", "main_ui")) for m in imported_modules(path)), path)

    def test_core_is_widget_independent(self):
        for path in (ROOT / "core").rglob("*.py"):
            if path.name == "language.py":
                continue  # Existing UI language/theme service uses QObject signals.
            self.assertFalse(any(m.startswith(("ui", "PySide6", "main_ui")) for m in imported_modules(path)), path)

    def test_no_camera_or_sequence_implementations_in_ui(self):
        path = ROOT / "main_ui.py"
        modules = imported_modules(path)
        self.assertNotIn("core.marker_detection", modules)
        self.assertNotIn("core.camera_processing", modules)
        tree = ast.parse(path.read_text())
        self.assertFalse([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Worker")])
        forbidden = {"capture_image", "get_marker_transform", "perform_head_sweep", "perform_joint_calibration"}
        self.assertFalse([n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden])

    def test_single_storage_module_and_flat_optimizer(self):
        self.assertTrue((ROOT / "core/storage.py").is_file())
        self.assertFalse((ROOT / "core/storage").exists())
        self.assertTrue((ROOT / "core/calibration/calibration_optimizer.py").is_file())
        self.assertFalse((ROOT / "core/calibration/optimizers").exists())

    def test_camera_class_was_extracted_only_once(self):
        def classes(path):
            return {n.name for n in ast.parse((ROOT / path).read_text()).body if isinstance(n, ast.ClassDef)}
        self.assertIn("RealSenseCamera", classes("core/camera_processing.py"))
        self.assertNotIn("RealSenseCamera", classes("core/marker_detection.py"))
        self.assertTrue({"Marker_Detection", "Marker_Transform", "SimulationModel"} <= classes("core/marker_detection.py"))

    def test_home_offset_lives_under_robot(self):
        self.assertTrue((ROOT / "core/robot/home_offset.py").exists())
        self.assertFalse((ROOT / "core/calibration/homeoffset_core.py").exists())
        self.assertTrue((ROOT / "ui/core_bridge.py").exists())
        self.assertFalse((ROOT / "ui/calibration_bridge.py").exists())
