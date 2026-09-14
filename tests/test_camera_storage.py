"""Camera producer, observation and storage tests without hardware."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from core.camera_processing import RealSenseCamera
from core.calibration.observation import ObservationSource
from core.calibration.sequences.result import SequenceCancelled
from core.storage import ConfigStorage, DatasetStorage, FileStorage, ResultStorage


def fake_camera():
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.lock = threading.RLock()
    camera.io_lock = threading.RLock()
    camera.camera_running = True
    camera.camera_monitoring = False
    camera.thread = None
    camera.frame_id = 1
    camera.frame_timestamp = 123.0
    camera.temperature = 32.5
    camera.temperature_timestamp = 120.0
    camera.actual_exposure = 6000.0
    camera.last_error = ""
    camera.color_image = np.ones((4, 5, 3), dtype=np.uint8)
    camera.pipeline = MagicMock()
    camera.profile = None
    return camera


class TestCameraOwnership(unittest.TestCase):
    def test_snapshot_is_detached_and_does_not_capture_or_detect(self):
        camera = fake_camera()
        with patch.object(camera, "capture_image") as capture:
            view = camera.get_monitor_snapshot()
            view["image"][:] = 0
            self.assertEqual(camera.get_camera_temperature(), 32.5)
            self.assertEqual(camera.get_monitor_snapshot()["image"].sum(), 60)
        capture.assert_not_called()
        camera.pipeline.wait_for_frames.assert_not_called()

    def test_capture_has_one_producer_and_stop_joins_thread(self):
        camera = fake_camera()
        frame = MagicMock()
        frame.get_data.return_value = np.full((4, 5, 3), 7, dtype=np.uint8)
        frame.supports_frame_metadata.return_value = False
        camera.pipeline.wait_for_frames.return_value.get_color_frame.return_value = frame
        camera.start_capture()
        first_thread = camera.thread
        camera.start_capture()
        self.assertIs(camera.thread, first_thread)
        deadline = time.monotonic() + 2
        while camera.frame_id < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        camera.stream_off()
        self.assertFalse(first_thread.is_alive())
        self.assertGreater(camera.frame_id, 1)
        self.assertTrue(np.all(camera.get_monitor_snapshot()["image"] == 7))
        camera.pipeline.stop.assert_called_once()

    def test_external_capture_call_does_not_read_pipeline(self):
        camera = fake_camera()
        camera.camera_monitoring = True
        camera.thread = object()
        camera.capture_image()
        camera.pipeline.wait_for_frames.assert_not_called()

    def test_camera_error_keeps_last_frame_and_reports_error(self):
        camera = fake_camera()
        camera.pipeline.wait_for_frames.side_effect = RuntimeError("unplugged")
        camera.capture_image()
        self.assertEqual(camera.get_monitor_snapshot()["error"], "unplugged")
        self.assertEqual(camera.frame_id, 1)

    def test_observation_uses_cached_frame_and_waits_for_fresh_one(self):
        camera = fake_camera()
        engine = SimpleNamespace(camera=camera)
        engine.get_marker_transform = lambda **kw: {"right": engine.camera.get_color_image()}
        event = threading.Event()
        with patch.object(camera, "start_capture"):
            source = ObservationSource(engine, event)
        result = source.get_marker_transform()
        self.assertEqual(result["right"].shape, (4, 5, 3))
        self.assertEqual(source.last_frame["frame_id"], 1)
        event.set()
        with self.assertRaises(SequenceCancelled):
            source.get_marker_transform()
        camera.pipeline.wait_for_frames.assert_not_called()

    def test_observation_serializes_existing_detector(self):
        calls = []
        def detect():
            calls.append("start")
            time.sleep(.01)
            calls.append("end")
            return {"right": np.eye(4)}
        source = ObservationSource(SimpleNamespace(camera=None, get_marker_transform=detect), threading.Event())
        workers = [threading.Thread(target=source.get_marker_transform) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(2)
        self.assertEqual(calls, ["start", "end", "start", "end"])

    def test_observation_rejects_old_image_after_capture_error(self):
        camera = fake_camera()
        camera.last_error = "unplugged"
        engine = SimpleNamespace(camera=camera)
        engine.get_marker_transform = lambda: engine.camera.get_color_image()
        with patch.object(camera, "start_capture"):
            source = ObservationSource(engine, threading.Event())
        with self.assertRaisesRegex(RuntimeError, "unplugged"):
            source.get_marker_transform()


class TestStorage(unittest.TestCase):
    def test_npz_roundtrip_and_legacy_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.npz"
            q, marker, head = np.ones((2, 14)), np.tile(np.eye(4), (2, 1, 1)), np.zeros((2, 2))
            DatasetStorage.save_calibration(path, q, marker, head)
            loaded = DatasetStorage.load_calibration(path)
            for a, b in zip(loaded, (q, head, marker)):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(set(DatasetStorage.load(path)), {"q", "q_arm", "q_head", "marker"})
            DatasetStorage.save(path, q=q, marker=marker)
            legacy = DatasetStorage.load_calibration(path)
            np.testing.assert_array_equal(legacy[0], q)
            self.assertIsNone(legacy[1])

    def test_atomic_json_failure_preserves_previous_document(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            ResultStorage.save(path, {"old": True})
            with patch("core.storage.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    ResultStorage.save(path, {"new": True})
            self.assertEqual(ResultStorage.load(path), {"old": True})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_config_key_update_retains_comments_and_other_sections(self):
        lines = ["camera:\n", "  mount_to_cam: [0, 0, 0, 0, 0, 0] # important\n", "marker:\n", "  size: 0.02\n"]
        ConfigStorage.update_camera_key_in_lines(lines, "mount_to_cam", [1, 2, 3, 4, 5, 6])
        self.assertIn("# important", lines[1])
        self.assertEqual(lines[2:], ["marker:\n", "  size: 0.02\n"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            FileStorage.write_text(path, "".join(lines))
            self.assertEqual(ConfigStorage.load(path)["camera"]["mount_to_cam"], [1, 2, 3, 4, 5, 6])
