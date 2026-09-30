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
    camera._frame_times = []
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


def fake_sensor(streams, low, high, step=1.0, default=1.0):
    """A sensor exposing the given stream types and exposure range; records what is written."""
    import pyrealsense2 as rs
    sensor = SimpleNamespace(written={})
    sensor.get_stream_profiles = lambda: [SimpleNamespace(stream_type=lambda t=t: t) for t in streams]
    sensor.supports = lambda option: option in (rs.option.exposure, rs.option.enable_auto_exposure,
                                               rs.option.auto_exposure_priority)
    sensor.get_option_range = lambda option: SimpleNamespace(min=low, max=high, step=step, default=default)
    sensor.set_option = lambda option, value: sensor.written.__setitem__(option, value)
    sensor.get_option = lambda option: sensor.written.get(option, default)
    return sensor


def camera_with(sensors):
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.lock = threading.RLock()
    camera.io_lock = threading.RLock()
    camera.camera_running = True
    camera.profile = SimpleNamespace(
        get_device=lambda: SimpleNamespace(query_sensors=lambda: sensors))
    return camera


class TestExposureFollowsTheColourSensor(unittest.TestCase):
    """Exposure belongs to whichever sensor produces the colour stream, and only to that one.

    Which sensor that is, and the range and unit of its exposure option, differ by model: a D405
    streams colour from the stereo module, a D435 from a separate RGB camera whose range is far
    smaller. Writing one number to every sensor that accepts `exposure` set a sane value on one
    and a nonsensical one on the other, and the UI's fixed floor of 100 put short exposures out
    of reach on a D435 entirely.
    """

    def stereo_and_rgb(self):
        import pyrealsense2 as rs
        stereo = fake_sensor([rs.stream.depth, rs.stream.infrared], 1.0, 165000.0, default=8500.0)
        rgb = fake_sensor([rs.stream.color], 1.0, 10000.0, default=156.0)
        return stereo, rgb

    def test_range_is_read_from_the_camera_not_assumed(self):
        stereo, rgb = self.stereo_and_rgb()
        self.assertEqual(camera_with([stereo, rgb]).get_exposure_range()[:2], (1.0, 10000.0))

    def test_a_value_below_the_old_fixed_floor_reaches_the_colour_sensor(self):
        import pyrealsense2 as rs
        stereo, rgb = self.stereo_and_rgb()

        camera_with([stereo, rgb]).set_exposure(50)

        self.assertEqual(rgb.written[rs.option.exposure], 50.0)
        self.assertNotIn(rs.option.exposure, stereo.written,
                         "the depth sensor's exposure is not the colour exposure")

    def test_a_value_the_camera_cannot_take_is_clamped_not_rejected(self):
        import pyrealsense2 as rs
        stereo, rgb = self.stereo_and_rgb()

        self.assertTrue(camera_with([stereo, rgb]).set_exposure(60000))

        self.assertEqual(rgb.written[rs.option.exposure], 10000.0)

    def test_a_camera_whose_stereo_module_carries_colour_is_driven_there(self):
        import pyrealsense2 as rs
        # A D405 has no separate RGB camera; colour comes off the stereo module.
        stereo = fake_sensor([rs.stream.depth, rs.stream.color], 1.0, 200000.0, default=33000.0)

        camera = camera_with([stereo])
        camera.set_exposure(6000)

        self.assertEqual(camera.get_exposure_range()[:2], (1.0, 200000.0))
        self.assertEqual(stereo.written[rs.option.exposure], 6000.0)
        self.assertEqual(camera.get_exposure(), (False, 6000.0))

    def test_no_colour_sensor_is_reported_rather_than_guessed(self):
        import pyrealsense2 as rs
        stereo = fake_sensor([rs.stream.depth], 1.0, 165000.0)

        camera = camera_with([stereo])

        self.assertIsNone(camera.get_exposure_range())
        self.assertFalse(camera.set_exposure(100))
        self.assertNotIn(rs.option.exposure, stereo.written)

    def test_auto_mode_keeps_the_frame_rate_instead_of_a_longer_exposure(self):
        import pyrealsense2 as rs
        stereo, rgb = self.stereo_and_rgb()

        camera_with([stereo, rgb]).set_exposure(0, auto_exposure=True)

        self.assertEqual(rgb.written[rs.option.enable_auto_exposure], 1)
        self.assertEqual(rgb.written[rs.option.auto_exposure_priority], 0,
                         "auto exposure defaults to trading frame rate for exposure; left on, it "
                         "silently ran the dimmer arm's sweeps at 19.6 fps instead of 31.0")


class TestResolutionIsNotNegotiatedDown(unittest.TestCase):
    """A camera that cannot open the configured resolution must stop, not quietly drop to a
    lower one: the intrinsics and results are only valid at the configured size, and the usual
    cause is a USB 2 link that should be fixed with a cable, not worked around."""

    def test_refused_profile_raises_with_a_cable_hint_and_never_falls_back(self):
        from core.camera_processing import CameraUnavailableError
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.serial_number = "test"
        camera.config = MagicMock()
        camera.pipeline = MagicMock()
        camera.pipeline.start.side_effect = RuntimeError("Couldn't resolve requests")

        with self.assertRaises(CameraUnavailableError) as caught:
            camera.initialize_camera(1280, 720, 30)

        self.assertIn("USB 3", str(caught.exception))
        self.assertEqual(camera.pipeline.start.call_count, 1, "no lower-resolution retry")
        self.assertEqual((camera.width, camera.height), (1280, 720))


class TestStreamInfoForTheWizard(unittest.TestCase):
    def test_reports_the_negotiated_stream_and_the_measured_rate(self):
        camera = fake_camera()
        camera.width, camera.height, camera.fps = 1280, 720, 30
        camera.device_name, camera.serial_number = "Intel RealSense D435I", "123"
        stream = SimpleNamespace(width=lambda: 848, height=lambda: 480, fps=lambda: 15)
        camera.profile = SimpleNamespace(get_stream=lambda kind: SimpleNamespace(as_video_stream_profile=lambda: stream))
        now = time.time()
        camera._frame_times = [now - 1.0 + i * 0.05 for i in range(21)]
        info = camera.get_stream_info()
        self.assertEqual((info["width"], info["height"], info["fps"]), (848, 480, 15))
        self.assertAlmostEqual(info["measured_fps"], 20.0, places=3)
        self.assertEqual(info["device_name"], "Intel RealSense D435I")

    def test_no_rate_before_enough_frames(self):
        camera = fake_camera()
        camera._frame_times = [time.time()]
        self.assertIsNone(camera.measured_fps())

    def test_capture_records_frame_times(self):
        camera = fake_camera()
        frame = MagicMock()
        frame.get_data.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        frame.supports_frame_metadata.return_value = False
        camera.pipeline.wait_for_frames.return_value.get_color_frame.return_value = frame
        for _ in range(3):
            camera.capture_image()
        self.assertEqual(len(camera._frame_times), 3)


class TestExposureUnitFollowsTheColourModule(unittest.TestCase):
    """A D435's RGB module is a UVC camera (exposure in 100 us steps); a D405's colour comes from
    the stereo module (microseconds). "100" is 10 ms on the first and 0.1 ms on the second."""

    def named(self, sensor, name):
        sensor.get_info = lambda info: name
        return sensor

    def test_rgb_module_counts_in_100_us(self):
        import pyrealsense2 as rs
        stereo = self.named(fake_sensor([rs.stream.depth], 1.0, 165000.0), "Stereo Module")
        rgb = self.named(fake_sensor([rs.stream.color], 1.0, 10000.0), "RGB Camera")
        self.assertEqual(camera_with([stereo, rgb]).get_exposure_unit_us(), 100.0)

    def test_colour_from_the_stereo_module_counts_in_us(self):
        import pyrealsense2 as rs
        stereo = self.named(fake_sensor([rs.stream.depth, rs.stream.color], 1.0, 165000.0), "Stereo Module")
        self.assertEqual(camera_with([stereo]).get_exposure_unit_us(), 1.0)
