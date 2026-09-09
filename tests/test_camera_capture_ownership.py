"""Preview must not consume the worker's RealSense frames during a sweep."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PySide6.QtWidgets import QApplication
from main_ui import UnifiedCalibrationApp


class CameraCaptureOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def app(self, running=True, teaching=False):
        image = np.full((24, 32, 3), 42, dtype=np.uint8)
        camera = SimpleNamespace(capture_image=Mock(), get_color_image=Mock(return_value=image))
        marker = SimpleNamespace(camera=camera, get_marker_transform=Mock(return_value=[]))
        app = SimpleNamespace(sim=False, marker_st=marker,
            active_worker=SimpleNamespace(isRunning=lambda: running),
            marker_problem_dlg=SimpleNamespace() if teaching else None,
            left_tabs=SimpleNamespace(currentIndex=lambda: 1),
            step1_tabs=SimpleNamespace(currentIndex=lambda: 1),
            update_marker_indicator=Mock(), btn_monitor=SimpleNamespace(isChecked=lambda: False))
        if hasattr(UnifiedCalibrationApp, '_camera_capture_owned_by_worker'):
            app._camera_capture_owned_by_worker = MethodType(
                UnifiedCalibrationApp._camera_capture_owned_by_worker, app)
        return app, image

    def test_running_worker_preview_renders_cached_image_without_consuming_frames(self):
        app, image = self.app()
        UnifiedCalibrationApp.update_video_frame(app)
        app.marker_st.camera.capture_image.assert_not_called()
        app.marker_st.get_marker_transform.assert_not_called()
        np.testing.assert_array_equal(app.current_frame, image)

    def test_poll_does_not_restart_detection_during_worker_capture(self):
        app, _ = self.app()
        app.step1_tabs.currentIndex = lambda: 0
        UnifiedCalibrationApp.poll_camera_status(app)
        app.marker_st.get_marker_transform.assert_not_called()

    def test_worker_paused_between_captures_keeps_last_frame_and_labels_wait(self):
        app, image = self.app()
        app.current_frame = image.copy()
        app.marker_st.camera.get_color_image.return_value = None
        with patch('main_ui.cv2.putText') as text:
            UnifiedCalibrationApp.update_video_frame(app)
        np.testing.assert_array_equal(app.current_frame, image)
        self.assertTrue(any('Waiting for' in call.args[1] for call in text.call_args_list))
        self.assertFalse(any('No Camera' in call.args[1] for call in text.call_args_list))
        app.marker_st.camera.capture_image.assert_not_called()

    def test_idle_and_teaching_preview_can_still_acquire(self):
        for running, teaching in [(False, False), (True, True)]:
            app, image = self.app(running, teaching)
            UnifiedCalibrationApp.update_video_frame(app)
            app.marker_st.camera.capture_image.assert_called_once()
            app.marker_st.get_marker_transform.assert_called_once()
            np.testing.assert_array_equal(app.current_frame, image)
