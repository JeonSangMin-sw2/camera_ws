"""Real-camera check of production preview/worker ownership; no robot connection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import threading
import time
from types import MethodType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication
from core.marker_detection import Marker_Transform
from main_ui import UnifiedCalibrationApp


if __name__ == '__main__':
    qt = QApplication([])
    marker = Marker_Transform(sim=False)
    if marker.sim or marker.camera is None:
        raise RuntimeError('Real camera required')
    marker.set_marker_type('plate')
    captures = {'gui': 0, 'worker': 0}
    original = marker.camera.capture_image
    gui_ident = threading.get_ident()
    def capture(*args, **kwargs):
        captures['gui' if threading.get_ident() == gui_ident else 'worker'] += 1
        return original(*args, **kwargs)
    marker.camera.capture_image = capture
    class Reader(QThread):
        error = None
        def run(self):
            try:
                deadline = time.monotonic() + 6.
                while time.monotonic() < deadline:
                    marker.get_marker_transform(sampling_time=0, side='right', use_filter=False)
                    time.sleep(.01)
            except Exception as error:
                self.error = error
    reader = Reader()
    app = SimpleNamespace(sim=False, marker_st=marker, active_worker=reader,
        marker_problem_dlg=None, left_tabs=SimpleNamespace(currentIndex=lambda: 1),
        step1_tabs=SimpleNamespace(currentIndex=lambda: 1),
        update_marker_indicator=lambda value: None)
    app._camera_capture_owned_by_worker = MethodType(UnifiedCalibrationApp._camera_capture_owned_by_worker, app)
    timer = QTimer()
    # Record only calls where the worker is still active; after it stops the
    # normal idle preview is explicitly permitted to acquire again.
    def update():
        if reader.isRunning():
            UnifiedCalibrationApp.update_video_frame(app)
    timer.timeout.connect(update)
    reader.finished.connect(qt.quit)
    try:
        reader.start()
        timer.start(50)
        qt.exec()
        reader.wait()
        timer.stop()
        print('DURING_WORKER', captures, flush=True)
        assert reader.error is None, reader.error
        assert captures['gui'] == 0, captures
        assert captures['worker'] > 100, captures
        assert app.current_frame is not None
        UnifiedCalibrationApp.update_video_frame(app)
        assert captures['gui'] > 0, captures
        print('LIVE_PREVIEW_CAPTURE_PASS', captures, app.current_frame.shape, flush=True)
    finally:
        reader.wait()
        marker.camera.stream_off()
