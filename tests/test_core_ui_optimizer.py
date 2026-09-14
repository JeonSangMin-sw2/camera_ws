import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Slot
from core.calibration import CalibrationCore
from core.calibration.calibration_optimizer import CalibrationOptimizer, QPCalibrationOptimizer
from core.calibration.sequences.result import SequenceCancelled
from ui.core_bridge import CoreBridge, SequenceWorker


class TestOptimizerCancellation(unittest.TestCase):
    def test_both_solvers_preserve_last_accepted_iteration(self):
        for cls in (CalibrationOptimizer, QPCalibrationOptimizer):
            with self.subTest(solver=cls.__name__):
                solver = cls.__new__(cls)
                solver.use_head_kinematics = False
                solver.arm_idx = [0, 1]
                solver.max_iter = 5
                solver.stop_event = threading.Event()
                solver.eps = 1e-8
                solver.compute_step = MagicMock(return_value=(np.array([.01, .02]), 1.))
                def update(q, head, camera, delta):
                    solver.stop_event.set()
                    return q + delta, head, camera
                solver.apply_update = update
                with self.assertRaises(SequenceCancelled) as caught:
                    solver.optimize(np.zeros((1, 2)), None, np.eye(4)[None])
                partial = caught.exception.partial["optimizer"]
                self.assertEqual(partial["iterations"], 1)
                np.testing.assert_allclose(partial["q_arm_offset"], [.01, .02])
                solver.compute_step.assert_called_once()


class TestQtBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_returns_cancelled_result_on_gui_thread(self):
        core = CalibrationCore()
        bridge = CoreBridge(core)
        received = []
        main_thread = threading.get_ident()
        class Receiver(QObject):
            @Slot(object)
            def result(self, value):
                received.append((value, threading.get_ident(), core.is_busy))
        receiver = Receiver()
        bridge.result.connect(receiver.result)
        worker = SequenceWorker(core, "step1")
        core.cancel()
        worker.start()
        self.assertTrue(worker.wait(2000))
        self.app.processEvents()
        self.assertEqual(worker.result.status, "cancelled")
        self.assertEqual(received[0][1], main_thread)
        self.assertFalse(received[0][2])

    def test_ui_video_reads_snapshot_without_detection(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        window.left_tabs.setCurrentIndex(1)
        window.step1_tabs.setCurrentIndex(1)
        window.ui_only = False
        observer = MagicMock()
        window.core.observer = observer
        image = np.full((32, 48, 3), 55, dtype=np.uint8)
        with patch.object(window.core, "get_monitor_snapshot", return_value={"image": image, "exposure": 5000.}) as snapshot:
            window.update_video_frame()
        snapshot.assert_called_once()
        observer.get_marker_transform.assert_not_called()
        observer.camera.capture_image.assert_not_called()
        self.assertEqual(window.current_frame.shape, image.shape)
        window.close()

    def test_second_ui_start_does_not_clear_pending_stop(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        window.core.prepare_run()
        window.core.cancel()
        self.assertFalse(window.start_full_auto())
        self.assertTrue(window.core.stop_event.is_set())
        window.core.run("step1", prepared=True)
        window.close()
