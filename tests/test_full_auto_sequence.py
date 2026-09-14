import unittest
from unittest.mock import MagicMock, patch
import threading
from core.calibration.sequences.step1 import execute_step1_sequence
from ui.core_bridge import FullAutoWorker
from core.calibration import CalibrationCore, SequenceResult


class TestFullAutoSequenceContracts(unittest.TestCase):
    def setUp(self):
        self.joint_calibrator = MagicMock()
        self.marker_calibrator = MagicMock()
        self.joint_offsets_store = {
            "right": {"joint6": 0.0, "joint5": 0.0, "joint3": 0.0},
            "left": {"joint6": 0.0, "joint5": 0.0, "joint3": 0.0}
        }
        self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES = {
            "1.3": {"right": [0.1, 0.0, 0.0, 0, 0, 0], "left": [0.1, 0.0, 0.0, 0, 0, 0]},
            "1.2": {"right": [0.1, 0.0, 0.0, 0, 0, 0], "left": [0.1, 0.0, 0.0, 0, 0, 0]}
        }
        self.marker_calibrator.camera_config = {}

    def test_robot_not_connected_returns_failed_result(self):
        self.joint_calibrator.robot = None
        result = execute_step1_sequence(
                joint_calibrator=self.joint_calibrator,
                marker_calibrator=self.marker_calibrator,
                joint_offsets_store=self.joint_offsets_store
            )
        self.assertEqual(result.status, "failed")
        self.assertIn("Robot is not connected", result.error)

    def test_stop_event_triggers_early_return(self):
        self.joint_calibrator.robot = MagicMock()
        self.marker_calibrator.get_robot_version.return_value = "1.3"
        self.marker_calibrator.perform_move_to_ready_pose.return_value = True

        stop_event = threading.Event()
        stop_event.set()  # Already stopped before sweeps start

        logs = []
        execute_step1_sequence(
            joint_calibrator=self.joint_calibrator,
            marker_calibrator=self.marker_calibrator,
            joint_offsets_store=self.joint_offsets_store,
            stop_event=stop_event,
            log_callback=logs.append
        )
        # Should exit without calling marker sweep
        self.marker_calibrator.perform_calibration_sweep.assert_not_called()

    def test_full_auto_worker_delegates_to_core(self):
        core = MagicMock(spec=CalibrationCore)
        core.run.return_value = SequenceResult("full").finish("completed")
        worker = FullAutoWorker(core)
        core.prepare_run.assert_called_once()
        worker.run()
        core.run.assert_called_once_with("full", prepared=True)
        self.assertTrue(worker.result.success)


if __name__ == "__main__":
    unittest.main()
