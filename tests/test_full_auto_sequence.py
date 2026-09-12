import unittest
from unittest.mock import MagicMock, patch
import threading
from core.calibration.FullAutoSequence import execute_full_auto_sequence
from main_ui import FullAutoWorker


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

    def test_robot_not_connected_raises_runtime_error(self):
        self.joint_calibrator.robot = None
        with self.assertRaises(RuntimeError) as ctx:
            execute_full_auto_sequence(
                joint_calibrator=self.joint_calibrator,
                marker_calibrator=self.marker_calibrator,
                joint_offsets_store=self.joint_offsets_store
            )
        self.assertIn("Robot is not connected", str(ctx.exception))

    def test_stop_event_triggers_early_return(self):
        self.joint_calibrator.robot = MagicMock()
        self.marker_calibrator.get_robot_version.return_value = "1.3"
        self.marker_calibrator.perform_move_to_ready_pose.return_value = True

        stop_event = threading.Event()
        stop_event.set()  # Already stopped before sweeps start

        logs = []
        execute_full_auto_sequence(
            joint_calibrator=self.joint_calibrator,
            marker_calibrator=self.marker_calibrator,
            joint_offsets_store=self.joint_offsets_store,
            stop_event=stop_event,
            log_callback=logs.append
        )
        # Should exit without calling marker sweep
        self.marker_calibrator.perform_calibration_sweep.assert_not_called()

    def test_full_auto_worker_delegates_to_execute_full_auto_sequence(self):
        worker = FullAutoWorker(
            joint_calibrator=self.joint_calibrator,
            marker_calibrator=self.marker_calibrator,
            joint_offsets_store=self.joint_offsets_store
        )
        with patch("main_ui.execute_full_auto_sequence") as mock_exec:
            worker.run()
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args[1]
            self.assertEqual(call_kwargs["joint_calibrator"], self.joint_calibrator)
            self.assertEqual(call_kwargs["marker_calibrator"], self.marker_calibrator)
            self.assertEqual(call_kwargs["joint_offsets_store"], self.joint_offsets_store)


if __name__ == "__main__":
    unittest.main()
