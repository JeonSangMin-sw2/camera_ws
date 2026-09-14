"""Offline core/sequence/cancellation contracts. No device connections."""
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, create_autospec
import numpy as np
from core.calibration import CalibrationCore, SequenceResult
from core.robot.robot_core import motion_cancellation, check_motion_cancelled, RobotOperationCancelled


class TestCoreExecution(unittest.TestCase):
    def setUp(self):
        self.core = CalibrationCore()
        self.core.robot = MagicMock()
        self.core.observer = SimpleNamespace(sim=True, last_frame={})

    def test_unknown_sequence_failure_releases_lock(self):
        result = self.core.run("unknown")
        self.assertEqual(result.status, "failed")
        self.assertFalse(self.core.is_busy)
        self.assertEqual(self.core.run("step1").status, "failed")

    def test_missing_camera_fails_before_any_marker_motion(self):
        self.core.observer = None
        with patch.object(self.core.marker_calibrator, "perform_move_to_ready_pose") as ready:
            result = self.core.run("marker")
        self.assertEqual(result.status, "failed")
        ready.assert_not_called()

    def test_reserved_worker_can_be_cancelled_before_start(self):
        self.core.prepare_run()
        self.core.cancel()
        with patch("core.calibration.sequences.step1.execute_step1_sequence") as step:
            result = self.core.run("step1", prepared=True)
        self.assertEqual(result.status, "cancelled")
        step.assert_not_called()
        self.assertFalse(self.core.is_busy)

    def test_second_sequence_and_camera_reconnect_are_rejected(self):
        self.core.prepare_run()
        with self.assertRaises(RuntimeError):
            self.core.run("step1")
        with self.assertRaises(RuntimeError):
            self.core.connect_camera()
        self.core.cancel()
        self.core.run("step1", prepared=True)

    def test_full_reuses_steps_in_order_and_passes_memory_forward(self):
        calls = []
        def step1(*args, **kwargs):
            calls.append("step1")
            self.core.joint_offsets_store["right"]["joint6"] = 2.0
            return SequenceResult("step1", completed={"joint": 2.0}).finish("completed")
        def head(core, ctx, **kwargs):
            calls.append("step1_5")
            self.assertEqual(core.joint_offsets_store["right"]["joint6"], 2.0)
            core.accept_head_result({"success": True, "head_offsets_deg": {"pan": 1.0}})
            ctx.complete("head_camera", {"success": True})
        def optimize(core, **kwargs):
            calls.append("step2")
            self.assertEqual(core.joint_offsets_store["head"]["pan"], 1.0)
            return {"offset": [1.0]}
        samples = [{"q_arm": np.zeros(14), "q_head": np.zeros(2), "marker": np.eye(4)}]
        with patch("core.calibration.sequences.step1.execute_step1_sequence", side_effect=step1), patch("core.calibration.sequences.step1_5.run_step1_5", side_effect=head), patch("core.calibration.sequences.step2.optimize_step2", side_effect=optimize):
            result = self.core.run("full", step2={"samples": samples})
        self.assertTrue(result.success, result.error)
        self.assertEqual(calls, ["step1", "step1_5", "step2"])
        self.assertEqual(list(result.completed), calls)

    def test_full_cancel_preserves_prior_step_and_partial_head(self):
        def head(core, ctx, **kwargs):
            ctx.checkpoint("tilt_points", [[1, 2, 3]])
            core.cancel()
            ctx.check_cancelled()
        with patch("core.calibration.sequences.step1.execute_step1_sequence", return_value=SequenceResult("step1", completed={"joint": 3}).finish("completed")), patch("core.calibration.sequences.step1_5.run_step1_5", side_effect=head), patch("core.calibration.sequences.step2.optimize_step2") as opt:
            result = self.core.run("full")
        self.assertEqual(result.status, "cancelled")
        self.assertTrue(result.completed["step1"].success)
        self.assertEqual(result.partial["step1_5"].partial["tilt_points"], [[1, 2, 3]])
        opt.assert_not_called()
        result.partial.clear()
        self.assertIn("step1_5", self.core.get_run_status().partial)

    def test_full_failure_does_not_run_next_step(self):
        with patch("core.calibration.sequences.step1.execute_step1_sequence", return_value=SequenceResult("step1").finish("failed", "bad fit")), patch("core.calibration.sequences.step1_5.run_step1_5") as head:
            result = self.core.run("full")
        head.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "bad fit")

    def test_headless_step_is_explicitly_skipped_without_motion(self):
        self.core.include_head_motion = False
        with patch.object(self.core.head_camera_calibrator, "perform_head_sweep") as sweep:
            result = self.core.run("step1_5")
        sweep.assert_not_called()
        self.assertTrue(result.completed["head_camera"]["skipped"])

    def test_partial_calibrator_data_survives_exception(self):
        def joint(**kwargs):
            self.core.joint_calibrator.partial_data["points"] = [1, 2]
            raise RuntimeError("camera offline")
        with patch.object(self.core.joint_calibrator, "perform_joint_calibration", side_effect=joint):
            result = self.core.run("joint")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.partial["joint"]["points"], [1, 2])

    def test_collection_cancel_preserves_acquired_samples(self):
        from core.calibration.sequences import collection
        self.core.robot = MagicMock()
        self.core.observer = SimpleNamespace(sim=True, last_frame={"frame_id": 8})
        sample = (np.zeros(14), np.zeros(2), {"right": np.eye(4)})
        def capture(*args, **kwargs):
            self.core.stop_event.set()
            return sample
        with patch.object(collection, "get_both_arm_config", return_value={"arm_idx": list(range(14))}), patch.object(collection, "get_head_config", return_value={"head_idx": [14, 15]}), patch.object(collection, "execute_auto_motion_step") as move, patch.object(collection, "capture_one_sample", side_effect=capture):
            result = self.core.run("collect", plan=[{}, {}, {}])
        self.assertEqual(result.status, "cancelled", result.error)
        self.assertEqual(len(result.partial["samples"]), 1)
        self.assertEqual(result.partial["next_motion_index"], 1)
        move.assert_called_once()

    def test_home_reset_cancel_preserves_changed_joints(self):
        from core.robot import home_offset
        robot = MagicMock()
        model = SimpleNamespace(right_arm_idx=[0, 1, 2], left_arm_idx=[], head_idx=[])
        event = threading.Event()
        def reset(name):
            event.set()
            return True
        robot.home_offset_reset.side_effect = reset
        with patch.object(home_offset, "validate_home_offset_joint_limits", return_value=(True, "")), patch.object(home_offset, "wait_motion"), motion_cancellation(event):
            with self.assertRaises(RobotOperationCancelled) as caught:
                home_offset.reset_current_pose_home_offsets(robot, model, arm="right", include_head=False)
        robot.home_offset_reset.assert_called_once_with("right_arm_0")
        self.assertEqual(caught.exception.partial["home"]["reset_joints"], ["right_arm_0"])
        robot.power_off.assert_not_called()

    def test_robot_cancel_guard_is_scoped(self):
        event = threading.Event()
        event.set()
        with motion_cancellation(event):
            with self.assertRaises(RobotOperationCancelled):
                check_motion_cancelled()
        check_motion_cancelled()

    def test_marker_sequence_uses_real_signatures_and_stages_result(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        c = create_autospec(MarkerCalibrator, instance=True)
        c.partial_data = {}
        c.user_taught_ready_poses = {}
        c.camera_config = {}
        c.perform_move_to_ready_pose.return_value = True
        c.perform_calibration_sweep.side_effect = [{"axis_opt": [1, 0, 0]} for _ in range(3)]
        c.compute_unified_bracket_calibration.return_value = dict(x_e=10, y_e=20, z_e=30, roll_e=0, pitch_e=0, yaw_e=0)
        c.generate_marker_plot.return_value = False
        self.core.marker_calibrator = c
        self.core.robot = MagicMock()
        self.core.robot.model.return_value.right_arm_idx = list(range(7))
        self.core.robot.get_state.return_value.position = np.zeros(14)
        result = self.core.run("marker", arm_side="right")
        self.assertTrue(result.success, result.error)
        self.assertEqual([call.args[1] for call in c.perform_calibration_sweep.call_args_list], [4, 6, 5])
        self.assertEqual(c.camera_config["Tf_to_marker_right"][:3], [.01, .02, .03])

    def test_camera_factory_is_injected_into_existing_marker_engine(self):
        backend = object()
        core = CalibrationCore(camera_factory=backend)
        engine = MagicMock()
        engine.camera = None
        with patch("core.marker_detection.Marker_Transform", return_value=engine) as factory:
            core.connect_camera(serial_number="test")
        factory.assert_called_once_with(serial_number="test", camera_factory=backend)
        self.assertTrue(all(c.marker_st is core.observer for c in core.calibrators))
        core.close()

    def test_step2_two_pass_orchestration_uses_first_pass_results(self):
        from core.calibration.calibration_optimizer import QPCalibrationOptimizer
        from core.storage import ResultStorage
        self.core.model = SimpleNamespace(right_arm_idx=list(range(7)), left_arm_idx=list(range(7, 14)), head_idx=[])
        self.core.include_head_motion = False
        constructor = create_autospec(QPCalibrationOptimizer)
        optimizers = [MagicMock(), MagicMock()]
        for index, optimizer in enumerate(optimizers):
            optimizer.optimize.return_value = (np.full(14, .01), None, np.zeros(6), [0.] * 6, [0.] * 6)
            optimizer.noise_estimator.as_dict.return_value = {"test": True}
        constructor.side_effect = optimizers
        samples = [{"q_arm": np.zeros(14), "q_head": None, "marker": np.tile(np.eye(4), (2, 1, 1))}]
        with tempfile.TemporaryDirectory() as folder, patch("core.calibration.sequences.step2.QPCalibrationOptimizer", constructor):
            path = Path(folder) / "result.json"
            result = self.core.run("step2", samples=samples, optimization={"result_path": str(path)})
            self.assertTrue(result.success, result.error)
            self.assertEqual(len(ResultStorage.load(path)["joint_offset_deg"]), 14)
        self.assertEqual(constructor.call_count, 2)
        self.assertTrue(constructor.call_args_list[0].kwargs["optimize_arm"])
        self.assertFalse(constructor.call_args_list[1].kwargs["optimize_arm"])
        np.testing.assert_allclose(optimizers[1].optimize.call_args.kwargs["q_arm_offset_init"], .01)
        self.assertIn("optimizer_pass1", result.partial)
