"""Two offline regression contracts, adapted from backup ccfd791 tests.

Current API regression contracts; no expectedFailure/skip.
No robot client, camera, UI, motion, or production configuration writes.
"""
import json
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.CalibratorBase import BaseCalibrator
from core.storage import CONFIG_PATHS


class StepContracts(unittest.TestCase):
    def setUp(self):
        # Both contracts run solver/calibrator code that writes into CONFIG_PATHS["txt_dir"]
        # (head_camera_result_latest.json, the sweep debug log, joint_calib_debug_*). Pointed at
        # the real result/ they overwrite an actual calibration run's captures, which Trap 21 in
        # the project notes records as having happened. One redirect for the whole class.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.dict(CONFIG_PATHS, txt_dir=folder.name,
                             result_dir=folder.name, plot_dir=folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_head_solution_preserves_stationary_marker_in_3d(self):
        # Independent intersecting Z-pan/Y-tilt chain from v1.2 URDF.
        # An unknown stationary marker avoids assuming calibrated arm FK.
        nominal = [.047, .009, .057, -90., 0., -90.]
        mount_r = Rotation.from_euler('xyz', nominal[3:], degrees=True).as_matrix()
        mount_t = np.array(nominal[:3])
        marker = np.array([.30, -.04, .02])
        tilt_zero_deg = -2.
        tilt = np.linspace(-10., 10., 11)
        pan = np.linspace(-15., 15., 11)
        angles = [(0., t) for t in tilt] + [(p, 0.) for p in pan]

        def head_r(p, t):
            return (Rotation.from_euler('z', p, degrees=True).as_matrix()
                    @ Rotation.from_euler('y', t, degrees=True).as_matrix())

        observations = np.array([
            mount_r.T @ (head_r(p, t + tilt_zero_deg).T @ marker - mount_t)
            for p, t in angles
        ])
        solver = HeadCameraCalibrator.__new__(HeadCameraCalibrator)
        # Current solver uses FK. Supply an independent analytical chain instead
        # of a connected robot; the stationary-point prior is nominal, not truth.
        solver.robot = SimpleNamespace(
            model=lambda: SimpleNamespace(head_idx=[0, 1]),
            get_dynamics=lambda: None,
            get_state=lambda: SimpleNamespace(position=np.zeros(2)),
        )
        def fk(robot, dynamics, q, ee_link, base_link):
            transform = np.eye(4)
            transform[:3, :3] = head_r(*np.rad2deg(q))
            return transform
        nominal_marker_prior = mount_r @ observations[5] + mount_t
        with patch.object(BaseCalibrator, 'compute_fk', side_effect=fk):
            result = solver._compute_head_camera_solution(
                observations[:11], observations[11:], tilt, pan, nominal, mount_r,
                P_marker_t5_nom=nominal_marker_prior)
        fitted = result['calibrated_mount_to_cam']
        fitted_r = Rotation.from_euler('xyz', fitted[3:], degrees=True).as_matrix()
        fitted_t = np.array(fitted[:3])
        offsets = result['head_offsets_deg']
        reconstructed = np.array([
            head_r(p + offsets['pan'], t + offsets['tilt']) @ (fitted_r @ obs + fitted_t)
            for (p, t), obs in zip(angles, observations)
        ])
        rms_mm = float(np.sqrt(np.mean(np.sum(
            (reconstructed - reconstructed.mean(axis=0)) ** 2, axis=1))) * 1000)
        effective_t = head_r(0., tilt_zero_deg) @ mount_t
        print(json.dumps({'head_stationary_rms_mm': rms_mm,
                          'reported_quality': result['quality'],
                          'head_offsets_deg': offsets,
                          'effective_translation_m': effective_t.tolist(),
                          'returned_translation_m': fitted_t.tolist()}), flush=True)
        self.assertTrue(result['success'])
        # 0.01 mm is a numerical contract for noiseless inputs, not a sensor specification.
        self.assertLess(rms_mm, .01,
                        'Head/camera gauge must preserve full SE(3), including translation')

    def test_step1_constant_absolute_j6_target_converges_to_target(self):
        # Backup tests/test_joint_retry.py isolates the iterator from hardware.
        # Here the CURRENT API returns an absolute J6 target, not a residual.
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.robot = None
        cal.joint_offsets = {'right': {'wrist_yaw2': -2.}}
        cal.use_angle_based_fitting = True
        stages = []

        def measure(*args, **kwargs):
            stages.append(float(kwargs['current_offset_deg']))
            return {'optimal_offset': -1., 'angle_between_normals': 90.,
                    'r_A': 50., 'r_B': 50., 'center_dist': 0.}

        cal.perform_calibration_sweep_continuous = measure
        cal.save_calibration_comparison_plot = lambda *args, **kwargs: None
        result = cal.perform_joint_calibration('right', 'wrist_yaw2', current_offset_deg=-2.)
        print(json.dumps({'j6_staged_offsets_deg': stages,
                          'returned_offset_deg': result['recommended_joint_offset'],
                          'converged': result['converged']}), flush=True)
        self.assertAlmostEqual(result['recommended_joint_offset'], -1., places=10,
                               msg='Damp target minus current, never the absolute target itself')
        self.assertTrue(result['converged'])


if __name__ == '__main__':
    unittest.main()


class ReadyPoseBeforeEveryJointStage(unittest.TestCase):
    """Every joint stage in the Step 1 sequence must reposition the arm before it sweeps.

    `perform_joint_calibration` centres both of its sweeps on the arm's current pose, resetting
    only the joint being calibrated, so the stage before it decides where the sweep happens. The
    v1.2 J6 (wrist_yaw2) stage was the one that never moved: the axis-5 marker sweep ahead of it
    ends 40 deg away (MARKER_CONFIGS axis_5 runs 0 -> -40 deg), so the J5 arc was swept 40 deg low
    and ran off the bottom of the frame -- 22 of 30 deg captured on 2026-09-21 -- which biases the
    circle centre the J6 offset is solved from. Static check: the call graph here is a fixed
    sequence, and driving the whole of Step 1 would need a robot.
    """

    SEQUENCE = Path(__file__).resolve().parents[1] / "core/calibration/sequences/step1.py"
    LOOKBACK_LINES = 25

    def test_each_joint_calibration_is_preceded_by_a_move_to_its_own_ready_pose(self):
        lines = self.SEQUENCE.read_text(encoding="utf-8").split("\n")
        stages = [index for index, line in enumerate(lines)
                  if "perform_joint_calibration(" in line and "def " not in line]
        self.assertGreaterEqual(len(stages), 5, "Step 1 should still have every joint stage")

        for index in stages:
            mode = re.search(r'"([a-z0-9_]+)"', lines[index + 1])
            self.assertIsNotNone(mode, f"line {index + 2}: could not read the calibration mode")
            mode = mode.group(1)
            window = lines[max(0, index - self.LOOKBACK_LINES):index]
            moved = [re.search(r'perform_move_to_ready_pose\([^,]+,\s*"([a-z0-9_]+)"', line)
                     for line in window if "perform_move_to_ready_pose(" in line]
            moved = [match.group(1) for match in moved if match]
            self.assertIn(mode, moved,
                          f"step1.py line {index + 1}: the '{mode}' stage sweeps from wherever the "
                          f"previous stage left the arm; move to its ready pose first")


class BracketSweepsShareOnePosture(unittest.TestCase):
    """If the marker posture is re-taught part way through the bracket sweeps, every sweep done
    at the old posture is repeated at the new one before the bracket is solved (2026-09-21: axis
    4 stayed at the old posture, its axis ended 29 mm from the others, and the right bracket came
    out at y = -40.5 mm instead of about -54)."""

    OLD = [0.0] * 7
    NEW = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def fake_marker(self, reteach_on=None):
        from core.calibration.sequences.step1 import run_bracket_sweeps  # noqa: F401
        calls = []
        marker = SimpleNamespace(user_taught_ready_poses={})

        def sweep(arm_side, axis, initial_joint_pos=None, **kwargs):
            if axis == reteach_on and not marker.user_taught_ready_poses.get(arm_side):
                marker.user_taught_ready_poses[arm_side] = {"marker": list(self.NEW)}
            used = marker.user_taught_ready_poses.get(arm_side, {}).get("marker") or initial_joint_pos
            calls.append((axis, list(used)))
            return {"axis_opt": [0.0, 0.0, 1.0], "pose": list(used)}

        marker.perform_calibration_sweep = sweep
        return marker, calls

    def run_sweeps(self, marker):
        from core.calibration.sequences.step1 import run_bracket_sweeps
        return run_bracket_sweeps(marker, "right", list(self.OLD), log=lambda *_: None,
                                  emit_status=None, save_debug=False, pass_idx=2,
                                  is_stopped=lambda: False)

    def test_reteach_during_axis_6_repeats_axis_4_at_the_new_posture(self):
        marker, calls = self.fake_marker(reteach_on=6)

        sweeps = self.run_sweeps(marker)

        self.assertEqual([axis for axis, _ in calls], [4, 6, 5, 4])
        for axis in (4, 6, 5):
            self.assertEqual(sweeps[axis]["pose"], self.NEW, f"axis {axis} kept the old posture")

    def test_no_reteach_sweeps_each_axis_once(self):
        marker, calls = self.fake_marker(reteach_on=None)

        sweeps = self.run_sweeps(marker)

        self.assertEqual([axis for axis, _ in calls], [4, 6, 5])
        self.assertTrue(all(sweeps[a]["pose"] == self.OLD for a in (4, 6, 5)))


class ReteachAlwaysStartsFromTheReadyPose(unittest.TestCase):
    """Every re-teach prompt (marker lost) must first send the arm back to the ready pose and
    must pass the mode. Two visibility pre-checks asked with the arm still where the previous
    sweep ended, and one path passed no mode, so the taught posture was stored under None and
    the re-sweep ignored it."""

    FILES = ["core/calibration/MarkerCalibrator.py", "core/calibration/JointCalibrator.py",
             "core/calibration/CalibratorBase.py"]
    LOOKBACK = 8

    def test_each_prompt_passes_a_mode_and_follows_a_move_to_the_ready_pose(self):
        root = Path(__file__).resolve().parents[1]
        prompts = 0
        for rel in self.FILES:
            lines = (root / rel).read_text(encoding="utf-8").split("\n")
            for index, line in enumerate(lines):
                if "marker_problem_callback(" not in line or "def " in line or "hasattr" in line:
                    continue
                prompts += 1
                self.assertIn("mode=", line, f"{rel}:{index + 1} asks without a mode")
                before = lines[max(0, index - self.LOOKBACK):index]
                self.assertTrue(any("perform_move_to_ready_pose(" in b for b in before),
                                f"{rel}:{index + 1} asks for a re-teach before returning to the ready pose")
        self.assertGreaterEqual(prompts, 7)


class HeadSweepNeedsBothMarkers(unittest.TestCase):
    """Step 1.5 must not solve from one marker: on 2026-09-21 the left-only runs put about
    1.5 deg into the J0 common mode of both arms. A missing marker triggers ready pose +
    readjustment, and when it stays missing the step fails instead of continuing."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.dict(CONFIG_PATHS, txt_dir=folder.name,
                             result_dir=folder.name, plot_dir=folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)

        def no_dynamics():
            raise RuntimeError("no dynamics offline")

        self.calib = HeadCameraCalibrator()
        self.calib.robot = SimpleNamespace(model=lambda: SimpleNamespace(head_idx=[0, 1]),
                                           get_state=lambda: SimpleNamespace(position=np.zeros(20)),
                                           get_dynamics=no_dynamics, cancel_control=lambda: None)
        self.ready_moves = []
        self.calib.perform_move_to_ready_pose = lambda **k: self.ready_moves.append(k) or True
        self.calib.movej = lambda *a, **k: True
        self.prompts = []

    def prompt(self, answer):
        def callback(side, mode=None):
            self.prompts.append((side, mode))
            return answer
        return callback

    def visible(self, *sides):
        return lambda arm_side="auto", sampling_time=0.5: ((np.zeros(3), None) if arm_side in sides else (None, None))

    def test_marker_missing_at_centre_asks_for_readjustment_then_fails(self):
        self.calib._detect_marker_point = self.visible("left")
        self.calib.marker_problem_callback = self.prompt(False)
        with self.assertRaisesRegex(RuntimeError, "Right marker not visible"):
            self.calib.perform_head_sweep(step_delay=0, save_debug=False)
        self.assertEqual(self.prompts, [("right", "head_camera")])
        self.assertEqual(len(self.ready_moves), 1)

    def test_readjustment_is_retried_but_never_falls_back_to_one_marker(self):
        self.calib._detect_marker_point = self.visible("left")
        self.calib.marker_problem_callback = self.prompt(True)
        with self.assertRaisesRegex(RuntimeError, "needs both arm markers"):
            self.calib.perform_head_sweep(step_delay=0, save_debug=False, max_readjust=3)
        self.assertEqual(len(self.prompts), 3)
        # Only the first prompt needs the ready pose; after that the taught posture is kept.
        self.assertEqual(len(self.ready_moves), 1)

    def test_already_at_ready_pose_prompts_without_moving_again(self):
        self.calib._detect_marker_point = self.visible("left")
        self.calib.marker_problem_callback = self.prompt(False)
        with self.assertRaises(RuntimeError):
            self.calib.perform_head_sweep(step_delay=0, save_debug=False, at_ready_pose=True)
        self.assertEqual(self.prompts, [("right", "head_camera")])
        self.assertEqual(self.ready_moves, [])

    def test_marker_lost_during_sweep_asks_for_readjustment(self):
        import time as _time
        self.calib._detect_marker_point = self.visible("left", "right")
        ticks = iter(range(1, 10 ** 6))

        def transform(sampling_time=0, side=None, use_filter=False, q_encoder=None):
            if side != "left":
                return None
            pose = np.eye(4)
            pose[:3, 3] = [0.1, 0.0, 0.3 + 1e-6 * next(ticks)]
            return [pose.flatten().tolist()]

        self.calib.marker_st = SimpleNamespace(get_marker_transform=transform)
        self.calib.movej = lambda *a, **k: _time.sleep(0.6) or True
        self.calib.marker_problem_callback = self.prompt(False)
        with self.assertRaisesRegex(RuntimeError, r"Right marker lost during the head sweeps \(right 0 pts\)"):
            self.calib.perform_head_sweep(step_delay=0, save_debug=False, sweep_duration_s=0.3,
                                          at_ready_pose=True)
        self.assertEqual(self.prompts, [("right", "head_camera")])
        # The head moved during the sweep, so this time the robot does go back to the ready pose.
        self.assertEqual(len(self.ready_moves), 1)

    def test_one_readjustment_then_the_collected_points_are_used(self):
        """2026-09-22: every marker left the frame at the far end of the sweep (1-3 of 11 bins
        empty) and the prompts kept coming. One readjustment, then solve with what was seen."""
        import time as _time
        self.calib._detect_marker_point = self.visible("left", "right")
        ticks = iter(range(1, 10 ** 6))

        def transform(sampling_time=0, side=None, use_filter=False, q_encoder=None):
            pose = np.eye(4)
            pose[:3, 3] = [0.1 if side == "left" else -0.1, 0.0, 0.3 + 1e-6 * next(ticks)]
            return [pose.flatten().tolist()]

        self.calib.marker_st = SimpleNamespace(get_marker_transform=transform)
        self.calib.movej = lambda *a, **k: _time.sleep(0.4) or True
        self.calib.marker_problem_callback = self.prompt(True)
        solved = []
        self.calib._solve_head_camera = lambda phases, *a, **k: solved.append(phases) or {"success": True}
        # The encoder never moves here, so every sample lands in the centre bin: 10/11 bins empty.
        result = self.calib.perform_head_sweep(step_delay=0, save_debug=False, sweep_duration_s=0.2,
                                               at_ready_pose=True)
        self.assertEqual(result, {"success": True})
        self.assertEqual(self.prompts, [("right", "head_camera")])
        self.assertEqual(len(solved), 1)
        self.assertEqual({s["side"] for ph in solved[0] for s in ph["samples"]}, {"left", "right"})


class WristYaw2FromBracketSweeps(unittest.TestCase):
    """v1.2 J6 comes from the bracket's axis-6 / axis-5 sweeps (JointCalibrator.WRIST_YAW2_SOURCE).

    2026-09-22, at the J6 zero the user had checked by eye: the bracket sweeps read -0.07 / +0.30
    deg (right / left), the dedicated J6 sweep pair about -0.9 deg on both arms."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.dict(CONFIG_PATHS, txt_dir=folder.name, result_dir=folder.name, plot_dir=folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.jc = JointCalibrator.__new__(JointCalibrator)
        self.jc.robot = SimpleNamespace(model=lambda: SimpleNamespace(right_arm_idx=list(range(7)),
                                                                      left_arm_idx=list(range(7, 14))))

    @staticmethod
    def sweep(tag, n=30):
        qs = [np.full(20, float(tag)) + k * 1e-3 for k in range(n)]
        poses = [np.eye(4) * (tag + k) for k in range(n)]
        return {"captured_q_full": qs, "captured_poses": poses}

    def test_feeds_the_j6_sweep_as_a_and_the_j5_sweep_as_b(self):
        res_6, res_5 = self.sweep(6), self.sweep(5)
        seen = {}

        def compute(arm_side, mode, dataset_A, dataset_B, initial_joint_pos, **kwargs):
            seen.update(arm=arm_side, mode=mode, A=dataset_A, B=dataset_B, init=initial_joint_pos)
            return {"optimal_offset": 0.3, "r_A": 54.0, "r_B": 174.6, "estimate_failed": False}

        self.jc.compute_calibration_results = compute
        out = self.jc.wrist_yaw2_from_bracket_sweeps("left", res_6, res_5, log_callback=lambda *_: None)

        self.assertEqual((seen["arm"], seen["mode"]), ("left", "wrist_yaw2"))
        self.assertIs(seen["A"][0][0], res_6["captured_q_full"][0])
        self.assertIs(seen["A"][4][1], res_6["captured_poses"][4])
        self.assertIs(seen["B"][0][0], res_5["captured_q_full"][0])
        np.testing.assert_allclose(seen["init"], res_6["captured_q_full"][0][7:14])
        self.assertTrue(out["converged"])
        self.assertEqual(out["recommended_joint_offset"], 0.3)
        self.assertEqual(out["convergence_basis"], "bracket_sweeps")

    def test_a_failed_estimate_is_not_converged(self):
        self.jc.compute_calibration_results = lambda *a, **k: {
            "optimal_offset": 0.0, "r_A": 54.0, "r_B": 174.6, "estimate_failed": True}
        out = self.jc.wrist_yaw2_from_bracket_sweeps("right", self.sweep(6), self.sweep(5), log_callback=lambda *_: None)
        self.assertFalse(out["converged"])

    def test_too_few_frames_gives_none(self):
        self.jc.compute_calibration_results = lambda *a, **k: self.fail("should not compute")
        self.assertIsNone(self.jc.wrist_yaw2_from_bracket_sweeps(
            "right", self.sweep(6, n=5), self.sweep(5), log_callback=lambda *_: None))

    def test_step1_takes_j6_from_the_bracket_sweeps_without_a_j6_sweep(self):
        from unittest.mock import MagicMock
        from threading import Event
        from core.calibration.sequences.step1 import execute_step1_sequence
        from core.calibration.sequences.result import SequenceContext

        class NoWait(Event):
            def wait(self, timeout=None):
                return False

        joint, marker = MagicMock(), MagicMock()
        marker.get_robot_version.return_value = "1.2"
        marker.camera_config, marker.joint_offsets, joint.joint_offsets = {}, {}, {}
        marker.user_taught_ready_poses = {}
        joint.NOMINAL_BRACKET_TEMPLATES = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES
        joint.robot.model.return_value = SimpleNamespace(right_arm_idx=list(range(7)), left_arm_idx=list(range(7, 14)))
        joint.robot.get_state.return_value = SimpleNamespace(position=np.zeros(20))
        joint.perform_move_to_ready_pose.return_value = True
        marker.perform_move_to_ready_pose.return_value = True
        joint.perform_joint_calibration.side_effect = lambda arm, mode, **kw: {"converged": True, "recommended_joint_offset": 0.1}
        marker.perform_calibration_sweep.side_effect = lambda arm, axis, **kw: {"axis_opt": [0.0, 0.0, 1.0], "sweep": axis}
        marker.compute_unified_bracket_calibration.side_effect = lambda *a, **k: {
            "x_e": 0.0, "y_e": 54.0, "z_e": -49.0, "roll_e": 90.0, "pitch_e": 0.0, "yaw_e": 0.0}
        marker.generate_marker_plot.return_value = None
        joint.save_calibration_comparison_plot.return_value = None
        joint.wrist_yaw2_from_bracket_sweeps.side_effect = lambda arm, r6, r5, **kw: {
            "converged": True, "recommended_joint_offset": 0.25 if arm == "right" else -0.5}
        store = {arm: {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0} for arm in ("right", "left")}

        result = execute_step1_sequence(joint, marker, store, context=SequenceContext("step1", NoWait()))

        self.assertEqual(result.status, "completed", result.error)
        modes = [c.args[1] for c in joint.perform_joint_calibration.call_args_list]
        self.assertNotIn("wrist_yaw2", modes)
        calls = joint.wrist_yaw2_from_bracket_sweeps.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["right", "right", "left", "left"])   # every pass
        for c in calls:
            self.assertEqual((c.args[1]["sweep"], c.args[2]["sweep"]), (6, 5))
        self.assertEqual(store["right"]["joint6"], 0.25)
        self.assertEqual(store["left"]["joint6"], -0.5)
