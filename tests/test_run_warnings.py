"""Run-line cautions judged from each stage's result (core/calibration/sequences/run_warnings.py)."""
import unittest

from core.calibration.sequences.run_warnings import stage_warnings, step1_5_warnings

BOUND_MM = 20.0


def keys(items):
    return [item["key"] for item in items]


class Step1WarningsTest(unittest.TestCase):
    def joint(self, **extra):
        value = {"mode": "wrist_pitch", "arm_side": "right", "pass_idx": 1, "converged": True,
                 "iterations": 3, "max_iterations": 8, "clamped": False}
        value.update(extra)
        return value

    def test_clean_run_has_none(self):
        self.assertEqual(stage_warnings("step1", {"joint_0": self.joint()}, {}), [])

    def test_all_iterations_used(self):
        items = stage_warnings("step1", {"joint_0": self.joint(iterations=8)}, {})
        self.assertEqual(keys(items), ["iter_cap"])
        self.assertEqual(items[0]["params"], {"arm": "right", "joint": "J5", "n": 8})

    def test_pass_one_result_repeated_in_pass_two_is_reported_once(self):
        value = self.joint(iterations=8)
        self.assertEqual(keys(stage_warnings("step1", {"joint_0": value, "joint_5": dict(value)}, {})), ["iter_cap"])

    def test_clamp_j6_methods_and_pass_change(self):
        completed = {"joint_0": self.joint(clamped=True),
                     "joint_1": self.joint(mode="wrist_yaw2", wrist_yaw2_orientation_deg=0.70,
                                           wrist_yaw2_position_deg=0.01)}
        partial = {"pass_change_left": {"joint6_deg": 0.103, "joint5_deg": 0.0, "joint3_deg": 0.0,
                                        "bracket_pos_mm": 0.185, "bracket_rot_deg": 0.021}}
        items = stage_warnings("step1", completed, partial)
        self.assertEqual(keys(items), ["offset_clamped", "j6_methods", "pass_change"])
        self.assertEqual(items[1]["params"], {"arm": "right", "diff": 0.69})
        self.assertEqual(items[2]["params"]["arm"], "left")

    def test_j6_methods_from_both_passes_is_one_line(self):
        # 2026-11-24 16:43: left J6 methods 0.90 deg apart in pass 1, 0.81 in pass 2.
        completed = {"joint_1": self.joint(arm_side="left", mode="wrist_yaw2", wrist_yaw2_orientation_deg=0.120,
                                           wrist_yaw2_position_deg=1.015),
                     "joint_6": self.joint(arm_side="left", mode="wrist_yaw2", pass_idx=2,
                                           wrist_yaw2_orientation_deg=0.060, wrist_yaw2_position_deg=0.874)}
        items = stage_warnings("step1", completed, {})
        self.assertEqual([(i["key"], i["params"]) for i in items], [("j6_methods", {"arm": "left", "diff": 0.89})])

    def test_j6_methods_within_limit(self):
        value = self.joint(mode="wrist_yaw2", wrist_yaw2_orientation_deg=0.40, wrist_yaw2_position_deg=0.0)
        self.assertEqual(stage_warnings("step1", {"joint_0": value}, {}), [])


class HeadCameraWarningsTest(unittest.TestCase):
    def test_marker_and_camera_at_their_bounds(self):
        # 2026-11-24 10:55 real v1.2 run (head not fixed firmly).
        head = {"success": True, "camera_at_range_end": ["x at the upper end (+1.0 mm)"],
                "marker_prior_shift_mm": {"left": [14.36, -20.0, 0.2], "right": [20.0, -13.7, 14.52]}}
        items = step1_5_warnings({"head_camera": head}, BOUND_MM)
        self.assertEqual(keys(items), ["camera_range", "marker_bound"])
        self.assertEqual(items[0]["params"], {"ends": "x +1.0 mm"})
        self.assertEqual(items[1]["params"], {"sides": "left, right", "bound": 20.0})

    def test_markers_inside_their_range(self):
        # 14:06, the head fixed again.
        head = {"success": True, "camera_at_range_end": [],
                "marker_prior_shift_mm": {"left": [0.9, -1.09, -5.02], "right": [0.54, 1.07, -5.45]}}
        self.assertEqual(step1_5_warnings({"head_camera": head}, BOUND_MM), [])

    def test_skipped_head_has_none(self):
        self.assertEqual(step1_5_warnings({"head_camera": {"skipped": True}}, BOUND_MM), [])

    def test_stage_uses_the_calibrator_bound(self):
        head = {"marker_prior_shift_mm": {"left": [0.0, -20.0, 0.0]}}
        self.assertEqual(keys(stage_warnings("step1_5", {"head_camera": head}, {})), ["marker_bound"])


class Step2WarningsTest(unittest.TestCase):
    def test_camera_range_and_head_tilt_gap(self):
        # 11:00 real run: x and z at the upper ends, tilt 0.94 (head-camera step) vs 0.34 (final).
        result = {"camera_mount_constraint": {"camera_at_range_end": ["x at the upper end (+1.0 mm)",
                                                                      "z at the upper end (+0.5 mm)"]},
                  "head_joint_offset_deg": [0.873, 0.341], "head_step1_5_offset_deg": [-2.992, 0.936]}
        items = stage_warnings("step2", {"optimization": result}, {})
        self.assertEqual(keys(items), ["camera_range", "head_tilt_gap"])
        self.assertEqual(items[0]["params"], {"ends": "x +1.0 mm, z +0.5 mm"})
        self.assertEqual(items[1]["params"], {"step1_5": 0.94, "step2": 0.34, "limit": 0.3})

    def test_agreeing_tilt_and_camera_inside(self):
        # Offline re-solve with the camera range widened: tilt 0.951 vs 0.936.
        result = {"camera_mount_constraint": {"camera_at_range_end": []},
                  "head_joint_offset_deg": [0.893, 0.951], "head_step1_5_offset_deg": [-2.992, 0.936]}
        self.assertEqual(stage_warnings("step2", {"optimization": result}, {}), [])

    def check_view(self, view):
        result = {"camera_mount_constraint": {"camera_at_range_end": []}}
        return stage_warnings("step2", {"optimization": result}, {"verification": {"check_pose_view": view}})

    def test_check_pose_gap_over_the_limit(self):
        # 2026-11-24 14:48: left hand about 4 mm low at the check pose by eye.
        meas = {"dx_mm": -0.7, "dy_mm": 0.3, "dz_mm": -4.04, "dist_mm": 4.11}
        items = self.check_view({"seen": True, "gap_mm": 4.11, "limit_mm": 2.0, "measured_symmetry": meas})
        self.assertEqual(keys(items), ["check_pose_gap"])
        self.assertEqual(items[0]["params"], {"gap": 4.1, "limit": 2.0, "dx": -0.7, "dy": 0.3, "dz": -4.0})

    def test_check_pose_within_the_limit(self):
        meas = {"dx_mm": 0.5, "dy_mm": 0.1, "dz_mm": 1.0, "dist_mm": 1.12}
        self.assertEqual(self.check_view({"seen": True, "gap_mm": 1.12, "limit_mm": 2.0, "measured_symmetry": meas}), [])

    def test_check_pose_unseen_or_skipped(self):
        self.assertEqual(keys(self.check_view({"seen": False, "reason": "markers"})), ["check_pose_unseen"])
        self.assertEqual(self.check_view(None), [])

    def test_other_sequences_have_none(self):
        self.assertEqual(stage_warnings("collect", {"optimization": {}}, {}), [])


class CalibrationLedTest(unittest.TestCase):
    """Robot LED around a run: off while it moves, red blinking after a failure or a caution."""
    def run_with(self, name, status, warned=False):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.calibration import CalibrationCore
        from core.calibration.sequences.result import SequenceResult
        calls = []
        robot = SimpleNamespace(set_led_color=lambda color, **kw: calls.append((color.r, kw)) or True)
        core = CalibrationCore()
        self.addCleanup(core.close)
        core.robot = robot
        self.addCleanup(setattr, core, "robot", None)     # runs before close: the fake has no session

        def execute(*_):
            core.run_warned = warned
            return SequenceResult(name).finish(status)

        with patch.object(core, "_execute", side_effect=execute):
            core.run(name)
        return calls

    def test_a_clean_run_turns_it_off_then_hands_it_back(self):
        calls = self.run_with("full", "completed")
        self.assertEqual([kw["duration"] for _, kw in calls], [7200.0, 0.01])
        self.assertTrue(all(r == 0 for r, _ in calls))

    def test_failure_or_caution_ends_in_red_blinking(self):
        for status, warned in (("failed", False), ("completed", True)):
            last_r, last_kw = self.run_with("full", status, warned)[-1]
            self.assertEqual(last_r, 255)
            self.assertTrue(last_kw["blinking"])

    def test_a_stop_by_the_operator_is_not_an_alert(self):
        self.assertEqual(self.run_with("full", "cancelled")[-1][1]["duration"], 0.01)

    def test_moves_outside_calibration_leave_the_led_alone(self):
        self.assertEqual(self.run_with("gripper_pose", "completed"), [])


class SettledByStep2Test(unittest.TestCase):
    """2026-11-24 (user): Step 1 cautions about J6 alone recur and Step 2 solves J6 again; log only after it."""
    J6_PASS = {"key": "pass_change", "params": {"arm": "right", "joint6_deg": 0.155, "joint5_deg": 0.0,
                                                "joint3_deg": 0.0, "bracket_pos_mm": 0.061, "bracket_rot_deg": 0.029}}
    J5_PASS = {"key": "pass_change", "params": {"arm": "left", "joint6_deg": 0.2, "joint5_deg": 0.12,
                                                "joint3_deg": 0.0, "bracket_pos_mm": 0.0, "bracket_rot_deg": 0.0}}
    METHODS = {"key": "j6_methods", "params": {"arm": "left", "diff": 0.9}}
    CAP = {"key": "iter_cap", "params": {"arm": "right", "joint": "J5", "n": 8}}

    @staticmethod
    def step2(solved):
        return {"optimization": {"camera_mount_constraint": {"camera_at_range_end": [], "joint6_solved": solved}}}

    def test_j6_only_items_settle_when_step2_solved_j6(self):
        from core.calibration.sequences.run_warnings import settled_by_step2
        kept, settled = settled_by_step2([self.J6_PASS, self.J5_PASS, self.METHODS, self.CAP], self.step2(True))
        self.assertEqual(kept, [self.J5_PASS, self.CAP])
        self.assertEqual(settled, [self.J6_PASS, self.METHODS])

    def test_nothing_settles_when_j6_was_held(self):
        from core.calibration.sequences.run_warnings import settled_by_step2
        items = [self.J6_PASS, self.METHODS]
        self.assertEqual(settled_by_step2(items, self.step2(False)), (items, []))
        self.assertEqual(settled_by_step2(items, {"optimization": {}}), (items, []))

    def test_full_auto_turns_step1_green_and_the_led_with_it(self):
        from types import SimpleNamespace
        from core.calibration.sequences.full import run_full
        from core.calibration.sequences.result import SequenceContext, SequenceResult
        children = {"step1": SequenceResult("step1", completed={"warnings": [self.J6_PASS]}),
                    "step1_5": SequenceResult("step1_5", completed={"head_camera": {"skipped": True}}),
                    "step2": SequenceResult("step2", completed=self.step2(True))}
        for child in children.values():
            child.finish("completed")
        logs = []
        core = SimpleNamespace(_execute=lambda name, options: children[name], log_msg=logs.append, run_warned=True)
        ctx = SequenceContext("full")
        run_full(core, ctx)
        self.assertNotIn("warnings", ctx.result.completed["step1"].completed)
        self.assertFalse(core.run_warned)
        self.assertTrue(any("settled by Step 2" in line for line in logs))


class WarningLogTest(unittest.TestCase):
    def test_the_stage_warn_lines_go_with_its_cautions(self):
        from core.calibration import CalibrationCore
        from core.calibration.sequences.result import SequenceContext
        core = CalibrationCore()
        self.addCleanup(core.close)
        ctx = SequenceContext("step2")
        ctx.warn_log = []
        ctx.result.completed["optimization"] = {"camera_mount_constraint": {"camera_at_range_end": ["x at the upper end"]}}
        core._contexts.append(ctx)
        core.log_msg("[INFO] not a warning")
        core.log_msg("[WARN] Check-pose view: both markers were not detected; the check pose is not verified.")
        core._attach_run_warnings("step2", ctx)
        core._contexts.pop()
        self.assertEqual(ctx.result.completed["warning_log"],
                         ["[WARN] Check-pose view: both markers were not detected; the check pose is not verified."])


if __name__ == "__main__":
    unittest.main()
