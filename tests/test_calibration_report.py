"""scratch/calibration_report.py on synthetic sweeps with known geometry (no robot, no saved runs)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibration_report", ROOT / "scratch" / "calibration_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

J6_LOG = """
=== NEW ITERATION ===
[INFO] wrist_yaw2 J6 estimates: orientation +0.418°, position +0.854° (difference +0.436°); using the orientation method.
[INFO] right J6 (wrist_yaw2) from the bracket sweeps: +0.4181° (J6 sweep 356 frames, r 54.5 mm; J5 sweep 257 frames, r 175.8 mm; orientation +0.4181°, position +0.8539°, method orientation)

=== NEW ITERATION ===
[INFO] right J6 (wrist_yaw2) from the bracket sweeps: +0.3564° (J6 sweep 330 frames, r 54.5 mm; J5 sweep 269 frames, r 175.6 mm; orientation +0.3564°, position +0.9132°, method orientation)
"""
J5_LOG = """
=== NEW ITERATION ===
  * Circle Size Error (abs: r_A - r_B)     : 1.7060 mm
  * Circle Size Error (abs: r_A - r_B)     : 0.0512 mm
"""


def flange_z_in_marker(pitch_deg, roll_deg):
    p, r = np.radians([pitch_deg, roll_deg])
    return np.array([-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)])


def sweep_rows(axis, centre_mm, radius_mm, marker_axis, span_deg, frames=90, start_deg=0.0):
    """A marker carried round `axis` (camera frame); `marker_axis` is that axis as the marker sees it."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    spoke = np.cross(axis, [0.0, 1.0, 0.0] if abs(axis[0]) > 0.9 else [1.0, 0.0, 0.0])
    spoke /= np.linalg.norm(spoke)
    marker_to_cam, _ = Rotation.align_vectors([axis], [marker_axis])
    rows = []
    for angle in np.linspace(0.0, span_deg, frames):
        turn = Rotation.from_rotvec(axis * np.radians(start_deg + angle))
        T = np.eye(4)
        T[:3, :3] = (turn * marker_to_cam).as_matrix()
        T[:3, 3] = (np.asarray(centre_mm) + radius_mm * turn.apply(spoke)) / 1000.0
        rows.append([angle, *(T[:3, 3] * 1000.0), 0, 0, 0, 0, 0, 0, *T.ravel()])
    return rows


def write_sweep(path, passes):
    lines = []
    for index, rows in enumerate(passes):
        if index:
            lines.append("=== NEW ITERATION ===")
        lines.append("# Joint_Angle(deg), Cam_X(mm), ...")
        lines += [", ".join(f"{v:.6f}" for v in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run(folder, pitch_deg=0.40, roll_deg=91.0, r6=54.0):
    """Both arms, two passes, three mutually perpendicular wrist axes through one point."""
    centre = np.array([50.0, -20.0, 230.0])
    for arm in report.ARMS:
        specs = {6: ([0, 0, 1], r6, flange_z_in_marker(pitch_deg, roll_deg), 60.0),
                 5: ([0, 1, 0], 175.5, [1.0, 0.0, 0.0], 40.0),
                 4: ([1, 0, 0], 184.0, [0.0, 0.0, 1.0], 40.0)}
        for axis_number, (axis, radius, marker_axis, span) in specs.items():
            rows = sweep_rows(axis, centre, radius, marker_axis, span)
            write_sweep(folder / f"sweep_points_{arm}_marker_axis_{axis_number}.txt", [rows, rows])
        (folder / f"joint_calib_debug_{arm}_wrist_yaw2.txt").write_text(J6_LOG, encoding="utf-8")
        (folder / f"joint_calib_debug_{arm}_wrist_pitch.txt").write_text(J5_LOG, encoding="utf-8")


class TestSweepAnalysis(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        write_run(self.folder)

    def test_sweep_file_is_split_into_passes(self):
        passes = report.load_sweep(self.folder / "sweep_points_right_marker_axis_6.txt")
        self.assertEqual([len(p) for p in passes], [90, 90])
        self.assertEqual(passes[0].shape[1], 26)

    def test_radii_axis_tilt_and_marker_position_match_the_geometry(self):
        for arm in report.ARMS:
            first = report.analyze_sweeps(self.folder, arm)[0]
            self.assertEqual(first["frames6"], 90)
            self.assertAlmostEqual(first["r6"], 54.0, delta=0.02)
            self.assertAlmostEqual(first["r5"], 175.5, delta=0.05)
            self.assertAlmostEqual(first["r4"], 184.0, delta=0.05)
            self.assertLess(first["rmse6"], 0.01)
            self.assertAlmostEqual(first["pitch"], 0.40, delta=0.01)
            self.assertAlmostEqual(first["roll"], 91.0, delta=0.01)
            for key in ("a45", "a56", "a46"):
                self.assertAlmostEqual(first[key], 90.0, delta=0.05)
            # Middle of the 0..60 deg J6 sweep: 54 mm from the centre (50, -20, 230) mm, turned 30 deg from +y.
            middle = np.array([50.0 - 54.0 * np.sin(np.radians(30.0)), -20.0 + 54.0 * np.cos(np.radians(30.0))])
            self.assertAlmostEqual(first["off_x"], np.degrees(np.arctan2(middle[0], 230.0)), delta=0.1)
            self.assertAlmostEqual(first["off_y"], np.degrees(np.arctan2(middle[1], 230.0)), delta=0.1)

    def test_missing_sweeps_give_no_passes(self):
        (self.folder / "sweep_points_left_marker_axis_5.txt").unlink()
        self.assertEqual(report.analyze_sweeps(self.folder, "left"), [])
        self.assertEqual(len(report.analyze_sweeps(self.folder, "right")), 2)


class TestLogParsing(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_j6_estimates_of_both_methods_per_pass(self):
        path = self.folder / "joint_calib_debug_right_wrist_yaw2.txt"
        path.write_text(J6_LOG, encoding="utf-8")
        first, second = report.parse_j6_log(path)
        self.assertEqual((first["j6_ori"], first["j6_pos"], first["j6_used"]), (0.4181, 0.8539, 0.4181))
        self.assertAlmostEqual(first["j6_diff"], 0.4358)
        self.assertEqual((first["log_frames6"], first["log_r6"], first["log_frames5"], first["log_r5"]), (356, 54.5, 257, 175.8))
        self.assertAlmostEqual(second["j6_diff"], 0.5568)

    def test_size_error_is_the_last_one_of_each_pass(self):
        path = self.folder / "joint_calib_debug_right_wrist_pitch.txt"
        path.write_text(J5_LOG, encoding="utf-8")
        self.assertEqual(report.parse_size_error(path), [0.0512])

    def test_missing_log_gives_nothing(self):
        self.assertEqual(report.parse_j6_log(self.folder / "none.txt"), [])


class TestMarks(unittest.TestCase):
    def test_tool_warning_limits(self):
        self.assertTrue(report.past_tool_limit("right", "j6_diff", 0.557))
        self.assertFalse(report.past_tool_limit("right", "j6_diff", -0.436))
        self.assertTrue(report.past_tool_limit("left", "r6", 55.2))   # bracket y design value 54 mm, warning at 1 mm
        self.assertFalse(report.past_tool_limit("left", "r6", 54.5))
        self.assertFalse(report.past_tool_limit("left", "pitch", 9.0))  # no limit in the tool: never red
        self.assertFalse(report.past_tool_limit("left", "j6_diff", None))

    def test_distance_outside_the_previous_runs_in_spreads(self):
        self.assertEqual(report.spreads_outside(1.5, (1.0, 2.0)), 0.0)
        self.assertAlmostEqual(report.spreads_outside(2.5, (1.0, 2.0)), 0.5)
        self.assertAlmostEqual(report.spreads_outside(-2.0, (1.0, 2.0)), 3.0)
        self.assertEqual(report.spreads_outside(5.0, (1.0, 1.0)), 0.0)  # a single earlier value has no spread
        self.assertEqual(report.spreads_outside(5.0, None), 0.0)
        self.assertEqual(report.tint(0.0), "#ffffff")


class TestReportFiles(unittest.TestCase):
    def test_report_compares_a_run_with_an_earlier_one(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        base = Path(folder.name)
        earlier, run, out = base / "earlier", base / "run", base / "out"
        earlier.mkdir()
        run.mkdir()
        write_run(earlier, pitch_deg=0.00, roll_deg=90.2, r6=54.1)
        # Two different earlier passes so the previous range of the J6 radius has a width.
        rows = sweep_rows([0, 0, 1], [50.0, -20.0, 230.0], 54.3, flange_z_in_marker(0.1, 90.4), 60.0)
        first = report.load_sweep(earlier / "sweep_points_right_marker_axis_6.txt")[0].tolist()
        write_sweep(earlier / "sweep_points_right_marker_axis_6.txt", [first, rows])
        write_run(run, pitch_deg=0.60, roll_deg=92.0, r6=55.3)
        result = report.main([str(run), "--label", "synthetic", "--out", str(out), "--history", f"before={earlier}"])
        png, numbers = out / "calib_check_synthetic.png", out / "calib_check_synthetic.json"
        self.assertGreater(png.stat().st_size, 20000)
        saved = json.loads(numbers.read_text(encoding="utf-8"))
        right = saved["arms"]["right"][0]
        self.assertAlmostEqual(right["r6"], 55.3, delta=0.02)
        self.assertNotIn("_track", right)
        self.assertEqual(list(saved["history"]), ["before"])
        self.assertGreater(right["spreads_outside"]["r6"], 4.0)      # 55.3 against 54.1 ~ 54.3
        self.assertGreater(right["spreads_outside"]["pitch"], 4.0)   # 0.6 against 0.0 ~ 0.1
        self.assertTrue(any(f["key"] == "r6" and f["arm"] == "right" for f in result["flags"]))
        self.assertTrue(any(f["key"] == "j6_diff" and f["pass"] == 2 for f in result["flags"]))
        self.assertIsNone(saved["brackets"])

    def test_folder_without_sweeps_is_refused(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with self.assertRaises(SystemExit):
            report.main([folder.name, "--out", folder.name, "--history"])


if __name__ == "__main__":
    unittest.main()
