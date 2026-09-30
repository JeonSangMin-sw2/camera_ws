"""Step 2 observability report (core/calibration/observability.py).

2026-09-29: after applying a Step 2 result with a 0.17 mm residual, the check pose was off
fore-aft (left hand ahead of the right). The 64 Step 2 samples span ~10 deg per joint, so some J2/J4/J6
combinations barely change the residual yet move the hands at the check pose by 1-2 mm per degree.
The report names those combinations. Offline: a linear fake model with one exact degeneracy.
"""
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from core.calibration import observability as ob

ROOT = Path(__file__).resolve().parents[1]
N_ARM, N_HEAD = 14, 2
MM_PER_DEG_PER_M_PER_RAD = 1000.0 * np.pi / 180.0


class FakeDynamics:
    """ee position (m) in the torso frame = HAND[link] @ q_full: RJ0 moves the right hand in x."""

    def __init__(self):
        self.q = None
        self.hand = {"ee_right": np.zeros((3, N_ARM)), "ee_left": np.zeros((3, N_ARM))}
        self.hand["ee_right"][0, 0] = 1.0
        self.hand["ee_left"][1, 7] = 0.5

    def make_state(self, links, names):
        return SimpleNamespace(link=links[1], set_q=lambda q: setattr(self, "q", np.asarray(q, dtype=float)))

    def compute_forward_kinematics(self, state):
        pass

    def compute_transformation(self, state, a, b):
        T = np.eye(4)
        T[:3, 3] = self.hand[state.link] @ self.q
        return T


class FakeOptimizer:
    """Marker pose error of sample i, side s = G[i, s] @ [arm offsets, head offsets]. Columns RJ0 and
    RJ1 are identical, so RJ0 - RJ1 is invisible to the residual; everything else is well determined."""

    def __init__(self, n_samples=12, rot_std_deg=0.3, pos_std_mm=0.17):
        self.G = np.random.default_rng(0).normal(size=(n_samples, 2, 6, N_ARM + N_HEAD))
        self.G[..., 1] = self.G[..., 0]
        self.use_head_kinematics = True
        self.optimize_camera = False
        self.lock_camera_head_axis_rotation = True
        self.ee_links = {"right": "ee_right", "left": "ee_left"}
        self.arm_idx = list(range(N_ARM))
        self.q_nominal = np.zeros(N_ARM)
        self.model = SimpleNamespace(robot_joint_names=[f"j{i}" for i in range(N_ARM)])
        self.dyn_model = FakeDynamics()

        def unweighted():
            raise AssertionError("the report must weight by 1/sigma, not by weights() (all ones when disabled)")
        self.noise_estimator = SimpleNamespace(rot_std_rad=np.radians(rot_std_deg), pos_std_m=pos_std_mm * 1e-3,
                                               weights=unweighted)

    def evaluate_sample(self, q_arm, q_head, side, q_arm_offset, q_head_offset, xi):
        e = self.G[int(q_arm[0]), 0 if side == "right" else 1] @ np.concatenate([q_arm_offset, q_head_offset])
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(e[:3]).as_matrix()
        T[:3, 3] = e[3:]
        return None, None, None, T


def report(opt, reference=True, offsets=None):
    n = opt.G.shape[0]
    q_arm = [np.full(N_ARM, float(i)) for i in range(n)]
    T_meas = [np.stack([np.eye(4), np.eye(4)])] * n
    arm = np.zeros(N_ARM) if offsets is None else offsets
    return ob.step2_weak_directions(opt, q_arm, [np.zeros(N_HEAD)] * n, T_meas, arm, np.zeros(N_HEAD), np.zeros(6),
                                    ["right", "left"], reference_q_arm=np.zeros(N_ARM) if reference else None)


class TestWeakDirections(unittest.TestCase):
    def test_the_invisible_combination_is_the_weakest(self):
        weakest = report(FakeOptimizer())["weak_directions"][0]
        self.assertLess(weakest["residual_rms_sigma_per_unit"], 1e-3)
        comp = weakest["components"]
        self.assertEqual(set(list(comp)[:2]), {"RJ0", "RJ1"})
        self.assertAlmostEqual(abs(comp["RJ0"]), np.sqrt(0.5), places=3)
        self.assertAlmostEqual(comp["RJ0"], -comp["RJ1"], places=3)
        self.assertGreater(report(FakeOptimizer())["weak_directions"][1]["residual_rms_sigma_per_unit"], 1.0)

    def test_hand_move_at_the_reference_pose_in_mm_per_degree(self):
        weakest = report(FakeOptimizer())["weak_directions"][0]
        move = weakest["reference_hand_move_mm_per_unit"]
        self.assertAlmostEqual(move["right"][0], weakest["components"]["RJ0"] * MM_PER_DEG_PER_M_PER_RAD, places=2)
        self.assertAlmostEqual(move["left"][1], 0.0, places=6)

    def test_flagged_and_warned(self):
        rep = report(FakeOptimizer())
        self.assertEqual(len(ob.weak_directions_needing_attention(rep)), 1)
        lines = ob.format_weak_directions(rep)
        self.assertTrue(lines[-1].startswith("[WARN] Step 2 poses barely constrain 1 offset combination"))
        self.assertRegex(lines[1], r"RJ[01] [+-]0\.71, RJ[01] [+-]0\.71")

    def test_no_reference_pose_no_hand_move_no_warning(self):
        rep = report(FakeOptimizer(), reference=False)
        self.assertNotIn("reference_hand_move_mm_per_unit", rep["weak_directions"][0])
        self.assertFalse(ob.weak_directions_needing_attention(rep))

    def test_residual_is_weighted_by_the_noise_estimate(self):
        offsets = np.full(N_ARM, np.radians(0.01))
        a = report(FakeOptimizer(), offsets=offsets)
        b = report(FakeOptimizer(rot_std_deg=0.6, pos_std_mm=0.34), offsets=offsets)
        self.assertGreater(a["residual_rms_sigma"], 0.0)
        self.assertAlmostEqual(b["residual_rms_sigma"], a["residual_rms_sigma"] / 2, places=6)
        self.assertAlmostEqual(b["weak_directions"][1]["residual_rms_sigma_per_unit"],
                               a["weak_directions"][1]["residual_rms_sigma_per_unit"] / 2, places=6)


class TestStep2Wiring(unittest.TestCase):
    def test_step2_reports_and_passes_the_check_pose_in_radians(self):
        source = (ROOT / "core" / "calibration" / "sequences" / "step2.py").read_text(encoding="utf-8")
        self.assertIn("step2_weak_directions(optimizer,", source)
        self.assertIn('result_dict["observability"] = observability', source)
        self.assertIn('"check_calib"', source)
        # get_ready_pose already returns radians (2026-09-29: an extra np.radians put the check pose at ~0).
        self.assertIsNone(re.search(r"np\.(radians|deg2rad)\(\s*np\.concatenate\(\[\s*self\.marker_calibrator\.get_ready_pose",
                                    source))


if __name__ == "__main__":
    unittest.main()
