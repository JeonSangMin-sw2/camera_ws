"""Regression and contract tests for geometric 3D circle fitting algorithms.

Validates that BaseCalibrator.fit_circle_3d accurately recovers known 3D circle
centers, radii, and normal vectors from synthetic trajectories without hardware.
"""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from core.calibration.CalibratorBase import BaseCalibrator


class TestGeometryFittingContracts(unittest.TestCase):

    def test_fit_circle_3d_recovers_exact_synthetic_circle(self):
        """Verify that fit_circle_3d recovers exact center, radius, and normal
        for an arbitrary noiseless 3D circular arc."""
        true_center = np.array([120.0, -45.0, 310.0])  # in mm
        true_radius = 65.5  # in mm
        true_normal = np.array([0.26726, 0.53452, 0.80178])
        true_normal /= np.linalg.norm(true_normal)

        # Generate orthogonal basis on circle plane
        arb = np.array([1.0, 0.0, 0.0]) if abs(true_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(true_normal, arb)
        u /= np.linalg.norm(u)
        v = np.cross(true_normal, u)

        # Sample points along a 120 degree arc
        thetas = np.linspace(np.radians(-60.0), np.radians(60.0), 35)
        pts = np.array([
            true_center + true_radius * (np.cos(th) * u + np.sin(th) * v)
            for th in thetas
        ])

        c3d, R3d, r3d, rmse3d, _, _, _ = BaseCalibrator.fit_circle_3d(pts, robust=False)

        self.assertAlmostEqual(r3d, true_radius, places=3,
                               msg=f"Radius should match {true_radius} mm, got {r3d}")
        np.testing.assert_allclose(c3d, true_center, atol=1e-2,
                                   err_msg="Circle center must match true center within 0.01 mm")
        self.assertLess(rmse3d, 1e-3, "RMS error on noiseless arc must be < 0.001 mm")

        # Normal vector is the 3rd column of R3d
        normal_fitted = R3d[:, 2]
        norm_alignment = float(abs(np.dot(normal_fitted, true_normal)))
        self.assertAlmostEqual(norm_alignment, 1.0, places=4,
                               msg="Fitted plane normal must align with true normal")

    def test_fit_circle_3d_robust_rejects_single_outlier(self):
        """Verify that robust=True maintains reasonable radius and center
        even in the presence of an isolated tracking glitch/outlier."""
        true_center = np.array([50.0, 100.0, 200.0])
        true_radius = 80.0
        thetas = np.linspace(0.0, np.pi, 40)

        pts = np.array([
            true_center + np.array([true_radius * np.cos(th), true_radius * np.sin(th), 0.0])
            for th in thetas
        ])

        # Inject 1 outlier spike in the middle
        pts[20] += np.array([25.0, -30.0, 50.0])

        c3d, _, r3d, _, _, _, _ = BaseCalibrator.fit_circle_3d(pts, robust=True)

        self.assertLess(abs(r3d - true_radius), 5.0,
                        f"Robust fit should suppress outlier: radius error {abs(r3d - true_radius):.2f} mm < 5.0 mm")
        self.assertLess(np.linalg.norm(c3d - true_center), 5.0,
                        f"Robust fit center error {np.linalg.norm(c3d - true_center):.2f} mm < 5.0 mm")


if __name__ == "__main__":
    unittest.main()
