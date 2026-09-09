"""Reusable replacement for the historical scratch intrinsic-fit experiment.

Synthetic image points test the numerical fit, not real ChArUco detection.
No camera connection and no parameter-file writes.
"""
import unittest

import cv2
import numpy as np

from core.calibration.IntrinsicsCalibrator import IntrinsicsCalibrator


class IntrinsicFitTests(unittest.TestCase):
    def test_recovers_intrinsics_and_removes_one_corrupted_view(self):
        rng = np.random.default_rng(42)
        intrinsic = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
        points = np.zeros((40, 3), np.float32)
        points[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * .04
        images = []
        for _ in range(20):
            rotation = rng.uniform(-.4, .4, 3)
            translation = np.r_[rng.uniform(-.15, .05, 2), rng.uniform(.55, .8)]
            image, _ = cv2.projectPoints(points, rotation, translation, intrinsic, np.zeros(5))
            images.append(image + rng.normal(0., .01, image.shape).astype(np.float32))
        for corrupt in (False, True):
            with self.subTest(corrupted_view=corrupt):
                observations = [image.copy() for image in images]
                if corrupt:
                    observations[2] += rng.normal(0., 3., observations[2].shape).astype(np.float32)
                calibrator = IntrinsicsCalibrator()
                self.assertTrue(calibrator._calibrate_and_validate(
                    [points.copy() for _ in images], observations, (1280, 720),
                    [np.arange(len(points)) for _ in images]))
                self.assertLess(calibrator.rms_error, .03)
                np.testing.assert_allclose(calibrator.cameraMatrix, intrinsic, atol=1.)
                self.assertEqual(len(calibrator.all_obj_points), 20 - int(corrupt))


if __name__ == '__main__':
    unittest.main()
