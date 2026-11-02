"""
Unit tests for core coordinate transformations and antenna offsets.
"""

import unittest
import numpy as np
from core.coordinates import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    get_turret_basis,
    compute_antenna_angular_offset,
)


class TestCoordinates(unittest.TestCase):
    def test_spherical_to_cartesian_cardinal_directions(self):
        # 0 deg Pan, 0 deg Tilt -> Points along +Y
        v_fwd = spherical_to_cartesian(0.0, 0.0, 5.0)
        np.testing.assert_allclose(v_fwd, [0.0, 5.0, 0.0], atol=1e-6)

        # 90 deg Pan, 0 deg Tilt -> Points along +X (Right)
        v_right = spherical_to_cartesian(90.0, 0.0, 5.0)
        np.testing.assert_allclose(v_right, [5.0, 0.0, 0.0], atol=1e-6)

        # 0 deg Pan, 90 deg Tilt -> Points along +Z (Up)
        v_up = spherical_to_cartesian(0.0, 90.0, 5.0)
        np.testing.assert_allclose(v_up, [0.0, 0.0, 5.0], atol=1e-6)

    def test_cartesian_to_spherical_roundtrip(self):
        test_points = [
            np.array([3.0, 4.0, 2.0]),
            np.array([-2.5, 6.0, 1.5]),
            np.array([0.0, 5.0, 0.0]),
            np.array([4.0, -1.0, 3.0]),
        ]
        for p in test_points:
            az, el, dist = cartesian_to_spherical(p)
            reconstructed = spherical_to_cartesian(az, el, dist)
            np.testing.assert_allclose(reconstructed, p, atol=1e-5)

    def test_turret_basis_orthonormality(self):
        for pan in [-90, -45, 0, 30, 90, 150]:
            for tilt in [-10, 0, 25, 45, 80]:
                fwd, right, up = get_turret_basis(pan, tilt)
                # Norms equal 1
                self.assertAlmostEqual(np.linalg.norm(fwd), 1.0, places=6)
                self.assertAlmostEqual(np.linalg.norm(right), 1.0, places=6)
                self.assertAlmostEqual(np.linalg.norm(up), 1.0, places=6)
                # Mutual orthogonality
                self.assertAlmostEqual(np.dot(fwd, right), 0.0, places=6)
                self.assertAlmostEqual(np.dot(fwd, up), 0.0, places=6)
                self.assertAlmostEqual(np.dot(right, up), 0.0, places=6)

    def test_antenna_angular_offset(self):
        turret_pos = np.array([0.0, 0.0, 1.0])
        pan = 0.0
        tilt = 0.0

        # Target directly in front (along boresight) at distance 5m
        target_center = np.array([0.0, 5.0, 1.0])
        off_h, off_v, dist = compute_antenna_angular_offset(turret_pos, pan, tilt, target_center)
        self.assertAlmostEqual(off_h, 0.0, places=4)
        self.assertAlmostEqual(off_v, 0.0, places=4)
        self.assertAlmostEqual(dist, 5.0, places=4)

        # Target 10 deg to the right
        dist = 5.0
        target_right = np.array([dist * np.sin(np.radians(10.0)), dist * np.cos(np.radians(10.0)), 1.0])
        off_h, off_v, d = compute_antenna_angular_offset(turret_pos, pan, tilt, target_right)
        self.assertAlmostEqual(off_h, 10.0, places=2)
        self.assertAlmostEqual(off_v, 0.0, places=2)
        self.assertAlmostEqual(d, dist, places=4)


if __name__ == "__main__":
    unittest.main()
