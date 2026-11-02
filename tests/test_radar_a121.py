"""
Unit and physical calibration tests for Acconeer A121 60 GHz PCR radar model.
"""

import unittest
import numpy as np
from sensors.radar_a121 import AcconeerA121Radar


class TestRadarA121(unittest.TestCase):
    def setUp(self):
        self.radar = AcconeerA121Radar(
            fov_horizontal_deg=53.0,
            fov_vertical_deg=65.0,
            min_range_m=0.20,
            max_range_m=10.0,
            range_resolution_m=0.03,
            range_noise_std_m=0.015,
            cfar_threshold_db=8.0,
            seed=42,
        )
        self.turret_pos = np.array([0.0, 0.0, 1.0])

    def test_boresight_detection(self):
        # Target directly in front at 4.0 m
        target = np.array([0.0, 4.0, 1.0])
        result = self.radar.measure(
            self.turret_pos, turret_pan_deg=0.0, turret_tilt_deg=0.0,
            target_positions=[target], apply_noise=False
        )
        self.assertEqual(result.num_distances, 1)
        self.assertAlmostEqual(result.distances[0], 4.0, places=3)
        self.assertGreater(result.strengths[0], self.radar.cfar_threshold_db)
        self.assertFalse(result.near_start_edge_status)

    def test_fov_cutoff(self):
        # Half-Power Beamwidth horizontal is 53 deg (+/- 26.5 deg).
        # Target at 15 deg azimuth (inside FoV)
        dist = 4.0
        target_inside = np.array([dist * np.sin(np.radians(15.0)), dist * np.cos(np.radians(15.0)), 1.0])
        res_inside = self.radar.measure(
            self.turret_pos, 0.0, 0.0, [target_inside], apply_noise=False
        )
        self.assertEqual(res_inside.num_distances, 1)

        # Target at 45 deg azimuth (well outside 53 deg FoV)
        target_outside = np.array([dist * np.sin(np.radians(45.0)), dist * np.cos(np.radians(45.0)), 1.0])
        res_outside = self.radar.measure(
            self.turret_pos, 0.0, 0.0, [target_outside], apply_noise=False
        )
        self.assertEqual(res_outside.num_distances, 0)

    def test_range_accuracy_and_noise(self):
        # 100 measurements with noise to verify mean and standard deviation
        target = np.array([0.0, 5.0, 1.0])
        measured_dists = []
        for _ in range(100):
            res = self.radar.measure(
                self.turret_pos, 0.0, 0.0, [target], apply_noise=True
            )
            self.assertEqual(res.num_distances, 1)
            measured_dists.append(res.distances[0])

        mean_d = float(np.mean(measured_dists))
        std_d = float(np.std(measured_dists))

        # Mean distance should be 5.0 within 0.01m
        self.assertAlmostEqual(mean_d, 5.0, delta=0.01)
        # Std should match ~0.015m (within tolerance [0.010, 0.022])
        self.assertGreater(std_d, 0.008)
        self.assertLess(std_d, 0.025)

    def test_multi_target_discrimination_and_resolution(self):
        # Target 1 at 3.0 m, Target 2 at 6.0 m (well separated > 0.03 m)
        t1 = np.array([0.0, 3.0, 1.0])
        t2 = np.array([0.0, 6.0, 1.0])
        res = self.radar.measure(
            self.turret_pos, 0.0, 0.0, [t1, t2], apply_noise=False
        )
        self.assertEqual(res.num_distances, 2)
        self.assertAlmostEqual(res.distances[0], 3.0, places=2)
        self.assertAlmostEqual(res.distances[1], 6.0, places=2)
        # Closer target has higher echo strength (due to 1/R^4 range loss)
        self.assertGreater(res.strengths[0], res.strengths[1])

    def test_near_edge_status(self):
        # Target too close (< 0.20 m)
        t_near = np.array([0.0, 0.10, 1.0])
        res = self.radar.measure(
            self.turret_pos, 0.0, 0.0, [t_near], apply_noise=False
        )
        self.assertTrue(res.near_start_edge_status)
        self.assertEqual(res.num_distances, 0)

    def test_radar_rotates_solidary_with_turret(self):
        """Verify that the radar sensor is coaxially mounted on the turret and rotates 1:1 with pan and tilt."""
        target_north = np.array([0.0, 4.0, 1.0])  # Straight North (+Y)
        target_east = np.array([4.0, 0.0, 1.0])   # Straight East (+X, Pan = +90 deg)
        target_zenith = np.array([0.0, 0.0, 5.0]) # Overhead (+Z, Tilt = +90 deg, dist = 4.0m from [0,0,1])

        # 1. Turret pointing North (Pan=0, Tilt=0)
        res_north = self.radar.measure(self.turret_pos, 0.0, 0.0, [target_north, target_east], apply_noise=False)
        self.assertEqual(res_north.num_distances, 1)
        self.assertAlmostEqual(res_north.distances[0], 4.0, places=2)
        # Target North is dead center on boresight
        off_h, off_v = res_north.estimated_angles_deg[0]
        self.assertAlmostEqual(off_h, 0.0, delta=0.5)
        self.assertAlmostEqual(off_v, 0.0, delta=0.5)

        # 2. Turret rotates +90 deg Pan to face East (Pan=90, Tilt=0)
        res_east = self.radar.measure(self.turret_pos, 90.0, 0.0, [target_north, target_east], apply_noise=False)
        self.assertEqual(res_east.num_distances, 1)
        self.assertAlmostEqual(res_east.distances[0], 4.0, places=2)
        # Target East is now dead center on boresight, Target North is completely out of FoV
        off_h_e, off_v_e = res_east.estimated_angles_deg[0]
        self.assertAlmostEqual(off_h_e, 0.0, delta=0.5)
        self.assertAlmostEqual(off_v_e, 0.0, delta=0.5)

        # 3. Turret tilts up to face Zenith (Pan=0, Tilt=90)
        res_zenith = self.radar.measure(self.turret_pos, 0.0, 90.0, [target_zenith, target_north], apply_noise=False)
        self.assertEqual(res_zenith.num_distances, 1)
        self.assertAlmostEqual(res_zenith.distances[0], 4.0, places=2)
        off_h_z, off_v_z = res_zenith.estimated_angles_deg[0]
        self.assertAlmostEqual(off_h_z, 0.0, delta=0.5)
        self.assertAlmostEqual(off_v_z, 0.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
