"""
Unit and dynamic empirical tests for the decoupled TurretController algorithm.
"""

import unittest
import numpy as np
from turret.turret_controller import TurretController, TurretInput, ControllerMode
from sensors.radar_a121 import A121DistanceResult
from core.coordinates import cartesian_to_spherical


class TestTurretController(unittest.TestCase):
    def setUp(self):
        self.controller = TurretController(muzzle_speed_m_s=100.0, fire_alignment_threshold_deg=2.5)

    def test_manual_test_mode(self):
        self.controller.set_manual_test_mode(pan_deg=25.0, tilt_deg=40.0, fire=True)
        dummy_radar = A121DistanceResult(num_distances=0)
        inputs = TurretInput(
            timestamp=0.0,
            current_pan_deg=0.0,
            current_tilt_deg=15.0,
            current_pan_vel_dps=0.0,
            current_tilt_vel_dps=0.0,
            ready_to_fire=True,
            radar_data=dummy_radar,
        )
        action = self.controller.update(inputs, dt=0.01)
        self.assertEqual(action.target_pan_deg, 25.0)
        self.assertEqual(action.target_tilt_deg, 40.0)
        self.assertTrue(action.fire)
        self.assertEqual(action.status_mode, "MANUAL_TEST")

    def test_search_mode_sweep(self):
        self.controller.set_autonomous_mode()
        dummy_radar = A121DistanceResult(num_distances=0)

        pan_history = []
        for step in range(100):
            inputs = TurretInput(
                timestamp=step * 0.01,
                current_pan_deg=self.controller.search_pan_target,
                current_tilt_deg=20.0,
                current_pan_vel_dps=0.0,
                current_tilt_vel_dps=0.0,
                ready_to_fire=True,
                radar_data=dummy_radar,
            )
            action = self.controller.update(inputs, dt=0.01)
            self.assertEqual(action.status_mode, "SEARCH_SCANNING")
            self.assertFalse(action.fire)
            pan_history.append(action.target_pan_deg)

        # Pan should smoothly change
        self.assertGreater(max(pan_history) - min(pan_history), 20.0)

    def test_track_and_fire_transition(self):
        self.controller.set_autonomous_mode()

        # Target at fixed world position [0.0, 5.0, 1.0] relative to turret
        target_world_pos = np.array([0.0, 5.0, 1.0])
        dist = float(np.linalg.norm(target_world_pos))
        az, el, _ = cartesian_to_spherical(target_world_pos)

        # Step 1: Turret currently pointing at [pan=5.0, tilt=20.0]
        curr_pan, curr_tilt = 5.0, 20.0
        off_h = az - curr_pan
        off_v = el - curr_tilt

        radar_det1 = A121DistanceResult(
            num_distances=1,
            distances=[dist],
            strengths=[25.0],
            estimated_angles_deg=[(off_h, off_v)],
        )

        inputs1 = TurretInput(
            timestamp=1.0,
            current_pan_deg=curr_pan,
            current_tilt_deg=curr_tilt,
            current_pan_vel_dps=0.0,
            current_tilt_vel_dps=0.0,
            ready_to_fire=True,
            radar_data=radar_det1,
        )
        action1 = self.controller.update(inputs1, dt=0.01)
        self.assertEqual(self.controller.mode, ControllerMode.TRACK)

        # Step 2: Turret now moves and aligns directly with target
        aligned_pan = action1.target_pan_deg
        aligned_tilt = action1.target_tilt_deg
        off_h2 = az - aligned_pan
        off_v2 = el - aligned_tilt

        radar_det2 = A121DistanceResult(
            num_distances=1,
            distances=[dist],
            strengths=[25.0],
            estimated_angles_deg=[(off_h2, off_v2)],
        )
        inputs2 = TurretInput(
            timestamp=1.05,
            current_pan_deg=aligned_pan,
            current_tilt_deg=aligned_tilt,
            current_pan_vel_dps=0.0,
            current_tilt_vel_dps=0.0,
            ready_to_fire=True,
            radar_data=radar_det2,
        )
        action2 = self.controller.update(inputs2, dt=0.01)
        self.assertTrue(action2.fire)
        self.assertEqual(action2.status_mode, "TRACK_ENGAGE")

    def test_multi_target_association_and_retargeting(self):
        """Verify that when 2 drones appear in radar, gating tracks target 1 then smoothly acquires target 2."""
        self.controller.set_autonomous_mode()

        # Target 1 at 3.0m straight, Target 2 at 7.0m
        radar_multi = A121DistanceResult(
            num_distances=2,
            distances=[3.0, 7.0],
            strengths=[30.0, 18.0],
            estimated_angles_deg=[(0.0, 10.0), (15.0, 20.0)],
        )

        inp1 = TurretInput(
            timestamp=0.1,
            current_pan_deg=0.0,
            current_tilt_deg=10.0,
            current_pan_vel_dps=0.0,
            current_tilt_vel_dps=0.0,
            ready_to_fire=True,
            radar_data=radar_multi,
        )
        act1 = self.controller.update(inp1, dt=0.01)
        self.assertEqual(self.controller.mode, ControllerMode.TRACK)
        # Should track target 1 (~3m)
        tracked_dist = np.linalg.norm(self.controller.tracked_pos - np.array([0.0, 0.0, 0.5]))
        self.assertAlmostEqual(tracked_dist, 3.0, delta=0.2)

        # Now target 1 is eliminated; only target 2 (7.0m) remains
        radar_only_t2 = A121DistanceResult(
            num_distances=1,
            distances=[7.0],
            strengths=[18.0],
            estimated_angles_deg=[(15.0, 20.0)],
        )
        inp2 = TurretInput(
            timestamp=0.15,
            current_pan_deg=act1.target_pan_deg,
            current_tilt_deg=act1.target_tilt_deg,
            current_pan_vel_dps=0.0,
            current_tilt_vel_dps=0.0,
            ready_to_fire=True,
            radar_data=radar_only_t2,
        )
        act2 = self.controller.update(inp2, dt=0.01)
        # Controller cleanly switches to target 2 without jumping or carrying over old velocity
        tracked_dist2 = np.linalg.norm(self.controller.tracked_pos - np.array([0.0, 0.0, 0.5]))
        self.assertAlmostEqual(tracked_dist2, 7.0, delta=0.3)
        self.assertAlmostEqual(np.linalg.norm(self.controller.tracked_vel), 0.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
