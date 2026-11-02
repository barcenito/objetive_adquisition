"""
Unit and dynamic empirical tests for Ballistic Interception and Hit Detection.
"""

import unittest
import numpy as np
from ballistics.projectile_system import ProjectileManager, compute_ballistic_lead
from targets.trajectory_gen import TrajectoryGenerator
from targets.drone_target import DroneTarget, TargetStatus


class TestBallistics(unittest.TestCase):
    def setUp(self):
        self.turret_pos = np.array([0.0, 0.0, 0.5])
        self.muzzle_speed = 100.0

    def test_lead_stationary_gravity_drop(self):
        # Target at 6m forward, 2m high
        target_pos = np.array([0.0, 6.0, 2.0])
        target_vel = np.array([0.0, 0.0, 0.0])

        pan, tilt, tof = compute_ballistic_lead(
            self.turret_pos, target_pos, target_vel, muzzle_speed=self.muzzle_speed
        )

        # Pan should remain 0 deg
        self.assertAlmostEqual(pan, 0.0, places=3)

        # Direct geometric angle
        diff = target_pos - self.turret_pos
        direct_tilt = np.degrees(np.arctan2(diff[2], diff[1]))

        # Tilt with gravity drop MUST be higher than direct line-of-sight
        self.assertGreater(tilt, direct_tilt)
        # Verify gravity drop angle matches approx 0.5 * g * tof^2
        expected_drop = 0.5 * 9.81 * (tof ** 2)
        expected_tilt = np.degrees(np.arctan2(diff[2] + expected_drop, diff[1]))
        self.assertAlmostEqual(tilt, expected_tilt, delta=0.2)

    def test_lead_moving_target_deflection(self):
        # Target at 5m forward, moving left-to-right along +X at 4.0 m/s
        target_pos = np.array([0.0, 5.0, 1.5])
        target_vel = np.array([4.0, 0.0, 0.0])

        pan, tilt, tof = compute_ballistic_lead(
            self.turret_pos, target_pos, target_vel, muzzle_speed=self.muzzle_speed
        )

        # Must aim to the right (+X, so pan > 0)
        self.assertGreater(pan, 0.0)
        # Expected X offset approx vx * tof
        expected_x = target_vel[0] * tof
        expected_pan = np.degrees(np.arctan2(expected_x, 5.0))
        self.assertAlmostEqual(pan, expected_pan, delta=0.5)

    def test_projectile_hit_detection(self):
        manager = ProjectileManager(muzzle_speed_m_s=self.muzzle_speed)
        traj_gen = TrajectoryGenerator(seed=42)
        target = DroneTarget(target_id=0, nominal_speed_m_s=2.0, trajectory_gen=traj_gen)

        # Position target directly at known location
        target.position = np.array([0.0, 5.0, 1.5])
        target.velocity = np.array([2.5, 0.0, 0.0])

        # Compute exact lead
        pan, tilt, tof = compute_ballistic_lead(
            self.turret_pos, target.position, target.velocity, muzzle_speed=self.muzzle_speed
        )

        # Compute aim direction vector
        pan_rad = np.radians(pan)
        tilt_rad = np.radians(tilt)
        aim_dir = np.array([
            np.sin(pan_rad) * np.cos(tilt_rad),
            np.cos(pan_rad) * np.cos(tilt_rad),
            np.sin(tilt_rad)
        ])

        # Spawn projectile
        manager.spawn_projectile(self.turret_pos, aim_dir)
        self.assertEqual(len(manager.projectiles), 1)

        # Step simulation until hit
        dt = 0.005  # 5ms step
        hit_occurred = False

        for _ in range(100):  # 0.5s max
            target.update(dt)
            hits = manager.update(dt, [target])
            if hits:
                hit_occurred = True
                self.assertEqual(target.status, TargetStatus.HIT)
                break

        self.assertTrue(hit_occurred, "Ballistic projectile fired at lead angle should hit the moving target!")


if __name__ == "__main__":
    unittest.main()
