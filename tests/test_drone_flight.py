"""
Unit and dynamic empirical tests for Drone Targets and 3D Trajectory Safety.
"""

import unittest
import numpy as np
from targets.trajectory_gen import TrajectoryGenerator
from targets.drone_target import DroneTarget, TargetStatus


class TestDroneFlight(unittest.TestCase):
    def setUp(self):
        self.traj_gen = TrajectoryGenerator(seed=777)

    def test_drone_no_ground_crash_during_nominal_flight(self):
        # Create 3 drones with distinct speeds
        speeds = [1.5, 3.0, 4.5]
        drones = [
            DroneTarget(target_id=i, nominal_speed_m_s=speeds[i], trajectory_gen=self.traj_gen)
            for i in range(3)
        ]

        # Simulate 1000 steps at 100 Hz (10 seconds of continuous flight)
        dt = 0.01
        for step in range(1000):
            for drone in drones:
                drone.update(dt)
                # Drones must remain ALIVE
                self.assertEqual(drone.status, TargetStatus.ALIVE)
                # Drones must NEVER collide with the ground during nominal flight
                self.assertGreater(drone.position[2], 0.4, f"Drone {drone.target_id} altitude fell to {drone.position[2]}")

    def test_drones_have_different_speeds(self):
        speeds = [1.5, 3.0, 4.5]
        drones = [
            DroneTarget(target_id=i, nominal_speed_m_s=speeds[i], trajectory_gen=self.traj_gen)
            for i in range(3)
        ]

        # Advance 200 steps and measure average speed
        dt = 0.01
        measured_speeds = []
        for drone in drones:
            speeds_sample = []
            for _ in range(200):
                drone.update(dt)
                speeds_sample.append(np.linalg.norm(drone.velocity))
            measured_speeds.append(float(np.mean(speeds_sample)))

        # Verify that speeds are strictly ascending matching configuration
        self.assertLess(measured_speeds[0], measured_speeds[1])
        self.assertLess(measured_speeds[1], measured_speeds[2])
        self.assertAlmostEqual(measured_speeds[0], 1.5, delta=0.5)
        self.assertAlmostEqual(measured_speeds[1], 3.0, delta=0.7)
        self.assertAlmostEqual(measured_speeds[2], 4.5, delta=0.9)

    def test_hit_reaction_and_downed_transition(self):
        drone = DroneTarget(target_id=0, nominal_speed_m_s=2.5, trajectory_gen=self.traj_gen)
        # Advance 100 steps
        for _ in range(100):
            drone.update(0.01)
        self.assertEqual(drone.status, TargetStatus.ALIVE)

        initial_z = drone.position[2]

        # Apply hit
        bullet_impulse = np.array([0.0, 0.0, -0.05])
        drone.apply_hit(bullet_impulse)
        self.assertEqual(drone.status, TargetStatus.HIT)

        # Simulate descent
        downed = False
        for _ in range(500):  # 5 seconds max
            drone.update(0.01)
            if drone.status == TargetStatus.DOWNED:
                downed = True
                break

        self.assertTrue(downed, "Drone should have crashed and transitioned to DOWNED")
        self.assertEqual(drone.position[2], 0.0)
        self.assertFalse(drone.is_active_threat)


if __name__ == "__main__":
    unittest.main()
