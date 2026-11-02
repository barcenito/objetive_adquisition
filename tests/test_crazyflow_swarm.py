"""
Unit and dynamic empirical tests for CrazyflowSwarmManager (JAX / Crazyflow integration).
"""

import unittest
import numpy as np
from targets.crazyflow_swarm import CrazyflowSwarmManager, CRAZYFLOW_AVAILABLE
from targets.drone_target import TargetStatus


class TestCrazyflowSwarm(unittest.TestCase):
    def test_crazyflow_availability(self):
        self.assertTrue(CRAZYFLOW_AVAILABLE, "Crazyflow and JAX must be available in the environment")

    def test_init_and_multi_drone_shapes(self):
        swarm = CrazyflowSwarmManager(num_drones=3, drone_speeds=[1.5, 3.0, 4.5], seed=123)
        self.assertEqual(swarm.num_drones, 3)
        self.assertEqual(swarm.positions.shape, (3, 3))
        self.assertEqual(swarm.velocities.shape, (3, 3))
        # Initial positions must be above ground
        self.assertTrue(np.all(swarm.positions[:, 2] > 0.5))

    def test_physics_stepping_in_jax(self):
        swarm = CrazyflowSwarmManager(num_drones=2, drone_speeds=[2.0, 3.5], seed=42)
        pos_init = np.copy(swarm.positions)

        # Step 50 steps (0.5s at 100 Hz)
        for _ in range(50):
            swarm.update()

        pos_after = np.copy(swarm.positions)
        # Positions must have evolved smoothly along 3D trajectories
        diffs = np.linalg.norm(pos_after - pos_init, axis=1)
        self.assertTrue(np.all(diffs > 0.1), f"Drones should have moved, diffs: {diffs}")
        # Drones must remain airborne (z > 0.4)
        self.assertTrue(np.all(swarm.positions[:, 2] > 0.4))
        # Drones must remain ALIVE
        self.assertEqual(swarm.statuses[0], TargetStatus.ALIVE)
        self.assertEqual(swarm.statuses[1], TargetStatus.ALIVE)

    def test_ballistic_hit_and_downed_transition_in_jax(self):
        swarm = CrazyflowSwarmManager(num_drones=1, drone_speeds=[2.0], seed=99)
        swarm.update()
        self.assertEqual(swarm.statuses[0], TargetStatus.ALIVE)

        # Inject downward ballistic impulse
        hit_impulse = np.array([0.0, 0.0, -0.15])
        swarm.apply_hit(0, hit_impulse)
        self.assertEqual(swarm.statuses[0], TargetStatus.HIT)

        # Advance simulation until drone reaches ground
        downed = False
        for _ in range(400):  # 4 seconds
            swarm.update()
            if swarm.statuses[0] == TargetStatus.DOWNED:
                downed = True
                break

        self.assertTrue(downed, "Hit drone must crash into the ground and transition to DOWNED in Crazyflow physics")
        self.assertEqual(swarm.positions[0, 2], 0.0)

    def test_drone_views_proxy_compatibility(self):
        swarm = CrazyflowSwarmManager(num_drones=2, seed=42)
        views = swarm.get_drone_views()
        self.assertEqual(len(views), 2)
        self.assertTrue(views[0].is_active_threat)
        self.assertEqual(views[0].target_id, 0)
        self.assertEqual(views[1].target_id, 1)
        # Hit through proxy
        views[0].apply_hit()
        self.assertEqual(views[0].status, TargetStatus.HIT)
        self.assertEqual(swarm.statuses[0], TargetStatus.HIT)


if __name__ == "__main__":
    unittest.main()
