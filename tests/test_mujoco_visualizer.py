"""
Unit tests for MuJoCo visualizer and anti-crash mouse wheel scroll patch.
"""

import unittest
from sim.simulation_env import TurretSimulationEnv
from visualization.mujoco_visualizer import patch_mujoco_viewer_scroll_bug, MujocoVisualizer, MUJOCO_AVAILABLE


class TestMujocoVisualizer(unittest.TestCase):
    def test_patch_scroll_bug_applied(self):
        result = patch_mujoco_viewer_scroll_bug()
        self.assertTrue(result, "Mouse scroll monkey-patch should apply successfully")

    def test_mujoco_visualizer_offscreen_init(self):
        if not MUJOCO_AVAILABLE:
            self.skipTest("MuJoCo not installed in this environment")

        env = TurretSimulationEnv(num_drones=2, seed=42)
        vis = MujocoVisualizer(env, width=320, height=240)
        self.assertIsNotNone(vis.renderer, "Offscreen MuJoCo renderer should be initialized")


if __name__ == "__main__":
    unittest.main()
