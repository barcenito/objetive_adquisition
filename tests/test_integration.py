"""
Integration and End-to-End Simulation Tests (Rule 2 Compliance).
"""

import unittest
from sim.simulation_env import TurretSimulationEnv
from targets.drone_target import TargetStatus


class TestIntegration(unittest.TestCase):
    def test_full_simulation_run_triple_threat(self):
        """Simulate 600 steps (6.0 seconds) with 3 drones at 100 Hz."""
        env = TurretSimulationEnv(num_drones=3, drone_speeds=[1.5, 2.8, 4.0], seed=42)

        detections_count = 0
        shots_count = 0
        hits_count = 0

        for step in range(600):
            snapshot = env.step()
            if snapshot.radar_data.num_distances > 0:
                detections_count += 1
            if snapshot.controller_action.fire:
                shots_count += 1
            if snapshot.hits_confirmed > hits_count:
                hits_count = snapshot.hits_confirmed

        # 1. Verification of radar acquisition
        self.assertGreater(detections_count, 10, "Radar A121 should detect drones multiple times during sweep")

        # 2. Verification of weapon discharge
        self.assertGreater(shots_count, 0, "Turret should engage and fire at targets")

        # 3. Verification of telemetry continuity
        self.assertEqual(len(env.history), 600)
        self.assertAlmostEqual(env.sim_time, 6.0, places=2)

    def test_manual_test_mode_in_simulation(self):
        """Verify that the user can test the turret movements using manual angles."""
        env = TurretSimulationEnv(num_drones=1, seed=42)
        # Command manual angles: Pan = 45 deg, Tilt = 30 deg
        env.controller.set_manual_test_mode(pan_deg=45.0, tilt_deg=30.0, fire=False)

        # Advance 100 steps (1.0 second) to allow servos to slew
        for _ in range(100):
            snapshot = env.step()

        t_state = snapshot.turret_state
        # Servos must reach commanded manual test angles within deadband
        self.assertAlmostEqual(t_state.pan_angle_deg, 45.0, delta=1.0)
        self.assertAlmostEqual(t_state.tilt_angle_deg, 30.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
