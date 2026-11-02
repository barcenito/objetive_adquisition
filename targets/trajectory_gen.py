"""
3D Trajectory Generation and Obstacle/Boundary Avoidance for Aerial Targets.

Generates smooth parametric 3D paths for up to 3 drones flying at distinct speeds,
with soft potential-field steering to prevent ground impacts and boundary escapes.
"""

from typing import List, Tuple
import numpy as np


class TrajectoryGenerator:
    def __init__(
        self,
        flight_bounds_min: Tuple[float, float, float] = (-5.0, 1.8, 0.8),
        flight_bounds_max: Tuple[float, float, float] = (5.0, 8.5, 3.5),
        seed: int = 123,
    ):
        self.bounds_min = np.array(flight_bounds_min, dtype=np.float64)
        self.bounds_max = np.array(flight_bounds_max, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def generate_random_waypoints(self, num_waypoints: int = 8) -> np.ndarray:
        """Generate a sequence of 3D control waypoints within safe flight bounds."""
        wps = self.rng.uniform(self.bounds_min, self.bounds_max, size=(num_waypoints, 3))
        # Smoothly loop back to start for continuous flight
        wps[-1] = wps[0]
        return wps

    @staticmethod
    def evaluate_catmull_rom_spline(waypoints: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate closed Catmull-Rom spline at normalized phase t in [0, 1].
        Returns (position, velocity_direction).
        """
        n = len(waypoints) - 1
        scaled_t = (t % 1.0) * n
        i = int(np.floor(scaled_t))
        tau = scaled_t - i

        p0 = waypoints[(i - 1) % n]
        p1 = waypoints[i % n]
        p2 = waypoints[(i + 1) % n]
        p3 = waypoints[(i + 2) % n]

        # Catmull-Rom basis matrix
        t2 = tau * tau
        t3 = t2 * tau

        pos = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * tau +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

        vel = 0.5 * (
            (-p0 + p2) +
            2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tau +
            3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2
        ) * n

        vel_norm = np.linalg.norm(vel)
        dir_vec = vel / max(vel_norm, 1e-6)
        return pos, dir_vec

    def apply_boundary_steering(
        self,
        current_pos: np.ndarray,
        desired_vel: np.ndarray,
        margin: float = 0.5,
        repulsion_gain: float = 4.0,
    ) -> np.ndarray:
        """
        Apply soft potential field forces to prevent collision with ground (z=0) or boundary limits.
        """
        steered_vel = np.copy(desired_vel)

        # Ground and ceiling avoidance
        if current_pos[2] < self.bounds_min[2] + margin:
            penetration = (self.bounds_min[2] + margin) - current_pos[2]
            steered_vel[2] += repulsion_gain * max(0.0, penetration)
        elif current_pos[2] > self.bounds_max[2] - margin:
            penetration = current_pos[2] - (self.bounds_max[2] - margin)
            steered_vel[2] -= repulsion_gain * max(0.0, penetration)

        # Lateral boundaries (X)
        if current_pos[0] < self.bounds_min[0] + margin:
            penetration = (self.bounds_min[0] + margin) - current_pos[0]
            steered_vel[0] += repulsion_gain * max(0.0, penetration)
        elif current_pos[0] > self.bounds_max[0] - margin:
            penetration = current_pos[0] - (self.bounds_max[0] - margin)
            steered_vel[0] -= repulsion_gain * max(0.0, penetration)

        # Depth boundaries (Y)
        if current_pos[1] < self.bounds_min[1] + margin:
            penetration = (self.bounds_min[1] + margin) - current_pos[1]
            steered_vel[1] += repulsion_gain * max(0.0, penetration)
        elif current_pos[1] > self.bounds_max[1] - margin:
            penetration = current_pos[1] - (self.bounds_max[1] - margin)
            steered_vel[1] -= repulsion_gain * max(0.0, penetration)

        return steered_vel
