"""
Aerial Drone Target Model.

Represents an individual flying drone target with:
- Unique ID and speed configuration
- Closed Catmull-Rom spline path generated at startup
- Soft boundary and terrain avoidance
- Radar Cross Section (RCS) and physical collision bounding sphere
- Dynamic damage transition upon projectile hit (loss of control, spinning descent).
"""

from enum import Enum
from typing import Optional
import numpy as np
from targets.trajectory_gen import TrajectoryGenerator


class TargetStatus(Enum):
    ALIVE = "alive"       # Normal flight along generated 3D path
    HIT = "hit"           # Struck by projectile; tumbling/spiral descent
    DOWNED = "downed"     # Impacted ground; destroyed/neutralized


class DroneTarget:
    def __init__(
        self,
        target_id: int,
        nominal_speed_m_s: float,
        trajectory_gen: TrajectoryGenerator,
        rcs_m2: float = 0.03,
        bounding_radius_m: float = 0.18,
        num_waypoints: int = 8,
    ):
        self.target_id = target_id
        self.speed = float(nominal_speed_m_s)
        self.traj_gen = trajectory_gen
        self.rcs = float(rcs_m2)
        self.radius = float(bounding_radius_m)
        self.status = TargetStatus.ALIVE

        # Generate initial trajectory waypoints
        self.waypoints = self.traj_gen.generate_random_waypoints(num_waypoints)

        # Calculate total approx perimeter length of path for constant-speed pacing
        dists = [np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) for i in range(len(self.waypoints)-1)]
        self.path_length = max(sum(dists), 10.0)
        self.lap_duration_s = self.path_length / max(self.speed, 0.5)

        # Start at random phase along the path
        self.phase = float(self.traj_gen.rng.uniform(0.0, 1.0))
        initial_pos, initial_dir = self.traj_gen.evaluate_catmull_rom_spline(self.waypoints, self.phase)

        self.position = np.array(initial_pos, dtype=np.float64)
        self.velocity = np.array(initial_dir * self.speed, dtype=np.float64)

        # Hit / Damage dynamics
        self.damage_angular_vel = np.zeros(3)
        self.time_since_hit = 0.0

    def apply_hit(self, impact_impulse: Optional[np.ndarray] = None) -> None:
        """
        Trigger damage reaction when struck by a projectile.
        Alters path drastically: destabilizes attitude, cuts thrust, initiates crash descent.
        """
        if self.status == TargetStatus.DOWNED:
            return

        self.status = TargetStatus.HIT
        self.time_since_hit = 0.0

        if impact_impulse is not None:
            # Transfer kinetic impulse from bullet to drone (mass ~0.035 kg Crazyflie)
            drone_mass = 0.035
            self.velocity += impact_impulse / drone_mass
        else:
            # Default chaotic perturbation
            rand_kick = self.traj_gen.rng.normal(0.0, 2.0, size=3)
            rand_kick[2] = -1.5  # Downward impulse
            self.velocity += rand_kick

    def update(self, dt: float) -> None:
        """Advance drone physics by dt seconds."""
        if dt <= 0.0:
            return

        if self.status == TargetStatus.ALIVE:
            # Advance phase along spline path
            self.phase = (self.phase + (self.speed * dt) / self.path_length) % 1.0
            spline_pos, spline_dir = self.traj_gen.evaluate_catmull_rom_spline(self.waypoints, self.phase)

            desired_vel = spline_dir * self.speed
            # Apply terrain and boundary safety steering
            effective_vel = self.traj_gen.apply_boundary_steering(self.position, desired_vel)

            # 1st-order smoothing towards effective velocity
            accel_gain = 5.0
            self.velocity += (effective_vel - self.velocity) * min(1.0, accel_gain * dt)
            self.position += self.velocity * dt

        elif self.status == TargetStatus.HIT:
            # Damaged flight dynamics: gravity + aerodynamic drag + descending spiral
            self.time_since_hit += dt
            gravity = np.array([0.0, 0.0, -9.81])
            drag = -0.35 * self.velocity * np.linalg.norm(self.velocity)

            # Erratic corkscrew motion as broken rotors lose differential thrust
            spiral_freq = 15.0
            spiral_force = np.array([
                np.sin(spiral_freq * self.time_since_hit) * 3.0,
                np.cos(spiral_freq * self.time_since_hit) * 3.0,
                -2.0
            ])

            net_accel = gravity + drag + spiral_force
            self.velocity += net_accel * dt
            self.position += self.velocity * dt

            # Check ground impact
            if self.position[2] <= 0.05:
                self.position[2] = 0.0
                self.velocity = np.zeros(3)
                self.status = TargetStatus.DOWNED

        elif self.status == TargetStatus.DOWNED:
            # Stationary on the ground
            self.velocity = np.zeros(3)
            self.position[2] = 0.0

    @property
    def is_active_threat(self) -> bool:
        """Returns True if the drone is still flying (ALIVE or descending HIT)."""
        return self.status != TargetStatus.DOWNED
