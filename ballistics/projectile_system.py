"""
Ballistic Projectiles and Hit Detection System.

Simulates physical projectiles with:
- High muzzle velocity (e.g. 100 m/s)
- Gravity drop and aerodynamic drag
- Continuous swept-sphere / line-segment collision checking against targets
- Iterative ballistic lead (deflection angle) solver for target interception.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Projectile:
    position: np.ndarray
    velocity: np.ndarray
    mass_kg: float = 0.015         # 15 gram projectile
    drag_coeff: float = 0.0004     # 0.5 * rho * Cd * A / m
    alive: bool = True
    time_alive_s: float = 0.0
    max_lifetime_s: float = 3.0


@dataclass
class HitEvent:
    target_id: int
    projectile_index: int
    hit_position: np.ndarray
    relative_velocity: np.ndarray
    impact_impulse: np.ndarray


def compute_ballistic_lead(
    turret_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    muzzle_speed: float = 100.0,
    gravity: float = 9.81,
    iterations: int = 3,
) -> Tuple[float, float, float]:
    """
    Calculate required pan (azimuth) and tilt (elevation) to intercept moving target,
    compensating for target motion and gravitational projectile drop.

    Returns:
        aim_pan_deg: Required turret azimuth in degrees.
        aim_tilt_deg: Required turret elevation in degrees.
        time_of_flight_s: Estimated projectile flight time.
    """
    rel_pos = target_pos - turret_pos
    dist = np.linalg.norm(rel_pos)
    tof = dist / max(muzzle_speed, 10.0)

    aim_point = np.copy(target_pos)

    for _ in range(iterations):
        # Target displacement during time of flight
        predicted_pos = target_pos + target_vel * tof
        # Gravity drop compensation (aim higher by 0.5 * g * tof^2)
        aim_point = predicted_pos + np.array([0.0, 0.0, 0.5 * gravity * (tof ** 2)])
        # Update flight time based on new distance to aim point
        new_dist = np.linalg.norm(aim_point - turret_pos)
        tof = new_dist / max(muzzle_speed, 10.0)

    diff = aim_point - turret_pos
    aim_dist_xy = np.hypot(diff[0], diff[1])

    pan_deg = float(np.degrees(np.arctan2(diff[0], diff[1])))
    pan_deg = (pan_deg + 180.0) % 360.0 - 180.0
    tilt_deg = float(np.degrees(np.arctan2(diff[2], aim_dist_xy)))

    return pan_deg, tilt_deg, tof


class ProjectileManager:
    def __init__(self, muzzle_speed_m_s: float = 100.0, gravity_m_s2: float = 9.81):
        self.muzzle_speed = float(muzzle_speed_m_s)
        self.gravity = float(gravity_m_s2)
        self.projectiles: List[Projectile] = []
        self.total_shots_fired = 0
        self.total_hits_confirmed = 0

    def spawn_projectile(self, muzzle_pos: np.ndarray, aim_dir_unit: np.ndarray) -> None:
        """Spawn a new projectile exiting the barrel at muzzle_speed."""
        norm_dir = aim_dir_unit / max(np.linalg.norm(aim_dir_unit), 1e-6)
        vel = norm_dir * self.muzzle_speed
        proj = Projectile(
            position=np.copy(muzzle_pos),
            velocity=vel,
        )
        self.projectiles.append(proj)
        self.total_shots_fired += 1

    def update(self, dt: float, targets: list) -> List[HitEvent]:
        """
        Advance all active projectiles and check swept-segment collisions against targets.
        """
        if dt <= 0.0:
            return []

        hits: List[HitEvent] = []
        g_accel = np.array([0.0, 0.0, -self.gravity])

        for p_idx, proj in enumerate(self.projectiles):
            if not proj.alive:
                continue

            proj.time_alive_s += dt
            if proj.time_alive_s >= proj.max_lifetime_s or proj.position[2] <= 0.0:
                proj.alive = False
                continue

            # Old position before step
            p_start = np.copy(proj.position)

            # Physics: gravity + quadratic aerodynamic drag
            speed = np.linalg.norm(proj.velocity)
            drag_accel = -proj.drag_coeff * speed * proj.velocity
            net_accel = g_accel + drag_accel

            proj.velocity += net_accel * dt
            proj.position += proj.velocity * dt
            p_end = proj.position

            # Swept segment collision check against each target
            seg = p_end - p_start
            seg_len_sq = np.dot(seg, seg)

            for target in targets:
                if not target.is_active_threat:
                    continue

                # Point-to-segment closest distance
                v_target = target.position - p_start
                if seg_len_sq < 1e-8:
                    proj_t = 0.0
                else:
                    proj_t = np.clip(np.dot(v_target, seg) / seg_len_sq, 0.0, 1.0)

                closest_point = p_start + proj_t * seg
                dist_to_center = np.linalg.norm(closest_point - target.position)

                if dist_to_center <= target.radius:
                    # Direct Hit!
                    proj.alive = False
                    rel_v = proj.velocity - target.velocity
                    impulse = proj.mass_kg * rel_v

                    hit = HitEvent(
                        target_id=target.target_id,
                        projectile_index=p_idx,
                        hit_position=closest_point,
                        relative_velocity=rel_v,
                        impact_impulse=impulse,
                    )
                    hits.append(hit)
                    target.apply_hit(impulse)
                    self.total_hits_confirmed += 1
                    break

        # Remove dead projectiles to prevent memory leak
        self.projectiles = [p for p in self.projectiles if p.alive]
        return hits
