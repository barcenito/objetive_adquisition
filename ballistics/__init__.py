"""Ballistics package: projectile physics, lead prediction and collision checking."""
from ballistics.projectile_system import Projectile, HitEvent, ProjectileManager, compute_ballistic_lead

__all__ = ["Projectile", "HitEvent", "ProjectileManager", "compute_ballistic_lead"]
