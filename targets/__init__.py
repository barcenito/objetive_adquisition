"""Targets package: 3D autonomous drone targets, trajectory generators and hit dynamics."""
from targets.trajectory_gen import TrajectoryGenerator
from targets.drone_target import DroneTarget, TargetStatus

__all__ = ["TrajectoryGenerator", "DroneTarget", "TargetStatus"]
