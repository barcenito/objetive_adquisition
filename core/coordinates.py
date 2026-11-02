"""
Coordinate transformations and geometry utilities for 2-DoF Turret and Radar.

Global Frame:
    X: Right (East)
    Y: Forward (North)
    Z: Up
Turret Angles:
    Pan / Azimuth (theta): Angle in horizontal XY plane. 0 deg is +Y (forward),
                           positive towards +X (clockwise looking down).
    Tilt / Elevation (phi): Angle above horizontal plane. 0 deg is horizontal,
                            +90 deg is vertically up (+Z).
"""

from typing import Tuple
import numpy as np


def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, distance: float = 1.0) -> np.ndarray:
    """
    Convert spherical coordinates (azimuth, elevation, distance) to Cartesian vector.
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)

    x = distance * np.sin(az_rad) * np.cos(el_rad)
    y = distance * np.cos(az_rad) * np.cos(el_rad)
    z = distance * np.sin(el_rad)
    return np.array([x, y, z], dtype=np.float64)


def cartesian_to_spherical(rel_pos: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert relative Cartesian vector [x, y, z] to (azimuth_deg, elevation_deg, distance).
    """
    x, y, z = float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])
    dist_xy = np.hypot(x, y)
    distance = np.sqrt(x**2 + y**2 + z**2)

    if distance < 1e-9:
        return 0.0, 0.0, 0.0

    azimuth_deg = np.degrees(np.arctan2(x, y))
    elevation_deg = np.degrees(np.arctan2(z, dist_xy))

    # Normalize azimuth to [-180, 180]
    azimuth_deg = (azimuth_deg + 180.0) % 360.0 - 180.0
    return float(azimuth_deg), float(elevation_deg), float(distance)


def get_turret_basis(pan_deg: float, tilt_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal basis (forward, right, up) aligned with turret boresight.

    Returns:
        forward: Unit vector along line-of-sight / barrel axis.
        right: Unit vector pointing to the right of turret antenna.
        up: Unit vector pointing up relative to turret antenna tilt.
    """
    pan_rad = np.radians(pan_deg)
    tilt_rad = np.radians(tilt_deg)

    # Forward boresight vector
    forward = np.array([
        np.sin(pan_rad) * np.cos(tilt_rad),
        np.cos(pan_rad) * np.cos(tilt_rad),
        np.sin(tilt_rad)
    ], dtype=np.float64)

    # Right vector (perpendicular to pan in horizontal plane)
    right = np.array([
        np.cos(pan_rad),
        -np.sin(pan_rad),
        0.0
    ], dtype=np.float64)

    # Up vector (right x forward = +Z)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    return forward, right, up


def compute_antenna_angular_offset(
    turret_pos: np.ndarray,
    pan_deg: float,
    tilt_deg: float,
    target_pos: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute angular offsets of target relative to turret boresight in antenna coordinates.

    Returns:
        offset_h_deg: Horizontal off-boresight angle (azimuth in antenna frame, degrees).
        offset_v_deg: Vertical off-boresight angle (elevation in antenna frame, degrees).
        distance: Radial range to target in meters.
    """
    rel = target_pos - turret_pos
    dist = np.linalg.norm(rel)
    if dist < 1e-9:
        return 0.0, 0.0, 0.0

    u_target = rel / dist
    forward, right, up = get_turret_basis(pan_deg, tilt_deg)

    # Projection onto antenna axes
    proj_f = np.dot(u_target, forward)
    proj_r = np.dot(u_target, right)
    proj_u = np.dot(u_target, up)

    # Compute horizontal and vertical off-axis angles
    offset_h_deg = float(np.degrees(np.arctan2(proj_r, max(proj_f, 1e-6))))
    offset_v_deg = float(np.degrees(np.arctan2(proj_u, max(proj_f, 1e-6))))

    return offset_h_deg, offset_v_deg, float(dist)
