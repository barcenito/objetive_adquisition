"""
Acconeer A121 60 GHz Pulsed Coherent Radar (PCR) Range Sensor Model.

Based on official Acconeer A121 SDK & Waveshare documentation:
- Operating frequency: 60 GHz
- Field of View (FoV): 53 deg (Horizontal) x 65 deg (Vertical)
- Range resolution: 0.03 m (Profile 1)
- Range accuracy: 0.03 m (sub-centimeter to 3 cm)
- Min distance: 0.2 m, Max distance: 10.0 m (for small UAS / drone RCS ~0.02 m^2)
- Output structure matches acc_detector_distance_result_t from Acconeer C SDK.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from core.coordinates import compute_antenna_angular_offset


@dataclass
class A121DistanceResult:
    """
    Python equivalent of acc_detector_distance_result_t from Acconeer A121 SDK.
    """
    num_distances: int
    distances: List[float] = field(default_factory=list)      # Detected peak distances in meters
    strengths: List[float] = field(default_factory=list)      # Peak signal strength in dB
    near_start_edge_status: bool = False
    temperature_c: float = 25.0

    # Internal diagnostic offsets relative to boresight (for simulation evaluation and visualization)
    estimated_angles_deg: List[tuple] = field(default_factory=list)


class AcconeerA121Radar:
    def __init__(
        self,
        fov_horizontal_deg: float = 53.0,
        fov_vertical_deg: float = 65.0,
        min_range_m: float = 0.20,
        max_range_m: float = 10.0,
        range_resolution_m: float = 0.03,
        range_noise_std_m: float = 0.015,   # 1-sigma giving +/- 0.03m accuracy
        cfar_threshold_db: float = 8.0,      # Constant False Alarm Rate threshold
        base_power_db: float = 65.0,         # Calibration constant (detects RCS=0.03 m^2 up to 10m)
        temperature_c: float = 25.0,
        seed: Optional[int] = None,
    ):
        self.fov_h = float(fov_horizontal_deg)
        self.fov_v = float(fov_vertical_deg)
        self.min_range = float(min_range_m)
        self.max_range = float(max_range_m)
        self.range_res = float(range_resolution_m)
        self.range_noise_std = float(range_noise_std_m)
        self.cfar_threshold_db = float(cfar_threshold_db)
        self.base_power_db = float(base_power_db)
        self.temperature_c = float(temperature_c)
        self.rng = np.random.default_rng(seed)

    def compute_antenna_gain_linear(self, off_h_deg: float, off_v_deg: float) -> float:
        """
        Compute Gaussian beam antenna gain G(theta, phi) normalized to 1.0 at boresight.
        Half-Power Beamwidth (HPBW, -3 dB) matches (fov_h, fov_v).
        """
        # Exponent -4*ln(2) gives exactly 0.5 power at off = HPBW/2
        ln2_4 = 4.0 * np.log(2.0)
        norm_h = (off_h_deg / (self.fov_h / 2.0)) ** 2
        norm_v = (off_v_deg / (self.fov_v / 2.0)) ** 2
        gain = np.exp(-ln2_4 * 0.25 * (norm_h + norm_v))
        return float(gain)

    def measure(
        self,
        turret_pos: np.ndarray,
        turret_pan_deg: float,
        turret_tilt_deg: float,
        target_positions: List[np.ndarray],
        target_rcs_list: Optional[List[float]] = None,
        apply_noise: bool = True,
    ) -> A121DistanceResult:
        """
        Perform a radar sweep from turret position with given pan and tilt orientation.

        Args:
            turret_pos: [x, y, z] of radar antenna.
            turret_pan_deg: Pan / Azimuth in degrees.
            turret_tilt_deg: Tilt / Elevation in degrees.
            target_positions: List of [x, y, z] coordinates for each target.
            target_rcs_list: Radar Cross Section (m^2) for each target (default 0.03 m^2 for drone).
            apply_noise: Whether to inject physical Gaussian noise into range and amplitude.

        Returns:
            A121DistanceResult matching the Acconeer A121 SDK distance detector output.
        """
        if target_rcs_list is None:
            target_rcs_list = [0.03] * len(target_positions)

        raw_detections = []
        near_edge = False

        for pos, rcs in zip(target_positions, target_rcs_list):
            off_h, off_v, distance = compute_antenna_angular_offset(
                turret_pos, turret_pan_deg, turret_tilt_deg, pos
            )

            # Check near-edge zone (< 0.2 m)
            if distance < self.min_range:
                near_edge = True
                continue

            # Check max theoretical detection boundary
            if distance > self.max_range * 1.5:
                continue

            # Check if within beam cone boundary (beam cutoff at ~1.15 * HPBW)
            half_h = self.fov_h / 2.0
            half_v = self.fov_v / 2.0
            norm_angle_sq = (off_h / half_h) ** 2 + (off_v / half_v) ** 2
            if norm_angle_sq > 1.35:  # Outside practical antenna mainlobe
                continue

            gain_lin = self.compute_antenna_gain_linear(off_h, off_v)
            if gain_lin < 0.05:  # Below -13 dB sidelobe cutoff
                continue

            gain_db = 10.0 * np.log10(max(gain_lin, 1e-6))

            # Radar Range Equation (Monostatic: 1/R^4 -> -40*log10(R))
            rcs_db = 10.0 * np.log10(max(rcs, 1e-4))
            range_loss_db = -40.0 * np.log10(max(distance, 0.20))

            snr_db = self.base_power_db + rcs_db + 2.0 * gain_db + range_loss_db

            if apply_noise:
                snr_db += self.rng.normal(0.0, 0.8)

            # CFAR detection threshold test
            if snr_db >= self.cfar_threshold_db and distance <= self.max_range:
                measured_dist = distance
                if apply_noise:
                    measured_dist += self.rng.normal(0.0, self.range_noise_std)
                measured_dist = max(self.min_range, measured_dist)

                raw_detections.append({
                    "distance": float(measured_dist),
                    "strength_db": float(snr_db),
                    "off_h": off_h,
                    "off_v": off_v,
                })

        # Group peaks that are separated by less than range_resolution_m
        raw_detections.sort(key=lambda d: d["distance"])
        filtered_peaks = []

        for det in raw_detections:
            if not filtered_peaks:
                filtered_peaks.append(det)
            else:
                last = filtered_peaks[-1]
                # If closer than range resolution, keep the stronger peak (CFAR peak sorting)
                if abs(det["distance"] - last["distance"]) < self.range_res:
                    if det["strength_db"] > last["strength_db"]:
                        filtered_peaks[-1] = det
                else:
                    filtered_peaks.append(det)

        # Acconeer Distance Detector limits output to maximum 10 distances
        filtered_peaks = filtered_peaks[:10]

        distances = [round(p["distance"], 4) for p in filtered_peaks]
        strengths = [round(p["strength_db"], 2) for p in filtered_peaks]
        angles = [(round(p["off_h"], 2), round(p["off_v"], 2)) for p in filtered_peaks]

        return A121DistanceResult(
            num_distances=len(distances),
            distances=distances,
            strengths=strengths,
            near_start_edge_status=near_edge,
            temperature_c=self.temperature_c,
            estimated_angles_deg=angles,
        )
