"""
Decoupled Turret Controller Algorithm.

Written as an independent modular file according to user specifications.
Receives turret telemetry and Acconeer A121 radar sensor input,
and outputs target servo angles (or PWM) and fire trigger commands.

Supports:
1. SEARCH: Systematic 3D airspace scanning pattern with pan and tilt servos.
2. ACQUIRE & TRACK: Locking onto targets detected by the Acconeer A121 radar,
   filtering position/velocity, and computing ballistic lead angle.
3. FIRE: Triggering weapon discharge when aligned within fire envelope.
4. MANUAL_TEST: Direct test movement mode for servo validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from sensors.radar_a121 import A121DistanceResult
from core.coordinates import spherical_to_cartesian, cartesian_to_spherical, get_turret_basis
from ballistics.projectile_system import compute_ballistic_lead


@dataclass
class TurretInput:
    """
    Input structure passed to the turret controller on every simulation tick.
    """
    timestamp: float
    current_pan_deg: float
    current_tilt_deg: float
    current_pan_vel_dps: float
    current_tilt_vel_dps: float
    ready_to_fire: bool
    radar_data: A121DistanceResult
    extra_data: dict = field(default_factory=dict)


@dataclass
class TurretAction:
    """
    Action commands returned by the turret controller to the actuators.
    """
    target_pan_deg: float
    target_tilt_deg: float
    fire: bool
    status_mode: str
    target_pan_pwm_us: float = 1500.0
    target_tilt_pwm_us: float = 1500.0


class ControllerMode(Enum):
    SEARCH = "search"
    ACQUIRE = "acquire"
    TRACK = "track"
    ENGAGE = "engage"
    FIRE = "fire"
    VERIFY = "verify"
    MANUAL_TEST = "manual_test"


class TurretController:
    def __init__(
        self,
        muzzle_speed_m_s: float = 100.0,
        fire_alignment_threshold_deg: float = 2.0,
        search_pan_range: Tuple[float, float] = (-50.0, 50.0),
        search_tilt_range: Tuple[float, float] = (10.0, 35.0),
        search_pan_speed_dps: float = 60.0,
        search_tilt_freq_hz: float = 0.25,
    ):
        self.muzzle_speed = float(muzzle_speed_m_s)
        self.fire_thresh = float(fire_alignment_threshold_deg)
        self.pan_min, self.pan_max = search_pan_range
        self.tilt_min, self.tilt_max = search_tilt_range
        self.search_pan_speed = search_pan_speed_dps
        self.search_tilt_freq = search_tilt_freq_hz

        self.mode = ControllerMode.SEARCH
        self.search_pan_dir = 1.0
        self.search_pan_target = 0.0

        # State estimation / Tracking filter
        self.tracked_pos: Optional[np.ndarray] = None
        self.tracked_vel: np.ndarray = np.zeros(3)
        self.last_track_time: Optional[float] = None
        self.track_loss_timeout_s = 0.5  # Return to search if target lost
        self.time_since_detection = 999.0
        self.shots_on_current_target = 0

        # Manual test overrides
        self.manual_pan: float = 0.0
        self.manual_tilt: float = 15.0
        self.manual_fire: bool = False

    def set_manual_test_mode(self, pan_deg: float, tilt_deg: float, fire: bool = False) -> None:
        """Switch to manual test mode to test turret servo movements."""
        self.mode = ControllerMode.MANUAL_TEST
        self.manual_pan = float(pan_deg)
        self.manual_tilt = float(tilt_deg)
        self.manual_fire = bool(fire)

    def set_autonomous_mode(self) -> None:
        """Switch to autonomous search-and-destroy radar mode."""
        self.mode = ControllerMode.SEARCH

    def _update_search_pattern(self, current_pan: float, dt: float) -> Tuple[float, float]:
        """Compute smooth scan pattern sweeping airspace with A121 radar beam."""
        self.search_pan_target += self.search_pan_dir * self.search_pan_speed * dt
        if self.search_pan_target >= self.pan_max:
            self.search_pan_target = self.pan_max
            self.search_pan_dir = -1.0
        elif self.search_pan_target <= self.pan_min:
            self.search_pan_target = self.pan_min
            self.search_pan_dir = 1.0

        # Sinusoidal nodding in tilt
        tilt_center = (self.tilt_min + self.tilt_max) * 0.5
        tilt_amp = (self.tilt_max - self.tilt_min) * 0.5
        phase = (self.search_pan_target - self.pan_min) / max(self.pan_max - self.pan_min, 1.0)
        target_tilt = tilt_center + tilt_amp * np.sin(2.0 * np.pi * phase)
        return float(self.search_pan_target), float(target_tilt)

    def _reconstruct_3d_position(
        self,
        distance: float,
        off_h_deg: float,
        off_v_deg: float,
        current_pan_deg: float,
        current_tilt_deg: float,
        turret_origin: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct 3D Cartesian position from radar spherical echo."""
        f, r, u = get_turret_basis(current_pan_deg, current_tilt_deg)
        tan_h = np.tan(np.radians(off_h_deg))
        tan_v = np.tan(np.radians(off_v_deg))
        u_target = (f + tan_h * r + tan_v * u) / np.sqrt(1.0 + tan_h**2 + tan_v**2)
        return turret_origin + distance * u_target

    def update(self, inputs: TurretInput, dt: float = 0.01) -> TurretAction:
        """
        Main execution step of the decoupled turret algorithm.

        Args:
            inputs: TurretInput containing current servo states and Acconeer A121 radar readings.
            dt: Simulation delta time.

        Returns:
            TurretAction with target angles and fire trigger.
        """
        # 1. Manual test mode override
        if self.mode == ControllerMode.MANUAL_TEST:
            return TurretAction(
                target_pan_deg=self.manual_pan,
                target_tilt_deg=self.manual_tilt,
                fire=self.manual_fire,
                status_mode="MANUAL_TEST",
                target_pan_pwm_us=1000.0 + ((self.manual_pan + 180.0) / 360.0) * 1000.0,
                target_tilt_pwm_us=1000.0 + ((self.manual_tilt + 15.0) / 100.0) * 1000.0,
            )

        turret_origin = np.array([0.0, 0.0, 0.5])
        radar = inputs.radar_data
        target_detected = radar.num_distances > 0

        if target_detected:
            self.time_since_detection = 0.0

            # Reconstruct 3D candidates for all detected peaks
            candidate_positions = []
            for idx in range(radar.num_distances):
                d = radar.distances[idx]
                if radar.estimated_angles_deg and idx < len(radar.estimated_angles_deg):
                    oh, ov = radar.estimated_angles_deg[idx]
                else:
                    oh, ov = 0.0, 0.0
                cand_pos = self._reconstruct_3d_position(
                    d, oh, ov, inputs.current_pan_deg, inputs.current_tilt_deg, turret_origin
                )
                candidate_positions.append(cand_pos)

            # Data association / Track gating
            delta_t = max(inputs.timestamp - self.last_track_time, 1e-4) if self.last_track_time else dt

            if self.tracked_pos is None or (self.last_track_time and (inputs.timestamp - self.last_track_time > self.track_loss_timeout_s)):
                # Fresh acquisition: choose closest target
                selected_pos = candidate_positions[0]
                self.tracked_pos = selected_pos
                self.tracked_vel = np.zeros(3)
                self.mode = ControllerMode.TRACK
                self.shots_on_current_target = 0
            else:
                pred_pos = self.tracked_pos + self.tracked_vel * delta_t
                # Nearest Neighbor matching
                dists_to_pred = [np.linalg.norm(c - pred_pos) for c in candidate_positions]
                min_idx = int(np.argmin(dists_to_pred))
                min_dist = dists_to_pred[min_idx]

                # Gate check: if closest detection is too far (> 1.6m in delta_t), old target was lost/neutralized
                association_gate = 1.6
                if min_dist <= association_gate:
                    # Valid associated measurement
                    meas_pos = candidate_positions[min_idx]
                    residual = meas_pos - pred_pos
                    # Alpha-Beta filter update
                    alpha = 0.45
                    beta = 0.05
                    self.tracked_pos = pred_pos + alpha * residual
                    vel_update = self.tracked_vel + (beta / delta_t) * residual
                    # Clamp maximum physical speed of aerial drone
                    v_norm = np.linalg.norm(vel_update)
                    if v_norm > 8.0:
                        vel_update = (vel_update / v_norm) * 8.0
                    self.tracked_vel = vel_update
                    self.mode = ControllerMode.TRACK
                else:
                    # Switch cleanly to next closest target in radar cone
                    self.tracked_pos = candidate_positions[min_idx]
                    self.tracked_vel = np.zeros(3)
                    self.mode = ControllerMode.TRACK
                    self.shots_on_current_target = 0

            self.last_track_time = inputs.timestamp

            # Check target descent / ground impact
            if self.tracked_pos is not None and self.tracked_pos[2] <= 0.25:
                # Target is hitting the ground, verify neutralization and search for others
                self.mode = ControllerMode.VERIFY
                self.tracked_pos = None
        else:
            self.time_since_detection += dt
            if self.time_since_detection > self.track_loss_timeout_s:
                self.mode = ControllerMode.SEARCH
                self.tracked_pos = None

        # 3. Action generation based on mode
        if self.mode in [ControllerMode.TRACK, ControllerMode.ACQUIRE] and self.tracked_pos is not None:
            # Ballistic lead angle calculation
            aim_pan, aim_tilt, _ = compute_ballistic_lead(
                turret_origin, self.tracked_pos, self.tracked_vel, muzzle_speed=self.muzzle_speed
            )

            # Check alignment error
            error_pan = abs(inputs.current_pan_deg - aim_pan)
            error_tilt = abs(inputs.current_tilt_deg - aim_tilt)
            alignment_error = np.hypot(error_pan, error_tilt)

            # Fire condition: well aligned, weapon ready
            should_fire = bool(alignment_error <= self.fire_thresh and inputs.ready_to_fire)

            if should_fire:
                mode_str = "TRACK_ENGAGE"
                self.mode = ControllerMode.ENGAGE
                self.shots_on_current_target += 1
            else:
                mode_str = "TRACK_AIMING" if self.mode == ControllerMode.TRACK else "ACQUIRE_LOCK"

            return TurretAction(
                target_pan_deg=float(aim_pan),
                target_tilt_deg=float(aim_tilt),
                fire=should_fire,
                status_mode=mode_str,
                target_pan_pwm_us=1000.0 + ((aim_pan + 180.0) / 360.0) * 1000.0,
                target_tilt_pwm_us=1000.0 + ((aim_tilt + 15.0) / 100.0) * 1000.0,
            )
        else:
            # SEARCH MODE: Sweep pattern
            target_pan, target_tilt = self._update_search_pattern(inputs.current_pan_deg, dt)
            return TurretAction(
                target_pan_deg=target_pan,
                target_tilt_deg=target_tilt,
                fire=False,
                status_mode="SEARCH_SCANNING",
                target_pan_pwm_us=1000.0 + ((target_pan + 180.0) / 360.0) * 1000.0,
                target_tilt_pwm_us=1000.0 + ((target_tilt + 15.0) / 100.0) * 1000.0,
            )
