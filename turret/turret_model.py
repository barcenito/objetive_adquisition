"""
2-DoF Turret Model with Pan/Tilt Servos, Radar Mount, and Ballistic Weapon.

Simulates the mechanical turret platform:
- Pan servo (rotation around vertical axis, azimuth)
- Tilt servo (rotation around horizontal axis, elevation)
- Coaxial radar and weapon barrel
- Fire trigger with rate-of-fire cooldown
- Muzzle position and boresight vector calculation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from actuators.servo_motor import ServoMotor
from core.coordinates import get_turret_basis


@dataclass
class TurretState:
    pan_angle_deg: float
    tilt_angle_deg: float
    pan_velocity_dps: float
    tilt_velocity_dps: float
    pan_pwm_us: float
    tilt_pwm_us: float
    ready_to_fire: bool
    muzzle_position: np.ndarray
    boresight_unit_vector: np.ndarray


class TurretModel:
    def __init__(
        self,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        barrel_length_m: float = 0.30,
        fire_rate_hz: float = 5.0,             # 5 rounds per second
        pan_slew_rate_dps: float = 360.0,
        tilt_slew_rate_dps: float = 300.0,
        pan_limits_deg: Tuple[float, float] = (-180.0, 180.0),
        tilt_limits_deg: Tuple[float, float] = (-15.0, 85.0),
    ):
        self.base_pos = np.array(base_position, dtype=np.float64)
        self.barrel_length = float(barrel_length_m)
        self.fire_cooldown_s = 1.0 / max(fire_rate_hz, 0.1)
        self.cooldown_timer = 0.0

        # Instantiate pan and tilt servos
        self.pan_servo = ServoMotor(
            name="pan_servo",
            min_angle_deg=pan_limits_deg[0],
            max_angle_deg=pan_limits_deg[1],
            max_velocity_deg_s=pan_slew_rate_dps,
            initial_angle_deg=0.0,
        )
        self.tilt_servo = ServoMotor(
            name="tilt_servo",
            min_angle_deg=tilt_limits_deg[0],
            max_angle_deg=tilt_limits_deg[1],
            max_velocity_deg_s=tilt_slew_rate_dps,
            initial_angle_deg=15.0,
        )

    def command_servos(self, target_pan_deg: float, target_tilt_deg: float) -> None:
        """Command target angles directly in degrees."""
        self.pan_servo.set_target_angle(target_pan_deg)
        self.tilt_servo.set_target_angle(target_tilt_deg)

    def command_pwm(self, pan_us: float, tilt_us: float) -> None:
        """Command target angles via PWM pulse widths in microseconds (1000 to 2000 us)."""
        self.pan_servo.set_target_pulse_us(pan_us)
        self.tilt_servo.set_target_pulse_us(tilt_us)

    def update(self, dt: float) -> None:
        """Advance servo mechanics and reload cooldown."""
        if dt <= 0.0:
            return
        self.pan_servo.update(dt)
        self.tilt_servo.update(dt)
        if self.cooldown_timer > 0.0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

    @property
    def ready_to_fire(self) -> bool:
        """Weapon ready status (not in cooldown)."""
        return self.cooldown_timer <= 1e-6

    def trigger_fire(self) -> bool:
        """
        Attempt to trigger weapon fire.
        Returns True if a round was successfully discharged, False if in cooldown.
        """
        if self.ready_to_fire:
            self.cooldown_timer = self.fire_cooldown_s
            return True
        return False

    def get_boresight_vector(self) -> np.ndarray:
        """Return 3D forward unit vector along the barrel and radar centerline."""
        forward, _, _ = get_turret_basis(self.pan_servo.current_angle_deg, self.tilt_servo.current_angle_deg)
        return forward

    def get_muzzle_position(self) -> np.ndarray:
        """Return 3D coordinates of the weapon muzzle tip where projectiles emerge."""
        forward = self.get_boresight_vector()
        return self.base_pos + forward * self.barrel_length

    def get_state(self) -> TurretState:
        """Return full snapshot of current turret kinematic and actuator telemetry."""
        fwd = self.get_boresight_vector()
        muzzle = self.get_muzzle_position()
        return TurretState(
            pan_angle_deg=self.pan_servo.current_angle_deg,
            tilt_angle_deg=self.tilt_servo.current_angle_deg,
            pan_velocity_dps=self.pan_servo.current_velocity_deg_s,
            tilt_velocity_dps=self.tilt_servo.current_velocity_deg_s,
            pan_pwm_us=self.pan_servo.angle_to_pulse_us(self.pan_servo.current_angle_deg),
            tilt_pwm_us=self.tilt_servo.angle_to_pulse_us(self.tilt_servo.current_angle_deg),
            ready_to_fire=self.ready_to_fire,
            muzzle_position=muzzle,
            boresight_unit_vector=fwd,
        )
