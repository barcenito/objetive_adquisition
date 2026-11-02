"""
Physical Servomotor Model.

Simulates real RC / Industrial servomotor dynamics:
- Angular position limits (mechanical end-stops)
- PWM pulse width translation (1000 - 2000 microseconds standard 50 Hz PWM)
- Finite maximum angular velocity (slew-rate limit)
- Finite torque and angular acceleration limit
- Deadband to eliminate jitter and micro-hunting
- Second-order dynamic response with velocity and acceleration saturation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ServoTelemetry:
    current_angle_deg: float
    current_velocity_deg_s: float
    target_angle_deg: float
    current_pwm_us: float
    is_at_target: bool


class ServoMotor:
    def __init__(
        self,
        name: str = "servo",
        min_angle_deg: float = -180.0,
        max_angle_deg: float = 180.0,
        max_velocity_deg_s: float = 360.0,      # Slew rate limit (~0.16s / 60 deg)
        max_acceleration_deg_s2: float = 2400.0, # Acceleration / torque limit
        deadband_deg: float = 0.3,              # Deadband to suppress jitter
        initial_angle_deg: float = 0.0,
        min_pulse_us: float = 1000.0,
        max_pulse_us: float = 2000.0,
        natural_frequency: float = 25.0,        # rad/s for internal PID controller response
        damping_ratio: float = 1.0,             # Critically damped
    ):
        self.name = name
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)
        self.max_velocity_deg_s = float(max_velocity_deg_s)
        self.max_acceleration_deg_s2 = float(max_acceleration_deg_s2)
        self.deadband_deg = float(deadband_deg)
        self.min_pulse_us = float(min_pulse_us)
        self.max_pulse_us = float(max_pulse_us)

        # Internal control gains (2nd order system: d2x/dt2 = -2*zeta*wn*dx/dt - wn^2*(x - x_target))
        self.wn = float(natural_frequency)
        self.zeta = float(damping_ratio)
        self.kp = self.wn ** 2
        self.kd = 2.0 * self.zeta * self.wn

        # States
        self.current_angle_deg = float(np.clip(initial_angle_deg, self.min_angle_deg, self.max_angle_deg))
        self.current_velocity_deg_s = 0.0
        self.target_angle_deg = self.current_angle_deg

    def angle_to_pulse_us(self, angle_deg: float) -> float:
        """Map angle within [min_angle, max_angle] to PWM pulse width [min_pulse_us, max_pulse_us]."""
        clamped_angle = np.clip(angle_deg, self.min_angle_deg, self.max_angle_deg)
        fraction = (clamped_angle - self.min_angle_deg) / max(self.max_angle_deg - self.min_angle_deg, 1e-6)
        return float(self.min_pulse_us + fraction * (self.max_pulse_us - self.min_pulse_us))

    def pulse_us_to_angle(self, pulse_us: float) -> float:
        """Map PWM pulse width to target angle."""
        clamped_pulse = np.clip(pulse_us, self.min_pulse_us, self.max_pulse_us)
        fraction = (clamped_pulse - self.min_pulse_us) / max(self.max_pulse_us - self.min_pulse_us, 1e-6)
        return float(self.min_angle_deg + fraction * (self.max_angle_deg - self.min_angle_deg))

    def set_target_angle(self, target_deg: float) -> None:
        """Command target setpoint angle in degrees (automatically clamped to mechanical range)."""
        self.target_angle_deg = float(np.clip(target_deg, self.min_angle_deg, self.max_angle_deg))

    def set_target_pulse_us(self, pulse_us: float) -> None:
        """Command target setpoint via standard PWM pulse width in microseconds."""
        angle = self.pulse_us_to_angle(pulse_us)
        self.set_target_angle(angle)

    def update(self, dt: float) -> None:
        """
        Integrate servo physics for a time step dt.
        Applies PID closed-loop torque, acceleration clipping, slew-rate clamping and deadband.
        """
        if dt <= 0.0:
            return

        error = self.target_angle_deg - self.current_angle_deg

        # Deadband check: if within deadband and near rest, eliminate hunting
        if abs(error) <= self.deadband_deg and abs(self.current_velocity_deg_s) < 1.0:
            self.current_velocity_deg_s = 0.0
            return

        # Desired acceleration from internal PD control
        desired_accel = self.kp * error - self.kd * self.current_velocity_deg_s

        # Clamp acceleration to maximum torque/acceleration limit
        accel = np.clip(desired_accel, -self.max_acceleration_deg_s2, self.max_acceleration_deg_s2)

        # Velocity integration
        new_velocity = self.current_velocity_deg_s + accel * dt

        # Slew rate (velocity) clamping
        new_velocity = np.clip(new_velocity, -self.max_velocity_deg_s, self.max_velocity_deg_s)

        # Position integration
        new_angle = self.current_angle_deg + new_velocity * dt

        # Mechanical hard stops
        if new_angle <= self.min_angle_deg:
            new_angle = self.min_angle_deg
            new_velocity = max(0.0, new_velocity)
        elif new_angle >= self.max_angle_deg:
            new_angle = self.max_angle_deg
            new_velocity = min(0.0, new_velocity)

        self.current_angle_deg = float(new_angle)
        self.current_velocity_deg_s = float(new_velocity)

    @property
    def is_at_target(self) -> bool:
        """Check if servo is aligned with target within deadband and practically stationary."""
        error = abs(self.target_angle_deg - self.current_angle_deg)
        return bool(error <= self.deadband_deg and abs(self.current_velocity_deg_s) < 2.0)

    def get_telemetry(self) -> ServoTelemetry:
        """Return snapshot of current servo states."""
        return ServoTelemetry(
            current_angle_deg=self.current_angle_deg,
            current_velocity_deg_s=self.current_velocity_deg_s,
            target_angle_deg=self.target_angle_deg,
            current_pwm_us=self.angle_to_pulse_us(self.current_angle_deg),
            is_at_target=self.is_at_target,
        )
