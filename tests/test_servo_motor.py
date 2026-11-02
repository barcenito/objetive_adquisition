"""
Unit and dynamic empirical tests for the ServoMotor physical model.
"""

import unittest
from actuators.servo_motor import ServoMotor


class TestServoMotor(unittest.TestCase):
    def setUp(self):
        self.servo = ServoMotor(
            name="pan_servo",
            min_angle_deg=-180.0,
            max_angle_deg=180.0,
            max_velocity_deg_s=360.0,
            max_acceleration_deg_s2=2400.0,
            deadband_deg=0.3,
            initial_angle_deg=0.0,
            min_pulse_us=1000.0,
            max_pulse_us=2000.0,
        )

    def test_pwm_conversion(self):
        # 0 deg (center) -> 1500 us
        self.assertAlmostEqual(self.servo.angle_to_pulse_us(0.0), 1500.0, places=2)
        # -180 deg (min) -> 1000 us
        self.assertAlmostEqual(self.servo.angle_to_pulse_us(-180.0), 1000.0, places=2)
        # +180 deg (max) -> 2000 us
        self.assertAlmostEqual(self.servo.angle_to_pulse_us(180.0), 2000.0, places=2)
        # Roundtrip
        for angle in [-180, -90, -45, 0, 45, 90, 180]:
            pulse = self.servo.angle_to_pulse_us(angle)
            reconstructed_angle = self.servo.pulse_us_to_angle(pulse)
            self.assertAlmostEqual(reconstructed_angle, angle, places=3)

    def test_slew_rate_limit(self):
        # Request a large step jump: 0 -> 90 deg
        self.servo.set_target_angle(90.0)
        dt = 0.01  # 10 ms (100 Hz physics clock)
        max_seen_vel = 0.0

        # Simulate 1 second
        for _ in range(100):
            self.servo.update(dt)
            max_seen_vel = max(max_seen_vel, abs(self.servo.current_velocity_deg_s))
            # Velocity must NEVER exceed max_velocity_deg_s
            self.assertLessEqual(abs(self.servo.current_velocity_deg_s), self.servo.max_velocity_deg_s + 1e-6)

        # Confirm that the slew rate limit was actually active and approached
        self.assertGreater(max_seen_vel, 300.0)
        # After 1 second, it must have reached 90 deg within the mechanical deadband
        self.assertAlmostEqual(self.servo.current_angle_deg, 90.0, delta=self.servo.deadband_deg)
        self.assertTrue(self.servo.is_at_target)

    def test_deadband_suppression(self):
        self.servo.current_angle_deg = 10.0
        self.servo.current_velocity_deg_s = 0.0
        # Command within deadband: 10.15 deg (difference 0.15 deg < 0.3 deg deadband)
        self.servo.set_target_angle(10.15)

        self.servo.update(0.01)
        # Must stay at 10.0 with 0 velocity (no jitter / micro-stepping)
        self.assertEqual(self.servo.current_angle_deg, 10.0)
        self.assertEqual(self.servo.current_velocity_deg_s, 0.0)

    def test_mechanical_limits_saturation(self):
        # Command out-of-range angles
        self.servo.set_target_angle(250.0)
        self.assertEqual(self.servo.target_angle_deg, 180.0)

        self.servo.set_target_angle(-300.0)
        self.assertEqual(self.servo.target_angle_deg, -180.0)


if __name__ == "__main__":
    unittest.main()
