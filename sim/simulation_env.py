"""
Full Simulation Environment Orchestrator.

Synchronizes at fixed physics frequency (default 100 Hz):
- Target Drones (1 to 3 targets with individual speeds and 3D paths)
- 2-DoF Pan-Tilt Turret actuated by physical servomotors
- Acconeer A121 60 GHz PCR radar sensor mounted on turret boresight
- Decoupled Turret Controller algorithm
- Ballistic Projectiles and collision detection
- Real-time telemetry logging and hit statistics.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from targets.trajectory_gen import TrajectoryGenerator
from targets.drone_target import DroneTarget, TargetStatus
from targets.crazyflow_swarm import CrazyflowSwarmManager, CRAZYFLOW_AVAILABLE
from actuators.servo_motor import ServoMotor
from sensors.radar_a121 import AcconeerA121Radar, A121DistanceResult
from turret.turret_model import TurretModel, TurretState
from turret.turret_controller import TurretController, TurretInput, TurretAction
from ballistics.projectile_system import ProjectileManager, HitEvent


@dataclass
class SimTelemetrySnapshot:
    sim_time_s: float
    turret_state: TurretState
    radar_data: A121DistanceResult
    controller_action: TurretAction
    drones_alive: int
    drones_hit: int
    drones_downed: int
    active_projectiles: int
    shots_fired: int
    hits_confirmed: int


class TurretSimulationEnv:
    def __init__(
        self,
        num_drones: int = 3,
        drone_speeds: Optional[List[float]] = None,
        physics_freq_hz: float = 100.0,
        turret_pos: tuple = (0.0, 0.0, 0.5),
        seed: Optional[int] = 42,
        use_crazyflow: bool = True,
    ):
        self.dt = 1.0 / max(physics_freq_hz, 10.0)
        self.sim_time = 0.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_crazyflow = bool(use_crazyflow and CRAZYFLOW_AVAILABLE)

        # 1. Turret & Radar
        self.turret = TurretModel(base_position=turret_pos)
        self.radar = AcconeerA121Radar(seed=seed)

        # 2. Controller
        self.controller = TurretController(muzzle_speed_m_s=100.0)

        # 3. Ballistics
        self.ballistics = ProjectileManager(muzzle_speed_m_s=100.0)

        # 4. Target Drones
        self.num_drones = min(max(1, num_drones), 3)  # Maximum 3 drones as specified
        if drone_speeds is None:
            # Default speeds for 1, 2, or 3 drones
            default_speeds = [1.8, 3.2, 4.6]
            self.drone_speeds = default_speeds[:self.num_drones]
        else:
            self.drone_speeds = drone_speeds[:self.num_drones]

        self.swarm_manager: Optional[CrazyflowSwarmManager] = None
        self.drones: list = []
        self._init_drones()

        # Telemetry history
        self.history: List[SimTelemetrySnapshot] = []
        self.all_hit_events: List[HitEvent] = []

    def _init_drones(self) -> None:
        """Instantiate targets using CrazyflowSwarmManager (JAX/Crazyflow) or standalone targets."""
        if self.use_crazyflow:
            self.swarm_manager = CrazyflowSwarmManager(
                num_drones=self.num_drones,
                drone_speeds=self.drone_speeds,
                physics_freq_hz=1.0 / self.dt,
                seed=self.seed,
            )
            self.drones = self.swarm_manager.get_drone_views()
        else:
            self.drones = []
            for i in range(self.num_drones):
                drone_seed = (self.seed + i * 137) if self.seed is not None else None
                traj_gen = TrajectoryGenerator(seed=drone_seed)
                speed = self.drone_speeds[i]
                drone = DroneTarget(target_id=i, nominal_speed_m_s=speed, trajectory_gen=traj_gen)
                self.drones.append(drone)

    def reset(self) -> None:
        """Reset simulation environment to t=0."""
        self.sim_time = 0.0
        self.turret = TurretModel(base_position=tuple(self.turret.base_pos))
        self.radar = AcconeerA121Radar(seed=self.seed)
        self.controller = TurretController(muzzle_speed_m_s=100.0)
        self.ballistics = ProjectileManager(muzzle_speed_m_s=100.0)
        self._init_drones()
        self.history.clear()
        self.all_hit_events.clear()

    def step(self) -> SimTelemetrySnapshot:
        """
        Advance simulation by one physics time step (dt = 0.01 s).
        """
        # 1. Advance drone targets physics (native JAX Crazyflow or fallback)
        if self.swarm_manager is not None:
            self.swarm_manager.update()
            self.drones = self.swarm_manager.get_drone_views()
        else:
            for drone in self.drones:
                drone.update(self.dt)

        # 2. Get current turret mechanical state
        t_state = self.turret.get_state()

        # 3. Radar scan from turret antenna
        target_positions = [d.position for d in self.drones if d.is_active_threat]
        target_rcs = [d.rcs for d in self.drones if d.is_active_threat]

        radar_result = self.radar.measure(
            turret_pos=self.turret.base_pos,
            turret_pan_deg=t_state.pan_angle_deg,
            turret_tilt_deg=t_state.tilt_angle_deg,
            target_positions=target_positions,
            target_rcs_list=target_rcs,
            apply_noise=True,
        )

        # 4. Pack inputs and feed decoupled turret controller
        c_input = TurretInput(
            timestamp=self.sim_time,
            current_pan_deg=t_state.pan_angle_deg,
            current_tilt_deg=t_state.tilt_angle_deg,
            current_pan_vel_dps=t_state.pan_velocity_dps,
            current_tilt_vel_dps=t_state.tilt_velocity_dps,
            ready_to_fire=t_state.ready_to_fire,
            radar_data=radar_result,
        )

        action = self.controller.update(c_input, dt=self.dt)

        # 5. Apply action to turret servos
        self.turret.command_servos(action.target_pan_deg, action.target_tilt_deg)

        # 6. Fire weapon if commanded and ready
        if action.fire:
            if self.turret.trigger_fire():
                muzzle = self.turret.get_muzzle_position()
                aim_dir = self.turret.get_boresight_vector()
                self.ballistics.spawn_projectile(muzzle, aim_dir)

        # 7. Update active projectiles and check collisions against drones
        step_hits = self.ballistics.update(self.dt, self.drones)
        self.all_hit_events.extend(step_hits)

        # 8. Update turret mechanical integration
        self.turret.update(self.dt)

        # 9. Log telemetry
        n_alive = sum(1 for d in self.drones if d.status == TargetStatus.ALIVE)
        n_hit = sum(1 for d in self.drones if d.status == TargetStatus.HIT)
        n_downed = sum(1 for d in self.drones if d.status == TargetStatus.DOWNED)

        snapshot = SimTelemetrySnapshot(
            sim_time_s=self.sim_time,
            turret_state=t_state,
            radar_data=radar_result,
            controller_action=action,
            drones_alive=n_alive,
            drones_hit=n_hit,
            drones_downed=n_downed,
            active_projectiles=len(self.ballistics.projectiles),
            shots_fired=self.ballistics.total_shots_fired,
            hits_confirmed=self.ballistics.total_hits_confirmed,
        )
        self.history.append(snapshot)
        self.sim_time += self.dt
        return snapshot

    @property
    def all_drones_neutralized(self) -> bool:
        """Check if all simulated targets have been destroyed."""
        return all(d.status == TargetStatus.DOWNED for d in self.drones)
