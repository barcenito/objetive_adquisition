"""
Native JAX / Crazyflow Swarm Physics Integration for Aerial Drone Targets.

Leverages the high-performance crazyflow simulator:
- Batch quadcopter dynamics simulated in JAX (crazyflow.sim.Sim)
- Internal state/attitude cascading flight controllers (Control.state)
- Realistic rotor thrust, aerodynamic drag, and mass/inertia parameters
- Injection of physical ballistic impact impulses directly into JAX state arrays
- Seamless extraction of 3D positions and velocities for Acconeer A121 radar & ballistics.
"""

from typing import List, Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from crazyflow.sim.sim import Sim
    from crazyflow.sim.physics import Physics
    from crazyflow.control.control import Control
    CRAZYFLOW_AVAILABLE = True
except ImportError:
    CRAZYFLOW_AVAILABLE = False
    Sim = None
    Physics = None
    Control = None

from targets.drone_target import TargetStatus
from targets.trajectory_gen import TrajectoryGenerator


class CrazyflowSwarmManager:
    """
    Manages 1 to 3 dynamic aerial drone targets driven by the Crazyflow JAX physics engine.
    """
    def __init__(
        self,
        num_drones: int = 3,
        drone_speeds: Optional[List[float]] = None,
        physics_freq_hz: float = 100.0,
        seed: Optional[int] = 42,
        bounds_min: Tuple[float, float, float] = (-5.0, 1.8, 0.8),
        bounds_max: Tuple[float, float, float] = (5.0, 8.5, 3.5),
    ):
        self.num_drones = min(max(1, num_drones), 3)
        self.dt = 1.0 / max(physics_freq_hz, 10.0)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        default_speeds = [1.8, 3.2, 4.6]
        self.drone_speeds = (drone_speeds[:self.num_drones] if drone_speeds is not None
                             else default_speeds[:self.num_drones])

        # Flight boundaries and trajectory generators for each drone
        self.bounds_min = np.array(bounds_min, dtype=np.float64)
        self.bounds_max = np.array(bounds_max, dtype=np.float64)
        self.traj_gens = [
            TrajectoryGenerator(flight_bounds_min=bounds_min, flight_bounds_max=bounds_max,
                                seed=((seed + i * 137) if seed is not None else None))
            for i in range(self.num_drones)
        ]

        # Pre-generate waypoints for smooth B-splines
        self.waypoints = [tg.generate_random_waypoints(8) for tg in self.traj_gens]
        self.phases = [float(tg.rng.uniform(0.0, 1.0)) for tg in self.traj_gens]
        self.path_lengths = []
        for wps in self.waypoints:
            dists = [np.linalg.norm(wps[k+1] - wps[k]) for k in range(len(wps) - 1)]
            self.path_lengths.append(max(sum(dists), 10.0))

        # Status and properties for each drone
        self.statuses = [TargetStatus.ALIVE for _ in range(self.num_drones)]
        self.rcs_values = [0.03 for _ in range(self.num_drones)]
        self.bounding_radii = [0.18 for _ in range(self.num_drones)]
        self.drone_mass_kg = 0.035  # Crazyflie 2.1 nominal mass (~35 grams)
        self.time_since_hits = [0.0 for _ in range(self.num_drones)]

        # Initial 3D spawn positions
        self.positions = np.zeros((self.num_drones, 3), dtype=np.float64)
        self.velocities = np.zeros((self.num_drones, 3), dtype=np.float64)
        for i in range(self.num_drones):
            pos, dir_vec = self.traj_gens[i].evaluate_catmull_rom_spline(self.waypoints[i], self.phases[i])
            self.positions[i] = pos
            self.velocities[i] = dir_vec * self.drone_speeds[i]

        # Initialize Crazyflow JAX Sim
        self.sim: Optional[Sim] = None
        if CRAZYFLOW_AVAILABLE:
            self._init_crazyflow()

    def _init_crazyflow(self) -> None:
        """Instantiate and prime the batched Crazyflow simulation in JAX."""
        self.sim = Sim(
            n_worlds=1,
            n_drones=self.num_drones,
            physics=Physics.analytical,
            control=Control.state,
            freq=int(1.0 / self.dt),
            state_freq=int(1.0 / self.dt),
            attitude_freq=int(1.0 / self.dt),
            thrust_freq=int(1.0 / self.dt),
            device="cpu",
        )

        # Set initial positions in JAX states
        new_states = self.sim.data.states.replace(
            pos=jnp.array(self.positions).reshape(1, self.num_drones, 3),
            vel=jnp.array(self.velocities).reshape(1, self.num_drones, 3),
        )
        self.sim.data = self.sim.data.replace(states=new_states)
        self.sim.default_data = self.sim.data.replace()

    def apply_hit(self, target_id: int, impact_impulse: Optional[np.ndarray] = None) -> None:
        """
        Inject a kinetic impact perturbation into a drone target.
        Updates JAX state velocity directly and transitions status to HIT/downward spiral.
        """
        if target_id < 0 or target_id >= self.num_drones:
            return
        if self.statuses[target_id] == TargetStatus.DOWNED:
            return

        self.statuses[target_id] = TargetStatus.HIT
        self.time_since_hits[target_id] = 0.0

        if impact_impulse is None:
            # Default chaotic perturbation
            impulse = self.rng.normal(0.0, 0.05, size=3)
            impulse[2] = -0.08  # Downward hit
        else:
            impulse = np.array(impact_impulse, dtype=np.float64)

        delta_v = impulse / self.drone_mass_kg

        # 1. Update native JAX simulation state if active
        if self.sim is not None and CRAZYFLOW_AVAILABLE:
            curr_vel = np.array(self.sim.data.states.vel)
            curr_vel[0, target_id] += delta_v
            new_states = self.sim.data.states.replace(
                vel=jnp.array(curr_vel)
            )
            self.sim.data = self.sim.data.replace(states=new_states)

        # 2. Local cache update
        self.velocities[target_id] += delta_v

    def update(self) -> None:
        """
        Advance drone swarm physics by one timestep dt using JAX / Crazyflow.
        """
        # Build command array for state control: shape (1, num_drones, 13)
        cmd = np.zeros((1, self.num_drones, 13), dtype=np.float64)

        for i in range(self.num_drones):
            status = self.statuses[i]

            if status == TargetStatus.ALIVE:
                # Advance phase along 3D B-spline trajectory
                speed = self.drone_speeds[i]
                path_len = self.path_lengths[i]
                self.phases[i] = (self.phases[i] + (speed * self.dt) / path_len) % 1.0

                desired_pos, desired_dir = self.traj_gens[i].evaluate_catmull_rom_spline(
                    self.waypoints[i], self.phases[i]
                )
                desired_vel = desired_dir * speed
                # Terrain and boundary safety steering
                steered_vel = self.traj_gens[i].apply_boundary_steering(self.positions[i], desired_vel)

                cmd[0, i, :3] = desired_pos
                cmd[0, i, 3:6] = steered_vel

            elif status == TargetStatus.HIT:
                # Damaged / crippled flight: gravity spiral down to ground
                self.time_since_hits[i] += self.dt
                # Command drone down into ground
                crash_pos = np.copy(self.positions[i])
                crash_pos[2] = 0.0
                cmd[0, i, :3] = crash_pos
                cmd[0, i, 3:6] = [0.0, 0.0, -3.0]

            elif status == TargetStatus.DOWNED:
                # Grounded target
                cmd[0, i, :3] = [self.positions[i][0], self.positions[i][1], 0.0]
                cmd[0, i, 3:6] = [0.0, 0.0, 0.0]

        # Step Crazyflow JAX physics
        if self.sim is not None and CRAZYFLOW_AVAILABLE:
            self.sim.state_control(cmd)
            self.sim.step(1)
            # Read back state from JAX arrays
            self.positions = np.array(self.sim.data.states.pos[0])
            self.velocities = np.array(self.sim.data.states.vel[0])
        else:
            # Fallback kinematics if JAX/Crazyflow is unavailable
            for i in range(self.num_drones):
                if self.statuses[i] == TargetStatus.ALIVE:
                    self.velocities[i] += (cmd[0, i, 3:6] - self.velocities[i]) * 0.1
                    self.positions[i] += self.velocities[i] * self.dt
                elif self.statuses[i] == TargetStatus.HIT:
                    self.velocities[i] += np.array([0.0, 0.0, -9.81]) * self.dt
                    self.positions[i] += self.velocities[i] * self.dt

        # Check ground impacts for HIT drones
        for i in range(self.num_drones):
            if self.statuses[i] == TargetStatus.HIT:
                if self.positions[i][2] <= 0.06:
                    self.positions[i][2] = 0.0
                    self.velocities[i] = np.zeros(3)
                    self.statuses[i] = TargetStatus.DOWNED
            elif self.statuses[i] == TargetStatus.DOWNED:
                self.positions[i][2] = 0.0
                self.velocities[i] = np.zeros(3)

    def get_drone_views(self) -> list:
        """
        Return target proxy objects compatible with the radar and ballistic systems.
        """
        proxies = []
        for i in range(self.num_drones):
            p = _DroneTargetProxy(
                target_id=i,
                speed=self.drone_speeds[i],
                position=self.positions[i],
                velocity=self.velocities[i],
                status=self.statuses[i],
                rcs=self.rcs_values[i],
                radius=self.bounding_radii[i],
                swarm_mgr=self,
            )
            proxies.append(p)
        return proxies


class _DroneTargetProxy:
    """
    Lightweight interface proxy exposing DroneTarget properties to radar & ballistics.
    """
    def __init__(self, target_id: int, speed: float, position: np.ndarray,
                 velocity: np.ndarray, status: TargetStatus, rcs: float, radius: float,
                 swarm_mgr: CrazyflowSwarmManager):
        self.target_id = target_id
        self.speed = float(speed)
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.status = status
        self.rcs = float(rcs)
        self.radius = float(radius)
        self._mgr = swarm_mgr

    def apply_hit(self, impact_impulse: Optional[np.ndarray] = None) -> None:
        self._mgr.apply_hit(self.target_id, impact_impulse)
        self.status = self._mgr.statuses[self.target_id]

    @property
    def is_active_threat(self) -> bool:
        return self.status != TargetStatus.DOWNED
