"""
High-Fidelity 3D Simulation Visualizer & Tactical Telemetry HUD.

Uses Matplotlib 3D with tactical dark aesthetic:
- Turret base and orientable barrel assembly
- 3D Radar radiation beam cone (53 deg Horizontal x 65 deg Vertical)
- Drones with trail history and status-dependent color coding
- Active ballistic tracer projectiles
- Telemetry HUD with servo angles, A121 radar detections, and engagement metrics
- Support for interactive rendering and animated GIF / video export.
"""

import os
import shutil
import subprocess
from typing import List, Optional
import numpy as np
import matplotlib
# Use Agg backend by default for headless compatibility, or TkAgg/Qt if available
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.animation as animation
except ImportError:
    plt = None

from sim.simulation_env import TurretSimulationEnv
from targets.drone_target import TargetStatus
from core.coordinates import get_turret_basis


class SimulationVisualizer3D:
    def __init__(self, env: TurretSimulationEnv, figsize=(10, 6.5)):
        self.env = env
        self.fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#0d1117")

        # Trail memory for each drone
        self.trail_history = {d.target_id: [] for d in self.env.drones}
        self.max_trail_len = 50

    def _draw_radar_cone(self, turret_pos: np.ndarray, pan_deg: float, tilt_deg: float, range_m: float = 6.0):
        """Draw 3D elliptical wireframe cone representing Acconeer A121 radar beam."""
        forward, right, up = get_turret_basis(pan_deg, tilt_deg)
        fov_h_rad = np.radians(self.env.radar.fov_h / 2.0)
        fov_v_rad = np.radians(self.env.radar.fov_v / 2.0)

        # Perimeter of ellipse at distance range_m
        num_pts = 24
        angles = np.linspace(0, 2 * np.pi, num_pts)
        pts = []
        for theta in angles:
            dx = range_m * np.tan(fov_h_rad) * np.cos(theta)
            dy = range_m * np.tan(fov_v_rad) * np.sin(theta)
            pt = turret_pos + forward * range_m + right * dx + up * dy
            pts.append(pt)

        pts = np.array(pts)
        # Draw perimeter loop
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#00ffcc", alpha=0.35, linewidth=1.2)

        # Draw 4 key boundary rays from apex to perimeter
        for idx in [0, num_pts // 4, num_pts // 2, (3 * num_pts) // 4]:
            self.ax.plot(
                [turret_pos[0], pts[idx, 0]],
                [turret_pos[1], pts[idx, 1]],
                [turret_pos[2], pts[idx, 2]],
                color="#00ffcc", alpha=0.25, linestyle="--", linewidth=1.0
            )

    def draw_frame(self, snapshot) -> None:
        """Render a single 3D simulation frame."""
        self.ax.clear()
        self.fig.texts.clear()  # Clear previous HUD texts to prevent Matplotlib memory leak
        self.ax.set_facecolor("#0d1117")

        # Arena bounds
        self.ax.set_xlim([-6.0, 6.0])
        self.ax.set_ylim([0.0, 9.0])
        self.ax.set_zlim([0.0, 4.5])

        # Grid and axis styling
        self.ax.grid(True, color="#21262d", linestyle=":", alpha=0.6)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.set_xlabel("X (East) [m]", color="#58a6ff", fontsize=9)
        self.ax.set_ylabel("Y (North) [m]", color="#58a6ff", fontsize=9)
        self.ax.set_zlabel("Z (Altitude) [m]", color="#58a6ff", fontsize=9)

        # Ground plane grid at z=0
        xx, yy = np.meshgrid(np.linspace(-6, 6, 7), np.linspace(0, 9, 7))
        zz = np.zeros_like(xx)
        self.ax.plot_wireframe(xx, yy, zz, color="#30363d", alpha=0.3, linewidth=0.8)

        # 1. Draw Turret Base and Barrel
        t_state = snapshot.turret_state
        base_pos = self.env.turret.base_pos
        muzzle_pos = t_state.muzzle_position

        # Base platform
        self.ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], color="#79c0ff", s=120, marker="o", label="Turret Base")
        # Barrel line
        self.ax.plot(
            [base_pos[0], muzzle_pos[0]],
            [base_pos[1], muzzle_pos[1]],
            [base_pos[2], muzzle_pos[2]],
            color="#58a6ff", linewidth=4.0, solid_capstyle="round"
        )

        # 2. Draw A121 Radar Beam Cone
        self._draw_radar_cone(base_pos, t_state.pan_angle_deg, t_state.tilt_angle_deg)

        # 3. Draw Drones and Trails
        drone_colors = {
            TargetStatus.ALIVE: "#3fb950",   # Green
            TargetStatus.HIT: "#d29922",     # Orange
            TargetStatus.DOWNED: "#f85149",  # Red
        }

        for drone in self.env.drones:
            pos = drone.position
            # Update trail
            self.trail_history[drone.target_id].append(np.copy(pos))
            if len(self.trail_history[drone.target_id]) > self.max_trail_len:
                self.trail_history[drone.target_id].pop(0)

            trail = np.array(self.trail_history[drone.target_id])
            color = drone_colors[drone.status]

            # Trail line
            if len(trail) > 1:
                self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, alpha=0.4, linewidth=1.5)

            # Drone Body & Bounding Sphere
            marker = "o" if drone.status != TargetStatus.DOWNED else "x"
            size = 90 if drone.status != TargetStatus.DOWNED else 120
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=size, marker=marker, edgecolors="white", linewidths=1.2)

            # Label
            label_text = f"Dron {drone.target_id} ({drone.speed:.1f} m/s) [{drone.status.value.upper()}]"
            self.ax.text(pos[0] + 0.15, pos[1] + 0.15, pos[2] + 0.15, label_text, color=color, fontsize=8, weight="bold")

        # 4. Draw Ballistic Projectiles
        for proj in self.env.ballistics.projectiles:
            if proj.alive:
                self.ax.scatter([proj.position[0]], [proj.position[1]], [proj.position[2]], color="#ffa657", s=30, marker="*")
                # Mini tracer behind bullet
                v_back = proj.position - proj.velocity * 0.02
                self.ax.plot([v_back[0], proj.position[0]], [v_back[1], proj.position[1]], [v_back[2], proj.position[2]], color="#ff7b72", linewidth=2.0)

        # 5. Draw Recent Hit Sparks
        for hit in self.env.all_hit_events[-3:]:
            self.ax.scatter([hit.hit_position[0]], [hit.hit_position[1]], [hit.hit_position[2]], color="#ffd700", s=180, marker="*")

        # 6. HUD Telemetry Overlay
        radar = snapshot.radar_data
        r_str = f"Dist: {radar.distances} m | Pwr: {radar.strengths} dB" if radar.num_distances > 0 else "SIN CONTACTOS"

        hud_lines = [
            f"SIM TIME: {snapshot.sim_time_s:05.2f} s",
            f"FSM MODE: {snapshot.controller_action.status_mode}",
            f"PAN SERVO : {t_state.pan_angle_deg:+.1f}° ({t_state.pan_pwm_us:.0f} µs)",
            f"TILT SERVO: {t_state.tilt_angle_deg:+.1f}° ({t_state.tilt_pwm_us:.0f} µs)",
            f"RADAR A121 (60 GHz): {radar.num_distances} ECO(S) | {r_str}",
            f"OBJETIVOS : {snapshot.drones_alive} Vivos | {snapshot.drones_hit} Averiado | {snapshot.drones_downed} Derribados",
            f"BALISTICA : {snapshot.shots_fired} Disparos | {snapshot.hits_confirmed} Impactos Confirmados",
        ]
        hud_text = "\n".join(hud_lines)
        self.fig.text(
            0.02, 0.98, hud_text,
            transform=self.fig.transFigure,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            color="#e6edf3",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22", edgecolor="#30363d", alpha=0.92)
        )

    def save_animation(self, steps: int = 300, filename: str = "simulation_recording.mp4", fps: int = 25) -> str:
        """
        Run simulation and export animation to MP4 (via streaming ffmpeg) or GIF.
        Using direct ffmpeg pipe streaming avoids accumulating frames in RAM, preventing OOM crashes.
        """
        is_mp4 = filename.lower().endswith(".mp4")
        ffmpeg_bin = shutil.which("ffmpeg")

        print(f"[Visualizer] Grabando animación ({steps} pasos a {fps} FPS -> {filename})...")

        # Test draw first frame to get exact canvas pixel dimensions
        snapshot = self.env.step()
        self.draw_frame(snapshot)
        self.fig.canvas.draw()
        rgba_sample = np.asarray(self.fig.canvas.buffer_rgba())
        height, width, _ = rgba_sample.shape

        # Ensure dimensions are even for ffmpeg libx264
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        use_ffmpeg_stream = bool(ffmpeg_bin is not None)

        if use_ffmpeg_stream:
            if is_mp4:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgba",
                    "-r", str(fps),
                    "-i", "-",
                    "-an",
                    "-vcodec", "libx264",
                    "-crf", "19",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    filename,
                ]
            else:
                # Direct palettegen + paletteuse filter for pristine, compact GIF
                cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgba",
                    "-r", str(fps),
                    "-i", "-",
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                    filename,
                ]

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            # Write initial frame
            proc.stdin.write(rgba_sample[:height, :width].tobytes())

            for step_i in range(1, steps):
                snapshot = self.env.step()
                self.draw_frame(snapshot)
                self.fig.canvas.draw()
                rgba = np.asarray(self.fig.canvas.buffer_rgba())
                proc.stdin.write(rgba[:height, :width].tobytes())
                if step_i % 50 == 0 or step_i == steps - 1:
                    print(f"  Paso {step_i + 1}/{steps} procesado...")

            proc.stdin.close()
            proc.wait()
            print(f"[Visualizer] Archivo multimedia generado con éxito (vía ffmpeg stream): {filename}")
        else:
            # Fallback for systems without ffmpeg: PIL in-memory (limited batch)
            from PIL import Image
            frames = [Image.fromarray(rgba_sample[:height, :width])]
            for step_i in range(1, steps):
                snapshot = self.env.step()
                self.draw_frame(snapshot)
                self.fig.canvas.draw()
                rgba = np.asarray(self.fig.canvas.buffer_rgba())
                frames.append(Image.fromarray(rgba[:height, :width]))
                if step_i % 50 == 0:
                    print(f"  Paso {step_i + 1}/{steps} procesado...")

            duration_ms = int(1000.0 / fps)
            gif_filename = filename if filename.endswith(".gif") else filename + ".gif"
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
            print(f"[Visualizer] Animación GIF guardada con éxito (vía PIL): {gif_filename}")
            filename = gif_filename

        plt.close(self.fig)
        return filename

    def save_animation_gif(self, steps: int = 300, filename: str = "simulation_recording.gif", fps: int = 25):
        """Backward-compatible wrapper for save_animation."""
        return self.save_animation(steps=steps, filename=filename, fps=fps)
