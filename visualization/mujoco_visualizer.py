"""
MuJoCo 3D Visualizer and Anti-Crash Mouse Input Protection.

Addresses the known Gymnasium/MuJoCo 3.0 mouse wheel scroll bug:
- Patches WindowViewer._scroll_callback to prevent C-level segfaults / crashes when scrolling with mouse wheel.
- Provides interactive visualization via crazyflow Sim.render() and visual markers (based on drone project rendering pipeline).
- Provides offscreen video export of Crazyflie 3D CAD models via mujoco.Renderer + visual markers (turret, radar beam, projectiles) + HUD overlay + ffmpeg pipe.
"""

import shutil
import subprocess
import time
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw

try:
    import mujoco
    import gymnasium.envs.mujoco.mujoco_rendering as mr
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None
    mr = None

from sim.simulation_env import TurretSimulationEnv


def patch_mujoco_viewer_mouse_bugs() -> bool:
    """
    Safely monkey-patches WindowViewer mouse callbacks in gymnasium to be compatible
    with modern MuJoCo 3.0+ C-API (where mjv_moveCamera accepts 5 arguments instead of 6,
    without the legacy 'scn' parameter).
    Fixes both cursor drag (rotation/pan) and mouse wheel scroll (zoom) crashes.
    """
    if not MUJOCO_AVAILABLE or mr is None:
        return False

    try:
        import glfw
    except ImportError:
        return False

    def safe_scroll_callback(self, window, x_offset, y_offset: float):
        try:
            if hasattr(self, "cam") and self.cam is not None:
                clamped_y = float(np.clip(y_offset, -2.0, 2.0))
                # MuJoCo 3.0+ signature: (model, action, reldx, reldy, cam)
                try:
                    mujoco.mjv_moveCamera(
                        self.model,
                        mujoco.mjtMouse.mjMOUSE_ZOOM,
                        0.0,
                        -0.05 * clamped_y,
                        self.cam,
                    )
                except TypeError:
                    # Backward compatibility fallback for MuJoCo < 3.0
                    mujoco.mjv_moveCamera(
                        self.model,
                        mujoco.mjtMouse.mjMOUSE_ZOOM,
                        0.0,
                        -0.05 * clamped_y,
                        self.scn,
                        self.cam,
                    )
        except Exception:
            pass

    def safe_cursor_pos_callback(self, window, xpos: float, ypos: float):
        try:
            if not (self._button_left_pressed or self._button_right_pressed):
                return

            mod_shift = (
                glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
            if self._button_right_pressed:
                action = (
                    mujoco.mjtMouse.mjMOUSE_MOVE_H
                    if mod_shift
                    else mujoco.mjtMouse.mjMOUSE_MOVE_V
                )
            elif self._button_left_pressed:
                action = (
                    mujoco.mjtMouse.mjMOUSE_ROTATE_H
                    if mod_shift
                    else mujoco.mjtMouse.mjMOUSE_ROTATE_V
                )
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            dx = int(self._scale * xpos) - self._last_mouse_x
            dy = int(self._scale * ypos) - self._last_mouse_y
            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                return

            reldx = float(dx / width)
            reldy = float(dy / height)

            # MuJoCo 3.0+ signature: (model, action, reldx, reldy, cam)
            try:
                mujoco.mjv_moveCamera(self.model, action, reldx, reldy, self.cam)
            except TypeError:
                # Backward compatibility fallback for MuJoCo < 3.0
                mujoco.mjv_moveCamera(self.model, action, reldx, reldy, self.scn, self.cam)

            self._last_mouse_x = int(self._scale * xpos)
            self._last_mouse_y = int(self._scale * ypos)
        except Exception:
            pass

    try:
        mr.WindowViewer._scroll_callback = safe_scroll_callback
        mr.WindowViewer._cursor_pos_callback = safe_cursor_pos_callback
        return True
    except Exception:
        return False


# Backward-compatible alias
patch_mujoco_viewer_scroll_bug = patch_mujoco_viewer_mouse_bugs

# Automatically patch on import
patch_mujoco_viewer_mouse_bugs()


class MujocoVisualizer:
    """
    Native MuJoCo visualizer utilizing Crazyflow's underlying physics scene,
    coaxial radar beam projection, 2-DoF turret assembly, target tracking markers,
    and anti-crash mouse wheel scroll protection.
    """
    def __init__(self, env: TurretSimulationEnv, width: int = 640, height: int = 480):
        self.env = env
        self.width = width
        self.height = height
        self.renderer: Optional[mujoco.Renderer] = None
        self.camera: Optional[mujoco.MjvCamera] = None

        # Ensure mouse scroll bug is patched
        patch_mujoco_viewer_scroll_bug()

        if MUJOCO_AVAILABLE and self.env.swarm_manager and self.env.swarm_manager.sim:
            sim = self.env.swarm_manager.sim
            # Clamp width and height to MuJoCo model framebuffer capacity
            fb_w = getattr(sim.mj_model.vis.global_, "offwidth", 640)
            fb_h = getattr(sim.mj_model.vis.global_, "offheight", 480)
            render_w = min(self.width, fb_w)
            render_h = min(self.height, fb_h)
            self.width, self.height = render_w, render_h
            self.renderer = mujoco.Renderer(sim.mj_model, render_h, render_w)

            # Low rear view: turret at the bottom, airspace above it.
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(sim.mj_model, self.camera)
            self._configure_camera(self.camera)
            self._camera_initialized = False

    def _configure_camera(self, camera) -> None:
        """Frame the turret from behind with an upward view of the drones."""
        camera.lookat[:] = np.asarray(self.env.turret.base_pos) + [0.0, 3.8, 2.9]
        camera.distance = 9.0
        camera.elevation = 20.0
        camera.azimuth = 90.0

    def _sync_jax_to_mujoco(self) -> None:
        """
        Synchronize JAX analytical drone states to MuJoCo's C qpos structures.
        Converts quaternion convention from Crazyflow [x, y, z, w] to MuJoCo [w, x, y, z].
        """
        sim = self.env.swarm_manager.sim
        try:
            sim.data = sim.sync_sim2mjx(sim.data)
        except Exception:
            pass
        for i in range(self.env.num_drones):
            pos = np.array(sim.data.states.pos[0, i])
            q_xyzw = np.array(sim.data.states.quat[0, i])
            q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            sim.mj_data.qpos[i * 7 : i * 7 + 3] = pos
            sim.mj_data.qpos[i * 7 + 3 : i * 7 + 7] = q_wxyz
        mujoco.mj_forward(sim.mj_model, sim.mj_data)

    def _populate_custom_geoms(self, scene: mujoco.MjvScene) -> None:
        """
        Populate MuJoCo scene with visual markers matching the post-training rendering
        architecture: turret cylinder base, barrel pointing along boresight, coaxial
        radar beam cone, target highlight spheres, and ballistic tracer rounds.
        """
        def add_geom(geom_type, size, pos, rot_mat, rgba):
            if scene.ngeom < scene.maxgeom:
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g, geom_type,
                    np.array(size, dtype=np.float64),
                    np.array(pos, dtype=np.float64),
                    np.array(rot_mat, dtype=np.float64).flatten(),
                    np.array(rgba, dtype=np.float32)
                )
                scene.ngeom += 1

        # 1. Turret Base
        base_pos = np.array(self.env.turret.base_pos)
        add_geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.22, 0.22, 0.12], base_pos, np.eye(3), [0.2, 0.5, 0.8, 1.0])

        # 2. Turret Barrel & Coaxial Radar Mount
        boresight = self.env.turret.get_boresight_vector()
        norm_b = np.linalg.norm(boresight)
        if norm_b > 1e-6:
            z_axis = boresight / norm_b
            x_axis = np.cross(np.array([0, 0, 1]), z_axis)
            if np.linalg.norm(x_axis) < 1e-4:
                x_axis = np.array([1, 0, 0])
            else:
                x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])

            # Barrel
            barrel_center = base_pos + [0, 0, 0.15] + z_axis * 0.35
            add_geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.04, 0.04, 0.35], barrel_center, rot_mat, [0.95, 0.75, 0.15, 1.0])

            # Radar Beam Cone (Translucent capsule along boresight)
            beam_center = base_pos + [0, 0, 0.15] + z_axis * 2.5
            add_geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [0.35, 0.35, 2.5], beam_center, rot_mat, [0.0, 0.85, 1.0, 0.22])

        # 3. Target Highlight Spheres around drones
        sim = self.env.swarm_manager.sim
        for i in range(self.env.num_drones):
            d_pos = np.array(sim.data.states.pos[0, i])
            is_down = bool(sim.data.states.pos[0, i, 2] <= 0.25)
            color = [0.2, 0.85, 0.2, 0.5] if not is_down else [0.85, 0.2, 0.2, 0.5]
            add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.15, 0.15, 0.15], d_pos, np.eye(3), color)

        # 4. Ballistic Projectiles (Tracer bullets)
        for proj in self.env.ballistics.projectiles:
            if proj.alive:
                add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0.05, 0.05], proj.position, np.eye(3), [1.0, 0.3, 0.1, 1.0])

    def _draw_hud_overlay(self, pixels: np.ndarray) -> np.ndarray:
        """
        Overlay a tactical HUD telemetry card onto the MuJoCo render using PIL.
        """
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img, "RGBA")

        # Semi-transparent HUD card in upper left
        draw.rectangle([(10, 10), (330, 140)], fill=(12, 18, 30, 215), outline=(0, 200, 255, 255), width=1)

        last_radar = self.env.history[-1].radar_data if self.env.history else None
        eco_count = last_radar.num_distances if last_radar else 0
        pan_pwm = self.env.turret.pan_servo.angle_to_pulse_us(self.env.turret.pan_servo.current_angle_deg)
        tilt_pwm = self.env.turret.tilt_servo.angle_to_pulse_us(self.env.turret.tilt_servo.current_angle_deg)

        hud_lines = [
            f"SIM TIME: {self.env.sim_time:05.2f} s",
            f"FSM MODE: {self.env.controller.mode.name}",
            f"PAN : {self.env.turret.pan_servo.current_angle_deg:+5.1f} deg | PWM: {pan_pwm:.0f} us",
            f"TILT: {self.env.turret.tilt_servo.current_angle_deg:+5.1f} deg | PWM: {tilt_pwm:.0f} us",
            f"RADAR A121: {eco_count} ecos detectados",
            f"OBJETIVOS: {self.env.num_drones} Drones (Crazyflow JAX)",
            f"BALISTICA: {self.env.ballistics.total_shots_fired} disparos | {len(self.env.all_hit_events)} impactos"
        ]
        y = 15
        for line in hud_lines:
            draw.text((18, y), line, fill=(210, 240, 255, 255))
            y += 17

        return np.array(img)

    def render_interactive(self) -> None:
        """
        Step one frame in the interactive MuJoCo GUI window with anti-crash protection
        and visual markers matching post-training rendering scripts.
        """
        if self.env.swarm_manager and self.env.swarm_manager.sim:
            sim = self.env.swarm_manager.sim
            self._sync_jax_to_mujoco()

            # Interactive viewer with markers
            if sim.viewer is None:
                sim.render()
            viewer = sim.viewer
            if viewer is not None and hasattr(viewer, 'viewer'):
                v = viewer.viewer
                if hasattr(v, '_markers'):
                    v._markers.clear()

                # Auto-initialize tactical camera on first render
                if not getattr(self, "_camera_initialized", False) and hasattr(v, "cam") and v.cam is not None:
                    self._configure_camera(v.cam)
                    self._camera_initialized = True

                # 1. Turret Base
                base_pos = np.array(self.env.turret.base_pos)
                v.add_marker(pos=base_pos, size=np.array([0.22, 0.22, 0.12]), rgba=np.array([0.2, 0.5, 0.8, 1.0]), type=mujoco.mjtGeom.mjGEOM_CYLINDER, label='Turret')

                # 2. Turret Barrel & Coaxial Radar Mount
                boresight = self.env.turret.get_boresight_vector()
                norm_b = np.linalg.norm(boresight)
                if norm_b > 1e-6:
                    z_axis = boresight / norm_b
                    x_axis = np.cross(np.array([0, 0, 1]), z_axis)
                    if np.linalg.norm(x_axis) < 1e-4:
                        x_axis = np.array([1, 0, 0])
                    else:
                        x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

                    barrel_center = base_pos + [0, 0, 0.15] + z_axis * 0.35
                    v.add_marker(pos=barrel_center, size=np.array([0.04, 0.04, 0.35]), mat=rot_mat.flatten(), rgba=np.array([0.95, 0.75, 0.15, 1.0]), type=mujoco.mjtGeom.mjGEOM_CYLINDER, label='Barrel')

                    beam_center = base_pos + [0, 0, 0.15] + z_axis * 2.5
                    v.add_marker(pos=beam_center, size=np.array([0.35, 0.35, 2.5]), mat=rot_mat.flatten(), rgba=np.array([0.0, 0.85, 1.0, 0.25]), type=mujoco.mjtGeom.mjGEOM_CAPSULE, label='Radar A121')

                # 3. Drone status markers
                for i in range(self.env.num_drones):
                    d_pos = np.array(sim.data.states.pos[0, i])
                    is_down = bool(sim.data.states.pos[0, i, 2] <= 0.25)
                    color = [0.2, 0.85, 0.2, 0.5] if not is_down else [0.85, 0.2, 0.2, 0.5]
                    v.add_marker(pos=d_pos, size=np.array([0.15, 0.15, 0.15]), rgba=np.array(color), type=mujoco.mjtGeom.mjGEOM_SPHERE, label=f'Dron {i}')

                # 4. Ballistic Projectiles (Tracer bullets)
                for proj in self.env.ballistics.projectiles:
                    if proj.alive:
                        v.add_marker(pos=proj.position, size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1.0, 0.3, 0.1, 1.0]), type=mujoco.mjtGeom.mjGEOM_SPHERE, label='Bullet')

            sim.render()

    def run_interactive(self, duration_s: float = 30.0, fps: float = 25.0) -> None:
        """
        Run real-time interactive simulation loop in the MuJoCo GUI window with anti-crash protection.
        """
        steps = int(duration_s / self.env.dt)
        dt_frame = 1.0 / fps
        print(f"=== [MuJoCo Interactive] Abriendo ventana 3D interactiva ({duration_s:.1f} s a {fps:.0f} FPS) ===")
        print("  • Botón izquierdo: Rotar cámara")
        print("  • Botón derecho:   Trasladar cámara")
        print("  • Rueda ratón:     Zoom seguro (anti-crash protegido)")
        print("  • Cierre la ventana o pulse Ctrl+C para finalizar.\n")

        start_time = time.time()
        for step_i in range(steps):
            t_step_start = time.time()
            s = self.env.step()
            self.render_interactive()

            # Console live telemetry every 1.0s or on hits
            if step_i % 25 == 0 or s.hits_confirmed > 0 and step_i % 5 == 0:
                r_info = f"{s.radar_data.num_distances} ecos ({s.radar_data.distances} m)" if s.radar_data.num_distances > 0 else "0 ecos"
                print(f"[t={s.sim_time_s:05.2f}s] FSM: {s.controller_action.status_mode:<12} | "
                      f"Pan={s.turret_state.pan_angle_deg:+5.1f}°, Tilt={s.turret_state.tilt_angle_deg:+5.1f}° | "
                      f"Radar: {r_info:<20} | Disparos: {s.shots_fired:2d} | Impactos: {s.hits_confirmed:2d} | "
                      f"Abatidos: {s.drones_downed}/{self.env.num_drones}")

            if self.env.all_drones_neutralized:
                print(f"\n[MISIÓN CUMPLIDA] Todos los drones abatidos en t={s.sim_time_s:.2f} s!")
                break

            # Frame rate limiter for real-time perception
            elapsed = time.time() - t_step_start
            sleep_time = dt_frame - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

        total_elapsed = time.time() - start_time
        print(f"\n[MuJoCo Interactive] Simulación finalizada. Duración real: {total_elapsed:.2f} s.")

    def export_mp4(self, steps: int = 300, filename: str = "mujoco_recording.mp4", fps: int = 25) -> str:
        """
        Render native MuJoCo frames offscreen and pipe directly into ffmpeg.
        Renders Crazyflie CAD models, turret assembly, boresight radar beam,
        projectile tracers, and tactical HUD. Immune to GLFW mouse events and zero RAM accumulation.
        """
        if self.renderer is None:
            raise RuntimeError("MuJoCo is not available or Crazyflow swarm is not initialized.")

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg is required for MP4 video export.")

        print(f"[MuJoCo Visualizer] Renderizando {steps} frames en MuJoCo offscreen -> {filename}...")

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-crf", "19",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            filename,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for step_i in range(steps):
            self.env.step()

            # Sync JAX analytical states to MuJoCo qpos
            self._sync_jax_to_mujoco()

            # Update MuJoCo scene with custom tactical camera
            sim = self.env.swarm_manager.sim
            self.renderer.update_scene(sim.mj_data, camera=self.camera)

            # Populate visual markers
            self._populate_custom_geoms(self.renderer.scene)

            # Offscreen render
            pixels = self.renderer.render()

            # Draw tactical HUD overlay
            framed_pixels = self._draw_hud_overlay(pixels)

            # Pipe raw pixels to ffmpeg
            proc.stdin.write(framed_pixels.tobytes())

            if step_i % 50 == 0 or step_i == steps - 1:
                print(f"  Frame {step_i + 1}/{steps} procesado...")

        proc.stdin.close()
        proc.wait()
        print(f"[MuJoCo Visualizer] Video MP4 exportado con éxito: {filename}")
        return filename
