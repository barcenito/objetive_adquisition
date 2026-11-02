#!/usr/bin/env python3
"""
Simulador de Adquisición de Objetivos y Derribo (Torreta 2-DoF & Radar Acconeer A121).

Permite ejecutar:
1. Modo Autónomo (Radar A121 + Algoritmo de Búsqueda, Rastreo y Disparo)
2. Modo Manual de Prueba de Movimientos de la Torreta (--manual-test --pan X --tilt Y)
3. Generación de animación grabada en GIF / imágenes
4. Pruebas de estrés y benchmarking headless de escenarios (1, 2 o 3 drones).
"""

import argparse
import sys
import time
import numpy as np

from sim.simulation_env import TurretSimulationEnv
from sim.scenarios import SCENARIOS
from targets.drone_target import TargetStatus
from visualization.visualizer_3d import SimulationVisualizer3D


def run_headless(env: TurretSimulationEnv, duration_s: float = 10.0, verbose: bool = True):
    steps = int(duration_s / env.dt)
    if verbose:
        print(f"=== Iniciando Simulación Headless ({duration_s:.1f} s, {steps} pasos a {1.0/env.dt:.0f} Hz) ===")
        print(f"Objetivos: {env.num_drones} drones con velocidades nominales: {env.drone_speeds} m/s")
        print("-" * 75)

    start_real = time.time()
    last_print = 0.0

    for step in range(steps):
        s = env.step()

        # Log periodic updates or hit events
        if verbose and (s.sim_time_s - last_print >= 1.0 or s.hits_confirmed > 0 and s.sim_time_s - last_print >= 0.2):
            last_print = s.sim_time_s
            r_info = f"{s.radar_data.num_distances} ecos ({s.radar_data.distances} m)" if s.radar_data.num_distances > 0 else "0 ecos"
            print(f"t={s.sim_time_s:05.2f}s | Modo: {s.controller_action.status_mode:<15} | "
                  f"Torreta: Pan={s.turret_state.pan_angle_deg:+5.1f}°, Tilt={s.turret_state.tilt_angle_deg:+5.1f}° | "
                  f"Radar: {r_info:<22} | Disparos: {s.shots_fired:2d} | Impactos: {s.hits_confirmed:2d} | "
                  f"Drones: {s.drones_downed}/{env.num_drones} abatidos")

        if env.all_drones_neutralized:
            if verbose:
                print(f"\n[MISIÓN CUMPLIDA] Todos los objetivos han sido derribados en t={s.sim_time_s:.2f} s!")
            break

    elapsed_real = time.time() - start_real
    sim_rate = (s.sim_time_s / max(elapsed_real, 1e-4))

    if verbose:
        print("=" * 75)
        print("=== RESUMEN FINAL DE LA SIMULACIÓN ===")
        print(f"Tiempo simulado: {s.sim_time_s:.2f} s (Tiempo real de cómputo: {elapsed_real:.2f} s | {sim_rate:.1f}x tiempo real)")
        print(f"Disparos efectuados : {s.shots_fired}")
        print(f"Impactos confirmados: {s.hits_confirmed}")
        accuracy = (s.hits_confirmed / max(s.shots_fired, 1)) * 100.0
        print(f"Precisión de tiro   : {accuracy:.1f} %")
        print(f"Drones derribados   : {s.drones_downed} de {env.num_drones}")
        for d in env.drones:
            print(f"  • Dron {d.target_id} (Vel: {d.speed} m/s) -> Estado: {d.status.value.upper()} (Pos: {np.round(d.position, 2)})")
        print("=" * 75)

    return s


def main():
    parser = argparse.ArgumentParser(description="Simulador de Adquisición de Objetivos y Derribo (Torreta 2-DoF & Radar A121)")
    parser.add_argument("--scenario", type=int, default=3, choices=[1, 2, 3],
                        help="Escenario a simular (1: 1 dron, 2: 2 drones, 3: 3 drones)")
    parser.add_argument("--duration", type=float, default=12.0, help="Duración de la simulación en segundos")
    parser.add_argument("--manual-test", action="store_true", help="Activar modo de prueba manual de movimientos de torreta")
    parser.add_argument("--pan", type=float, default=30.0, help="Ángulo pan manual (grados) en modo prueba")
    parser.add_argument("--tilt", type=float, default=25.0, help="Ángulo tilt manual (grados) en modo prueba")
    parser.add_argument("--fire", action="store_true", help="Activar disparo continuo en modo prueba")
    parser.add_argument("--gui", "--interactive", action="store_true", dest="gui",
                        help="Ejecutar simulación en tiempo real en ventana 3D interactiva de MuJoCo")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS objetivo en modo interactivo")
    parser.add_argument("--record-video", type=str, default="", help="Ruta de archivo para exportar video HUD táctico MP4 (ej. simulation.mp4)")
    parser.add_argument("--record-mujoco", type=str, default="", help="Ruta de archivo para exportar video nativo 3D MuJoCo MP4 (ej. mujoco.mp4)")
    parser.add_argument("--record-gif", type=str, default="", help="Ruta de archivo para guardar animación GIF (ej. anim.gif)")
    parser.add_argument("--frames", type=int, default=250, help="Número de frames para la animación GIF o video")
    parser.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria")

    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    print(f"Seleccionado {cfg['name']}: {cfg['description']}")

    env = TurretSimulationEnv(
        num_drones=cfg["num_drones"],
        drone_speeds=cfg["drone_speeds"],
        seed=args.seed,
    )

    if args.manual_test:
        print(f"\n[MODO PRUEBA DE MOVIMIENTO] Comandando torreta a Pan={args.pan}° y Tilt={args.tilt}° (Disparo={args.fire})...")
        env.controller.set_manual_test_mode(pan_deg=args.pan, tilt_deg=args.tilt, fire=args.fire)

    if args.gui:
        from visualization.mujoco_visualizer import MujocoVisualizer
        vis_mj = MujocoVisualizer(env, width=800, height=600)
        vis_mj.run_interactive(duration_s=args.duration, fps=args.fps)
    elif args.record_mujoco:
        from visualization.mujoco_visualizer import MujocoVisualizer
        vis_mj = MujocoVisualizer(env, width=640, height=480)
        vis_mj.export_mp4(steps=args.frames, filename=args.record_mujoco, fps=25)
    elif args.record_video:
        vis = SimulationVisualizer3D(env)
        vis.save_animation(steps=args.frames, filename=args.record_video, fps=25)
    elif args.record_gif:
        vis = SimulationVisualizer3D(env)
        vis.save_animation(steps=args.frames, filename=args.record_gif, fps=25)
    else:
        run_headless(env, duration_s=args.duration, verbose=True)


if __name__ == "__main__":
    main()
