# Plan de Implementación Actualizado: Sistema de Simulación de Adquisición de Objetivos y Derribo (Torreta 2-DoF & Radar Acconeer A121 con Física JAX / Crazyflow)

## Resumen del Proyecto y Objetivo

Desarrollar una aplicación de simulación modular y de alta fidelidad en Python aprovechando de manera nativa el entorno **`proyecto_dron` (JAX, MuJoCo y Crazyflow)**:
1. **Simular hasta 3 objetivos aéreos dinámicos (drones multirotor Crazyflie 2.X)** gobernados directamente por el **motor de física `crazyflow.sim.Sim` en JAX**, aprovechando sus ecuaciones de dinámica analítica y MuJoCo (`Physics.analytical` / `Physics.mujoco`), sus controladores internos de actitud y posición en cascada (`Control.state`), y su detección de colisiones de contacto (`sim.contacts()`).
2. **Generación de trayectorias tácticas de evasión**: Los drones siguen rutas 3D no lineales (curvas B-splines / Catmull-Rom) alimentadas como consignas al controlador de estado de Crazyflow, con evasión de colisiones contra el suelo y límites perimetrales.
3. **Simular una torreta de 2 grados de libertad (Pan/Tilt: Azimut y Elevación)** accionada por **modelos físicos de servomotores realistas** (slew rate finito, aceleración angular, banda muerta contra oscilaciones y traducción bidireccional a pulsos PWM de 1000–2000 µs).
4. **Simular con fidelidad física el sensor de radar Acconeer A121** (Pulsed Coherent Radar a 60 GHz, FoV $53^\circ \times 65^\circ$, resolución de rango de 3 cm, modelo de detección CFAR y estructura de datos tipada equivalente a `acc_detector_distance_result_t` del SDK oficial).
5. **Controlador de la torreta desacoplado e independiente** (`turret/turret_controller.py`): Máquina de estados finita (Búsqueda, Detección, Rastreo con filtro $\alpha$-$\beta$, Asociación de datos multiobjetivo por *nearest neighbor gating*, Cálculo de ángulo de deflexión balístico *lead angle* y Disparo).
6. **Simulación balística con gravedad, resistencia aerodinámica y transferencia de momento a JAX**: Al impactar un proyectil sobre la envolvente volumétrica de un dron, se inyecta la transferencia de momento lineal en los arrays inmutables de JAX (`sim.data.states.vel` / fuerzas externas), desestabilizando físicamente el dron en Crazyflow y provocando su caída en barrena hasta el suelo.
7. **Visualizador táctico 3D y telemetría de alto rendimiento**: Renderizado optimizado con exportación por streaming a video MP4 (vía `ffmpeg`, cero sobrecoste de RAM) o GIF, junto a una suite rigurosa de pruebas empíricas bajo la **Regla 2 de Control de Calidad**.

---

## Arquitectura de Integración con JAX / Crazyflow

```mermaid
graph TD
    subgraph CrazyflowEngine["Motor de Fisica Aerodinamica JAX (Crazyflow)"]
        CFSim["crazyflow.sim.Sim (n_worlds=1, n_drones=1..3)"]
        CFStates["JAX SimData (states.pos, states.vel, states.quat)"]
        CFControl["Controlador Interno de Vuelo (Control.state)"]
        CFSim -->|Integracion Fisica 100 Hz| CFStates
        CFControl -->|Consignas [x,y,z,vx,vy,vz]| CFSim
    end

    subgraph SimulationLoop["Bucle Central de Simulacion (sim/simulation_env.py)"]
        Waypoints["Generador de Trayectorias 3D y Guiado"]
        TurretPlatform["Torreta 2-DoF (Servos Pan/Tilt y Canon)"]
        RadarA121["Sensor Radar Acconeer A121 (60 GHz PCR)"]
        BallisticMgr["Sistema Balistico y Deteccion de Colisiones"]
    end

    subgraph DecoupledController["Algoritmo Desacoplado (turret/turret_controller.py)"]
        TurretAlgo["FSM: Search -> Acquire -> Track -> Engage -> Fire"]
    end

    subgraph VisualTelemetry["Visualizacion y Telemetria"]
        Renderer["Visualizador 3D HUD (Streaming MP4/GIF sin fuga de RAM)"]
    end

    Waypoints -->|Consignas de Navegacion 3D| CFControl
    CFStates -->|Posicion 3D Real de los Drones| RadarA121
    CFStates -->|Envolventes de Colision Esferica| BallisticMgr
    TurretPlatform -->|Orientacion Boresight de Antena| RadarA121
    TurretPlatform -->|Telemetria Servos y PWM| TurretAlgo
    RadarA121 -->|acc_detector_distance_result_t (Distancias y dB)| TurretAlgo
    TurretAlgo -->|Consignas Pan/Tilt y Trigger Disparo| TurretPlatform
    TurretPlatform -->|Disparo de Proyectil a Muzzle Velocity| BallisticMgr
    BallisticMgr -->|Inyeccion de Impulso y Desestabilizacion JAX| CFStates

    CFStates -.-> Renderer
    TurretPlatform -.-> Renderer
    RadarA121 -.-> Renderer
    BallisticMgr -.-> Renderer
```

---

## 1. Integración de Componentes Críticos

### 1.1. Objetivos Aéreos con Física Nativa en JAX / Crazyflow
En lugar de modelos puramente cinemáticos, aprovechamos el motor de cuatricópteros de `crazyflow`:
1. **Inicialización y Topología**:
   ```python
   sim = Sim(
       n_worlds=1,
       n_drones=num_drones,           # 1 a 3 drones en paralelo
       physics=Physics.analytical,     # o Physics.mujoco
       control=Control.state,          # Modo de control por estado
       freq=100,
       device="cpu",
   )
   ```
2. **Guiado y Navegación 3D**:
   - Cada dron sigue una curva 3D suave (Catmull-Rom Spline) calculada a su velocidad nominal diferenciada (ej. Dron 0: $1.8\text{ m/s}$, Dron 1: $3.2\text{ m/s}$, Dron 2: $4.6\text{ m/s}$).
   - En cada tick, se envía la consigna de posición y velocidad deseada mediante `sim.state_control(cmd)` con shape `(1, N, 13)` conteniendo `[x, y, z, vx, vy, vz, ...]`.
   - El controlador en cascada de Crazyflow calcula internamente el empuje de los rotores y los ángulos de actitud necesarios para seguir la ruta de forma físicamente plausible.
3. **Mecánica de Impacto Balístico y Daño en JAX**:
   - Al detectarse una colisión proyectil-dron:
     $$\Delta \vec{v} = \frac{m_{bullet} \cdot (\vec{v}_{bullet} - \vec{v}_{drone})}{m_{drone}}$$
   - Dado que los estados en JAX son inmutables, la perturbación se inyecta limpiamente:
     ```python
     new_vel = sim.data.states.vel.at[0, target_id].add(impulse / drone_mass)
     sim.data = sim.data.replace(states=sim.data.states.replace(vel=new_vel))
     ```
   - El dron entra en estado `HIT`: se altera su consigna enviándola al suelo y se le aplican momentos desestabilizadores, de modo que el simulador ejecuta una caída en espiral real hasta detenerse en el suelo (`DOWNED`).

### 1.2. Servomotores Pan/Tilt de la Torreta (`actuators/servo_motor.py`)
- Simulación física de actuadores reales de engranajes metálicos:
  * Pan: $[-180^\circ, +180^\circ]$, Tilt: $[-15^\circ, +85^\circ]$.
  * Slew rate limitado a $360^\circ/\text{s}$ y aceleración máxima de $2400^\circ/\text{s}^2$.
  * Banda muerta de $\pm 0.3^\circ$ para amortiguar oscilaciones.
  * Traducción bidireccional continua a microsegundos PWM ($1000 - 2000\ \mu s$).

### 1.3. Sensor Radar Acconeer A121 con Montaje Coaxial Solidario (`sensors/radar_a121.py`)
- **Montaje Rígido y Solidario al Cañón de la Torreta**:
  * El sensor A121 está rígidamente montado sobre el cañón orientable de la torreta (*piggyback / coaxial mounting*).
  * Su vector axial de radiación principal (*boresight vector*) se mueve **solidaria y simultáneamente 1:1 con los ejes de Pan (Azimut $\theta$) y Tilt (Elevación $\phi$)** accionados por los servomotores:
    $$\vec{f}_{radar}(t) = \vec{f}_{barrel}(t) = \begin{bmatrix} \sin\theta(t) \cos\phi(t) \\ \cos\theta(t) \cos\phi(t) \\ \sin\phi(t) \end{bmatrix}$$
  * Durante la fase de búsqueda (`SEARCH`), el haz de 60 GHz barre activamente el volumen del espacio aéreo al oscilar la torreta.
  * Al adquirir un dron (`TRACK`), la torreta encara el blanco situándolo directamente en el centro del haz de radiación, maximizando la ganancia de antena y la potencia de retorno (SNR).
- **Emulación fiel de especificaciones Waveshare / Acconeer C SDK**:
  * Haz con apertura elíptica de $53^\circ$ horizontal y $65^\circ$ vertical con patrón de ganancia gaussiano normalizado al boresight.
  * Ecuación radar monostática ($P_r \propto \text{RCS} \cdot G^2 / R^4$) con umbral de detección CFAR ($8.0\text{ dB}$).
  * Precisión de rango $\sigma = 0.015\text{ m}$ (resolución física de 3 cm).
  * Estructura de salida idéntica a `acc_detector_distance_result_t`.

### 1.4. Controlador Desacoplado de la Torreta (`turret/turret_controller.py`)
- Completamente aislado del entorno de simulación: sólo recibe `TurretInput` y produce `TurretAction`.
- Máquina de estados:
  1. `SEARCH`: Barrido espacial en Lissajous/nodding cubriendo el sector frontal.
  2. `ACQUIRE`: Detección de ecos A121 y encare inicial de antena.
  3. `TRACK`: Filtro $\alpha$-$\beta$ con compuerta de asociación de datos (*Nearest Neighbor gating*) para discriminar múltiples drones.
  4. `ENGAGE`: Cálculo balístico de intercepción (*lead angle*) con compensación de tiempo de vuelo y gravedad:
     $$\vec{P}_{aim} = \vec{P}_t + \vec{V}_t \cdot t_{tof} + \frac{1}{2} \vec{g} \cdot t_{tof}^2, \quad t_{tof} = \frac{\|\vec{P}_{aim} - \vec{P}_{turret}\|}{v_0}$$
  5. `FIRE`: Disparo del arma cuando el error angular $\|\Delta \theta, \Delta \phi\| < 2.0^\circ$ y el cañón no está en cooldown.
  6. `VERIFY`: Confirmación de caída/neutralización del objetivo y salto inmediato al siguiente contacto.

### 1.5. Visualizador 3D y Exportador Streaming (`visualization/visualizer_3d.py`)
- Representación táctica en 3D con cono de radiación de 60 GHz, proyectiles trazadores, estelas de drones y HUD de telemetría.
- Renderizado de alta eficiencia por tubería directa (*pipe*) a `ffmpeg`, permitiendo generar videos MP4 (`libx264`, `yuv420p`) sin acumulación de frames en memoria RAM, evitando cuelgues o fugas de memoria.

---

## 2. Arquitectura de Archivos del Proyecto

```
/home/barcenito/objetive_adquisition/
├── core/
│   ├── __init__.py
│   ├── coordinates.py          # Geometría esférica, cartesiana y base de orientación
│   └── math_utils.py           # Solvers y funciones auxiliares
├── actuators/
│   ├── __init__.py
│   └── servo_motor.py          # Modelo físico de servomotores (PWM, slew rate, deadband)
├── sensors/
│   ├── __init__.py
│   └── radar_a121.py           # Emulador físico del Acconeer A121 (60 GHz PCR)
├── targets/
│   ├── __init__.py
│   ├── crazyflow_swarm.py      # ENTORNO NATIVO JAX/CRAZYFLOW: Envoltorio multi-dron en JAX
│   ├── drone_target.py         # Interfaz de objetivo aéreo y máquina de estados (ALIVE, HIT, DOWNED)
│   └── trajectory_gen.py       # Rutas 3D seguras con prevención de colisión contra el suelo
├── ballistics/
│   ├── __init__.py
│   └── projectile_system.py    # Física balística (drag, gravedad, intercepción y colisión)
├── turret/
│   ├── __init__.py
│   ├── turret_model.py         # Plataforma mecánica 2-DoF y cálculo de boresight
│   └── turret_controller.py    # ARCHIVO INDEPENDIENTE: Algoritmo de control, FSM y lead
├── sim/
│   ├── __init__.py
│   ├── simulation_env.py       # Bucle de simulación coordinando Crazyflow + Torreta + Radar
│   └── scenarios.py            # Escenarios (1 dron lento, 2 drones mixtos, 3 drones enjambre)
├── visualization/
│   ├── __init__.py
│   └── visualizer_3d.py        # Visualizador 3D interactivo y exportador streaming MP4/GIF
├── tests/
│   ├── test_servo_motor.py     # Pruebas unitarias de respuesta de servos
│   ├── test_radar_a121.py      # Pruebas del haz de 60 GHz, CFAR y precisión
│   ├── test_ballistics.py      # Pruebas de intercepción balística y gravedad
│   ├── test_crazyflow_swarm.py # Pruebas del motor físico JAX/Crazyflow con drones
│   ├── test_drone_flight.py    # Pruebas de trayectorias sin impacto contra suelo
│   ├── test_turret_algo.py     # Pruebas del controlador desacoplado y retargeting
│   └── test_integration.py     # Pruebas integradas de ciclo completo de derribo
├── run_simulation.py           # Script principal de ejecución (headless, manual-test, MP4, GIF)
└── target_acquisition_turret_simulation_plan.md
```

---

## 3. Plan de Verificación y Control de Calidad Empírico (RULE 2)

En estricto cumplimiento de la regla **"CONTROL DE CALIDAD Y VERIFICACIÓN EXHAUSTIVA EMPÍRICA"**, cada componente se valida mediante tests automatizados y simulaciones ejecutadas en el entorno:

1. **Verificación de Dinámica JAX / Crazyflow (`tests/test_crazyflow_swarm.py`)**:
   - `test_crazyflow_init_multi_drone()`: Comprobar la instanciación de 1, 2 y 3 drones en paralelo en JAX.
   - `test_crazyflow_waypoint_tracking()`: Verificar que los drones convergen dinámicamente a las consignas de vuelo 3D.
   - `test_crazyflow_impact_perturbation()`: Comprobar que la inyección de impulso cinético en `sim.data.states.vel` altera la trayectoria del dron y activa la dinámica de caída.

2. **Verificación de Servomotores (`tests/test_servo_motor.py`)**:
   - Saturación de velocidad angular a $\omega_{max}$, banda muerta y correspondencia exacta de pulsos PWM (1000–2000 µs).

3. **Verificación de Radar Acconeer A121 (`tests/test_radar_a121.py`)**:
   - Haz elíptico $53^\circ \times 65^\circ$, discriminación de objetivos múltiples a distancias $> 3\text{ cm}$, relación señal/ruido CFAR y ruido gaussiano $\sigma = 0.015\text{ m}$.

4. **Verificación de Solución Balística (`tests/test_ballistics.py`)**:
   - Cálculo de compensación gravitatoria $0.5 g t^2$ e intercepción exitosa de objetivos en movimiento transversal.

5. **Verificación del Controlador Desacoplado (`tests/test_turret_algo.py`)**:
   - Transiciones de la FSM, modo manual de servomotores y asociación de datos multiobjetivo (*nearest neighbor gating*).

6. **Test de Integración y Derribo Completo (`tests/test_integration.py` y `run_simulation.py`)**:
   - Simulación de ciclo cerrado a 100 Hz con el enjambre de 3 drones gobernados por Crazyflow.
   - Medición empírica de tasa de derribo ($> 66\%$), tiempo de neutralización y verificación de ausencia de errores o fugas de memoria.

---

## 4. Fases de Ejecución

| Fase | Tarea Principal | Entregable / Verificación |
|---|---|---|
| **Fase 1** | Modelado Físico de Servos y Radar Acconeer A121 | `actuators/servo_motor.py`, `sensors/radar_a121.py`, tests unitarios pasando al 100%. |
| **Fase 2** | Integración del Motor Físico JAX / Crazyflow | `targets/crazyflow_swarm.py`, acoplamiento de dinámicas de vuelo y perturbación por impacto en JAX. |
| **Fase 3** | Balística, Cinemática de Torreta y Detección de Colisiones | `ballistics/projectile_system.py`, `turret/turret_model.py`, verificación de intercepción. |
| **Fase 4** | Algoritmo Desacoplado de la Torreta | `turret/turret_controller.py`, FSM de búsqueda, seguimiento, *gating* y tiro balístico en archivo aislado. |
| **Fase 5** | Entorno Integrado y Visualizador Streaming MP4 | `sim/simulation_env.py`, `visualization/visualizer_3d.py`, exportación por pipe a `ffmpeg`. |
| **Fase 6** | Validación y Verificación Empírica Exhaustiva | Suite completa de 25+ tests en `tests/`, benchmarks headless y reporte final. |
