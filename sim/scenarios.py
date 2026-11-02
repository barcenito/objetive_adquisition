"""
Simulation Scenario Presets.
"""

from typing import Dict, Any


SCENARIOS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Single Target (Lento / Reconocimiento)",
        "description": "1 dron volando a velocidad moderada (2.0 m/s) para validación básica de adquisición.",
        "num_drones": 1,
        "drone_speeds": [2.0],
    },
    2: {
        "name": "Dual Target (Crucero y Alta Velocidad)",
        "description": "2 drones simultáneos con velocidades diferenciadas (2.2 m/s y 3.8 m/s).",
        "num_drones": 2,
        "drone_speeds": [2.2, 3.8],
    },
    3: {
        "name": "Triple Threat (Enjambre Completo - 3 Drones)",
        "description": "3 drones a diferentes velocidades (1.5 m/s, 3.0 m/s y 4.8 m/s) con maniobras aleatorias.",
        "num_drones": 3,
        "drone_speeds": [1.5, 3.0, 4.8],
    },
}
