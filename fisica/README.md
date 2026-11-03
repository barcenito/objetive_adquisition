# Pruebas físicas

Esta sección reúne los ensayos de banco del proyecto. Cada prueba tiene su
propio directorio en `arduino/`. Las notas, los resultados y las grabaciones
pueden añadirse a `docs/` y `videos/` con el mismo nombre que el ensayo.
Los esquemas y las bibliotecas de diseño se encuentran en [hardware](../hardware/README.md).

| Carpeta | Contenido |
| --- | --- |
| [`arduino/`](arduino/) | Sketches de Arduino, uno por ensayo. |
| [`docs/`](docs/) | Notas del montaje, procedimiento y resultados. |
| [`videos/`](videos/) | Grabaciones de las pruebas. |

## Ensayo de Kalman con ultrasonidos y servo

El ensayo usa un HC-SR04 y un servomotor para observar la estimación de
distancia y ángulo frente a un objeto de prueba. El sketch aplica dos filtros
de Kalman independientes, uno a la distancia que mide el sensor y otro al
ángulo estimado a partir del barrido del servo. La telemetría serie a 9600 baudios
muestra el estado del ensayo y permite comparar el ángulo estimado con el
ángulo comandado al servo.

El archivo original `main.cpp` se conserva como sketch `.ino`, en una carpeta
del mismo nombre para abrirlo desde Arduino IDE. Usa el núcleo Arduino y la
biblioteca `Servo`. [Abrir el sketch](arduino/kalman_ultrasonidos_servo/kalman_ultrasonidos_servo.ino)

El ensayo funciona como prueba de banco independiente del simulador.
[Ver la grabación del ensayo sin audio](videos/test_ultrasonido.mp4).
La documentación del montaje queda pendiente de incorporar.
