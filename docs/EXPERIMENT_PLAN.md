# ASTRA N-BODY — Plan de pruebas y evidencia

Este documento define cómo recopilar resultados reproducibles para evaluar el simulador y preparar la presentación. No contiene resultados medidos del equipo objetivo.

## 1. Datos del entorno

Registrar antes de medir:

- Modelo de CPU y número de núcleos/hilos.
- Modelo de GPU y memoria disponible.
- Sistema operativo y versión del controlador gráfico.
- Runtime y versión OpenCL.
- Compilador, versión de CMake y configuración Release.
- Versión de ASTRA N-BODY.
- Número de hilos OpenMP.
- Precisión OpenCL seleccionada.

Guardar la salida de `astra-nbody --list-devices` junto con los resultados.

## 2. Parámetros controlados

| Parámetro | Valor de referencia |
|---|---:|
| Semilla | 42 |
| Paso temporal | 0.001 |
| Softening | 0.01 |
| Iteraciones | 100 |
| Partículas | 512, 1024, 2048, 4096 |
| Repeticiones | 5 |

Mantener idénticos semilla, número de partículas, paso temporal, softening y número de iteraciones al comparar CPU y GPU.

## 3. Procedimiento

1. Compilar en configuración Release.
2. Ejecutar CTest y conservar la salida.
3. Enumerar los dispositivos OpenCL.
4. Ejecutar una inicialización de calentamiento GPU antes de medir.
5. Ejecutar cinco repeticiones por backend y tamaño.
6. Utilizar la mediana como valor representativo.
7. Repetir la validación numérica para FP64 y FP32 cuando el dispositivo lo permita.
8. Exportar una trayectoria CPU para la visualización 3D.
9. Conservar CSV, capturas, PNG, GIF y registros de consola.

## 4. Métricas

- Tiempo CPU: duración del bucle de simulación CPU.
- Tiempo GPU kernel: duración de los kernels OpenCL.
- Transferencia: subida inicial y descarga final de partículas.
- Tiempo GPU total: tiempo GPU kernel más transferencias medidas.
- Speedup de kernel: tiempo CPU / tiempo GPU kernel.
- Speedup end-to-end: tiempo CPU / tiempo GPU total.
- Deriva energética: valor absoluto del cambio relativo de energía total entre inicio y final, expresado en porcentaje.
- Error de posición: máximo absoluto, máximo relativo y RMS entre CPU y GPU.
- Error de velocidad: máximo absoluto y relativo entre CPU y GPU.
- Momento lineal: magnitud del momento total del sistema.

No presentar el speedup de kernel como speedup total. Mostrar ambos cuando se discuta rendimiento.

## 5. Validación numérica

La validación debe cubrir:

- Órbita circular de dos cuerpos.
- Conservación de energía y momento en el sistema de 4096 partículas.
- Reproducibilidad con semilla fija.
- Resultados CPU/OpenCL para el mismo estado inicial.
- Valores finitos en posiciones, velocidades y masas.
- FP32 y FP64 como resultados separados, nunca mezclados en una misma categoría.

Los umbrales están implementados en `src/validator.cpp` y deben revisarse junto con los resultados reales del dispositivo antes de declarar la validación aprobada.

## 6. Archivos que deben conservarse

- `tests.txt`
- `opencl-devices.txt`
- `simulation-cpu.txt`
- `opencl-validation.txt`
- `benchmark/benchmark.csv`
- `benchmark/benchmark_environment.txt`
- `benchmark/benchmark_performance.png`
- `benchmark/benchmark_speedup.png`, si hay GPU
- `trajectory.csv`
- `visualization/astra_snapshot.png`
- `visualization/astra_evolution.gif`

## 7. Orden sugerido de diapositivas

1. Problema y objetivo del proyecto.
2. Modelo físico y ecuación gravitacional.
3. Integración Velocity Verlet.
4. Arquitectura CPU/OpenCL.
5. Paralelización de fuerzas O(N²).
6. Entorno de hardware y software.
7. Validación numérica y conservación física.
8. Visualización de la evolución de partículas.
9. Comparación de rendimiento CPU/GPU.
10. Speedup, transferencias y análisis de resultados.
11. Limitaciones y trabajo futuro.
12. Conclusiones.

Los resultados GPU deben completarse únicamente después de ejecutar las pruebas en el hardware objetivo.
