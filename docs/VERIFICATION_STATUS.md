# ASTRA N-BODY 0.2.2 — Estado de verificación

## Verificado en el entorno de construcción

- Configuración CMake Release con C++20.
- Compilación con OpenMP habilitado.
- Compilación CPU fallback cuando no están disponibles las cabeceras y la biblioteca de desarrollo OpenCL.
- Suite CTest: aprobada.
- Órbita circular de dos cuerpos: deriva energética dentro del umbral de prueba.
- Sistema de 4096 partículas: deriva energética y momento dentro de los umbrales de prueba.
- Reproducibilidad mediante semilla.
- Exportación CSV de trayectorias.
- Generación de PNG y GIF a partir de una trayectoria.
- Ejecución del benchmark CPU y exportación de CSV.
- Sintaxis de los scripts Python y Bash.

## Pendiente de verificar en el equipo con GPU

El entorno utilizado para generar esta entrega no dispone de cabeceras de desarrollo OpenCL, por lo que CMake construyó correctamente el fallback CPU. No fue posible compilar ni ejecutar el backend OpenCL en este entorno.

Antes de declarar terminada la validación GPU, ejecutar en el equipo objetivo:

1. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASTRA_ENABLE_OPENMP=ON -DASTRA_ENABLE_OPENCL=ON`
2. `cmake --build build --parallel`
3. `ctest --test-dir build --output-on-failure`
4. `./build/astra-nbody --list-devices`
5. `./build/astra-nbody --validate --backend opencl --no-fallback --precision fp32 --particles 1024 --steps 100`
6. Si el dispositivo anuncia FP64: `./build/astra-nbody --validate --backend opencl --no-fallback --precision fp64 --particles 1024 --steps 100`
7. `bash tools/collect_experiment.sh`

La ausencia de un dispositivo OpenCL no debe registrarse como una prueba GPU aprobada. Los resultados CPU y GPU deben mantenerse separados.
