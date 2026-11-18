# ASTRA N-BODY 0.2.2

Simulador gravitacional tridimensional de N cuerpos desarrollado en C++20. Implementa un backend de referencia en CPU con OpenMP y aceleración OpenCL opcional para GPU.

## Objetivo

Modelar la interacción gravitacional entre partículas, validar la integración numérica y comparar el rendimiento de la ejecución paralela en CPU y GPU.

## Características

- Simulación gravitacional newtoniana en 3D.
- Interacción directa entre todos los pares de partículas: complejidad O(N²).
- Suavizado gravitacional (*gravitational softening*).
- Integrador Velocity Verlet.
- Backend CPU en precisión doble con paralelismo OpenMP.
- Backend OpenCL con selección de dispositivo GPU y modos FP32/FP64.
- Fallback a CPU cuando se utiliza el modo automático.
- Validación numérica CPU/OpenCL, conservación de energía y momento.
- Exportación CSV del estado final o de snapshots durante la simulación.
- Herramientas Python para benchmarks repetidos y visualización 3D.
- Contenedores para ejecución CPU y configuración GPU opcional en Linux.

## Modelo matemático

Para la partícula `i`, la aceleración gravitacional es:

\[
\mathbf{a}_i = G \sum_{j \ne i} m_j
\frac{\mathbf{r}_j-\mathbf{r}_i}
{\left(\lVert\mathbf{r}_j-\mathbf{r}_i\rVert^2+\epsilon^2\right)^{3/2}}
\]

El parámetro `ε` suaviza las interacciones a distancias pequeñas. La integración temporal utiliza Velocity Verlet:

\[
\mathbf{v}_{i,n+1/2}=\mathbf{v}_{i,n}+\frac{\Delta t}{2}\mathbf{a}_{i,n}
\]

\[
\mathbf{r}_{i,n+1}=\mathbf{r}_{i,n}+\Delta t\mathbf{v}_{i,n+1/2}
\]

\[
\mathbf{v}_{i,n+1}=\mathbf{v}_{i,n+1/2}+\frac{\Delta t}{2}\mathbf{a}_{i,n+1}
\]

## Requisitos

### Compilación nativa

- CMake 3.20 o posterior.
- Compilador C++20.
- OpenMP opcional.
- OpenCL 1.2 SDK opcional para compilar el backend OpenCL.
- Runtime OpenCL y controlador compatible para ejecutar en GPU.

### Herramientas de análisis

- Python 3.10 o posterior.
- Dependencias de `requirements-visualization.txt` para gráficas y GIF.

## Compilación en Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASTRA_ENABLE_OPENMP=ON -DASTRA_ENABLE_OPENCL=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Si no están instaladas las cabeceras o bibliotecas OpenCL, CMake construye el backend CPU y muestra una advertencia.

## Compilación en Windows

Con Visual Studio 2022 y CMake:

```powershell
cmake -S . -B build -A x64 -DASTRA_ENABLE_OPENMP=ON -DASTRA_ENABLE_OPENCL=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Para OpenCL se necesita instalar un SDK compatible con el controlador de gráficos. La disponibilidad del dispositivo GPU depende del runtime instalado en el sistema.

## Ejecución

```bash
./build/astra-nbody --backend cpu --particles 1024 --steps 100
```

Selección automática con fallback:

```bash
./build/astra-nbody --backend auto --particles 1024 --steps 100
```

Ejecución OpenCL estricta, sin fallback:

```bash
./build/astra-nbody --backend opencl --no-fallback --precision auto --particles 1024 --steps 100
```

Enumerar dispositivos:

```bash
./build/astra-nbody --list-devices
```

Parámetros principales:

| Parámetro | Descripción | Predeterminado |
|---|---|---:|
| `--particles` | Número total de partículas | 1024 |
| `--steps` | Iteraciones | 100 |
| `--dt` | Paso temporal | 0.001 |
| `--softening` | Suavizado gravitacional | 0.01 |
| `--seed` | Semilla reproducible | 42 |
| `--backend` | `auto`, `cpu` u `opencl` | `auto` |
| `--precision` | `auto`, `fp32` o `fp64` | `auto` |
| `--no-central-body` | Desactiva la masa central | Desactivado |

`--particles` representa siempre el número total de partículas, incluida la masa central cuando está habilitada.

## Validación CPU/OpenCL

```bash
./build/astra-nbody --validate --particles 1024 --steps 100
```

El comando devuelve un código distinto de cero cuando no hay una GPU OpenCL disponible o cuando falla la validación. El informe incluye error absoluto y relativo de posición/velocidad, deriva energética, momento, tiempo de kernel, transferencias y aceleración. La validación considera explícitamente los errores relativos de velocidad. Ejecutar también `--precision fp32` y `--precision fp64` para separar la validación de ambas precisiones; FP64 requiere soporte del dispositivo.

La aceleración *kernel* compara únicamente los tiempos de cómputo. La aceleración *end-to-end* incluye las transferencias de datos reportadas por la aplicación.

## Exportación y visualización

Exportar el estado final:

```bash
./build/astra-nbody --backend cpu --particles 1024 --steps 100 --export-csv results/final.csv
```

Guardar snapshots durante la simulación:

```bash
./build/astra-nbody --backend cpu --particles 1024 --steps 500 --trajectory results/trajectory.csv --snapshot-every 10
```

Instalar dependencias y generar PNG/GIF:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-visualization.txt
python3 tools/visualize.py results/trajectory.csv --output results/visualization
```

El CSV de trayectoria contiene un registro por partícula y snapshot. El PNG muestra la distribución final y el GIF muestra su evolución temporal.

## Benchmark reproducible

Ejecutar cinco repeticiones por tamaño:

```bash
python3 tools/benchmark.py --binary ./build/astra-nbody --sizes 512 1024 2048 4096 --steps 100 --repetitions 5 --output results/benchmark
```

El script realiza calentamiento OpenCL, utiliza la misma semilla y parámetros para ambos backends y calcula medianas. Produce `benchmark.csv`, una gráfica de rendimiento y, si hay GPU, una gráfica de speedup. Si no hay GPU, conserva las mediciones CPU y deja explícitamente vacíos los resultados GPU.

Para recopilar automáticamente pruebas, información OpenCL, simulación, visualización, benchmarks y validación: `bash tools/collect_experiment.sh`. Se puede ajustar con `ASTRA_PARTICLES`, `ASTRA_STEPS`, `OMP_NUM_THREADS`, `ASTRA_BINARY` y `ASTRA_RESULTS_DIR`.

## Docker / Podman

Ejecución portable CPU/auto:

```bash
docker compose build
docker compose run --rm astra-nbody
```

Ejecución OpenCL GPU en Linux con `/dev/dri` disponible:

```bash
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
export VIDEO_GID RENDER_GID

docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm astra-nbody
```

La configuración GPU requiere que el host tenga `/dev/dri`, los grupos `video` y `render` (si existen) y un controlador/runtime OpenCL funcional. Si alguno de esos grupos no existe, elimina su entrada `group_add` correspondiente de `compose.gpu.yaml`. La imagen instala Mesa OpenCL/Rusticl; `RUSTICL_ENABLE=radeonsi` activa el controlador AMD. El acceso a GPU desde Docker Desktop en Windows depende de la integración del proveedor y no se garantiza mediante el montaje `/dev/dri`. Para Windows se garantiza el backend CPU; la GPU OpenCL se considera experimental hasta validar el passthrough del runtime. El modo CPU permanece disponible en Windows y Linux.

## Pruebas

```bash
ctest --test-dir build --output-on-failure
```

Las pruebas CPU cubren el layout de memoria, una órbita binaria, el conteo exacto de partículas, estabilidad energética y conservación del momento. La prueba OpenCL se ejecuta cuando se detecta un dispositivo GPU; en caso contrario se marca como omitida, no como validación GPU aprobada.

## Limitaciones

- El cálculo directo de fuerzas tiene complejidad O(N²).
- El resultado depende de `dt`, `softening`, condiciones iniciales y precisión numérica.
- FP32 y FP64 pueden producir trayectorias distintas en simulaciones sensibles a condiciones iniciales.
- OpenCL GPU requiere SDK/runtime y controlador compatibles en el host.
- La simulación utiliza unidades normalizadas; no representa automáticamente unidades astronómicas SI.

## Estructura

```text
Astra-N-Body/
├── CMakeLists.txt
├── Dockerfile
├── compose.yaml
├── compose.gpu.yaml
├── include/astra/
├── kernels/nbody.cl
├── src/
├── tests/
├── tools/benchmark.py
├── tools/visualize.py
└── requirements-visualization.txt
```
