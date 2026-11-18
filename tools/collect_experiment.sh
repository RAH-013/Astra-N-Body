#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${ASTRA_BINARY:-$ROOT/build/astra-nbody}"
OUTPUT="${ASTRA_RESULTS_DIR:-$ROOT/results/presentation}"
PARTICLES="${ASTRA_PARTICLES:-512}"
STEPS="${ASTRA_STEPS:-100}"
THREADS="${OMP_NUM_THREADS:-12}"

if [[ ! -x "$BINARY" ]]; then
    printf 'Executable not found: %s\n' "$BINARY" >&2
    exit 1
fi

mkdir -p "$OUTPUT"

ctest --test-dir "$ROOT/build" --output-on-failure | tee "$OUTPUT/tests.txt"
"$BINARY" --list-devices > "$OUTPUT/opencl-devices.txt" 2>&1 || true
OMP_NUM_THREADS="$THREADS" "$BINARY" --backend cpu --particles "$PARTICLES" --steps "$STEPS" --trajectory "$OUTPUT/trajectory.csv" --snapshot-every 5 > "$OUTPUT/simulation-cpu.txt"
python3 "$ROOT/tools/visualize.py" "$OUTPUT/trajectory.csv" --output "$OUTPUT/visualization" | tee "$OUTPUT/visualization.txt"
python3 "$ROOT/tools/benchmark.py" --binary "$BINARY" --sizes 512 1024 2048 4096 --steps 100 --repetitions 5 --output "$OUTPUT/benchmark" || benchmark_status=$?
benchmark_status="${benchmark_status:-0}"
validation_status=0
for precision in fp32 fp64; do
    OMP_NUM_THREADS="$THREADS" "$BINARY" --validate --precision "$precision" --particles 1024 --steps 100 > "$OUTPUT/opencl-validation-$precision.txt" 2>&1 || validation_status=$?
done

printf 'Results directory: %s\n' "$OUTPUT"
printf 'Benchmark status: %s\n' "$benchmark_status"
printf 'OpenCL validation status: %s\n' "$validation_status"
