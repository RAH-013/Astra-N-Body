import argparse
import csv
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path


def run_case(binary, backend, particles, steps, precision):
    command = [
        str(binary),
        "--backend", backend,
        "--particles", str(particles),
        "--steps", str(steps),
        "--seed", "42",
        "--dt", "0.001",
        "--softening", "0.01",
    ]
    if backend == "opencl":
        command.extend(["--no-fallback", "--precision", precision])
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 4, 12)))
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return None, output

    elapsed_match = re.search(r"Elapsed time\s*:\s*([0-9.eE+-]+)\s*s", output)
    transfer_match = re.search(r"Host transfer time:\s*([0-9.eE+-]+)\s*s", output)
    drift_match = re.search(r"Energy drift\s*:\s*([0-9.eE+-]+)\s*%", output)
    backend_match = re.search(r"Active Backend\s*:\s*(.*)", output)
    device_match = re.search(r"OpenCL Device:\s*\[[^]]+\]\s*(.*?)\s*\(", output)
    if elapsed_match is None:
        return None, output

    elapsed = float(elapsed_match.group(1))
    transfer = float(transfer_match.group(1)) if transfer_match else 0.0
    drift = float(drift_match.group(1)) if drift_match else float("nan")
    return {
        "kernel_seconds": elapsed,
        "transfer_seconds": transfer,
        "total_seconds": elapsed + transfer,
        "energy_drift_pct": drift,
        "backend_description": backend_match.group(1).strip() if backend_match else backend,
        "device_name": device_match.group(1).strip() if device_match else "",
    }, output


def median_or_none(values):
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp64"], default="auto")
    args = parser.parse_args()

    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"Executable not found: {binary}")
    if args.steps < 1 or args.repetitions < 1 or any(size < 2 for size in args.sizes):
        raise SystemExit("Steps and repetitions must be positive; particle counts must be at least 2.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for count in args.sizes:
        backend_results = {}
        for backend in ("cpu", "opencl"):
            samples = []
            if backend == "opencl":
                warmup, warmup_output = run_case(binary, backend, count, min(args.steps, 2), args.precision)
                if warmup is None:
                    print(f"[GPU unavailable] N={count}: {warmup_output.strip().splitlines()[-1] if warmup_output.strip() else 'OpenCL unavailable'}")
                    backend_results[backend] = None
                    continue

            for repetition in range(args.repetitions):
                sample, output = run_case(binary, backend, count, args.steps, args.precision)
                if sample is None:
                    print(f"[Failed] {backend.upper()} N={count}: {output}")
                    samples = []
                    break
                samples.append(sample)
                print(f"{backend.upper():6} N={count:6} run={repetition + 1}/{args.repetitions} total={sample['total_seconds']:.6f}s")

            backend_results[backend] = samples if samples else None

        cpu_samples = backend_results.get("cpu") or []
        gpu_samples = backend_results.get("opencl") or []
        cpu_kernel = median_or_none([item["kernel_seconds"] for item in cpu_samples])
        gpu_kernel = median_or_none([item["kernel_seconds"] for item in gpu_samples])
        gpu_transfer = median_or_none([item["transfer_seconds"] for item in gpu_samples])
        gpu_total = median_or_none([item["total_seconds"] for item in gpu_samples])
        cpu_total = cpu_kernel

        rows.append({
            "particles": count,
            "steps": args.steps,
            "repetitions": args.repetitions,
            "cpu_seconds_median": cpu_kernel,
            "gpu_kernel_seconds_median": gpu_kernel,
            "gpu_transfer_seconds_median": gpu_transfer,
            "gpu_total_seconds_median": gpu_total,
            "kernel_speedup": cpu_kernel / gpu_kernel if cpu_kernel and gpu_kernel else None,
            "end_to_end_speedup": cpu_total / gpu_total if cpu_total and gpu_total else None,
            "cpu_energy_drift_pct_median": median_or_none([item["energy_drift_pct"] for item in cpu_samples]),
            "gpu_energy_drift_pct_median": median_or_none([item["energy_drift_pct"] for item in gpu_samples]),
            "gpu_status": "available" if gpu_samples else "unavailable",
            "gpu_device": gpu_samples[0]["device_name"] if gpu_samples else "",
            "gpu_backend": gpu_samples[0]["backend_description"] if gpu_samples else "",
        })

    metadata_path = output_dir / "benchmark_environment.txt"
    metadata_path.write_text(
        "ASTRA N-BODY benchmark environment\n"
        f"Platform: {platform.platform()}\n"
        f"Architecture: {platform.machine()}\n"
        f"Python: {platform.python_version()}\n"
        f"Logical CPUs: {os.cpu_count()}\n"
        f"OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'runtime default')}\n"
        f"Executable: {binary}\n"
        f"Particle sizes: {args.sizes}\n"
        f"Steps: {args.steps}\n"
        f"Repetitions: {args.repetitions}\n"
        f"Precision: {args.precision}\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        valid_cpu = [(row["particles"], row["cpu_seconds_median"]) for row in rows if row["cpu_seconds_median"] is not None]
        valid_gpu = [(row["particles"], row["gpu_total_seconds_median"]) for row in rows if row["gpu_total_seconds_median"] is not None]
        if valid_cpu:
            plt.figure()
            plt.plot([item[0] for item in valid_cpu], [item[1] for item in valid_cpu], marker="o", label="CPU")
            if valid_gpu:
                plt.plot([item[0] for item in valid_gpu], [item[1] for item in valid_gpu], marker="o", label="OpenCL GPU total")
            plt.xlabel("Particle count")
            plt.ylabel("Median execution time (s)")
            plt.title("ASTRA N-BODY performance")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "benchmark_performance.png", dpi=180)
            plt.close()

        valid_speedup = [(row["particles"], row["end_to_end_speedup"]) for row in rows if row["end_to_end_speedup"] is not None]
        if valid_speedup:
            plt.figure()
            plt.plot([item[0] for item in valid_speedup], [item[1] for item in valid_speedup], marker="o")
            plt.axhline(1.0, linestyle="--")
            plt.xlabel("Particle count")
            plt.ylabel("End-to-end speedup (CPU / GPU)")
            plt.title("ASTRA N-BODY GPU speedup")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "benchmark_speedup.png", dpi=180)
            plt.close()
    except ImportError:
        print("matplotlib not installed; CSV was generated without plots.")

    print(f"\nResults: {csv_path.resolve()}")
    if not any(row["gpu_status"] == "available" for row in rows):
        print("No OpenCL device was available. CPU measurements were saved; GPU columns remain empty.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
