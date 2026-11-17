#include "astra/validator.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace astra {

ValidationReport Validator::compare_cpu_and_opencl(
    const SimulationConfig& base_config,
    std::size_t steps,
    bool run_warmup
) {
    ValidationReport report{};

    // 1. Reference CPU execution (FP64)
    SimulationConfig cpu_cfg = base_config;
    cpu_cfg.requested_backend = BackendType::CPU;

    Simulation cpu_sim(cpu_cfg);
    cpu_sim.initialize_galaxy();
    const double cpu_initial_energy = cpu_sim.stats().total_energy;
    cpu_sim.run(steps);

    const auto& cpu_stats = cpu_sim.stats();
    report.cpu_elapsed_seconds = cpu_stats.elapsed_seconds;
    report.cpu_energy_drift_pct = (cpu_initial_energy != 0.0)
        ? std::abs((cpu_stats.total_energy - cpu_initial_energy) / cpu_initial_energy) * 100.0
        : 0.0;

    // 20 FLOPs per pairwise gravitational interaction per step
    const double flops_total =
        20.0 * static_cast<double>(base_config.particle_count) *
        static_cast<double>(base_config.particle_count) *
        static_cast<double>(steps);

    report.cpu_gflops = (report.cpu_elapsed_seconds > 0.0)
        ? (flops_total / report.cpu_elapsed_seconds) * 1e-9
        : 0.0;

    // 2. OpenCL GPU execution
    SimulationConfig gpu_cfg = base_config;
    gpu_cfg.requested_backend = BackendType::OpenCL;
    gpu_cfg.allow_cpu_fallback = false;

    try {
        // Warmup run avoids JIT kernel compilation skewing benchmark timing
        if (run_warmup && steps > 0) {
            Simulation warmup_sim(gpu_cfg);
            warmup_sim.initialize_galaxy();
            warmup_sim.run(1);
        }

        Simulation gpu_sim(gpu_cfg);
        if (gpu_sim.active_backend() != BackendType::OpenCL) {
            report.gpu_available = false;
            return report;
        }

        report.gpu_available = true;
        report.gpu_using_fp64 = gpu_sim.is_using_fp64();
        if (gpu_sim.active_opencl_device().has_value()) {
            report.gpu_device_name = gpu_sim.active_opencl_device()->device_name;
        }

        gpu_sim.initialize_galaxy();
        const double gpu_initial_energy = gpu_sim.stats().total_energy;
        gpu_sim.run(steps);

        const auto& gpu_stats = gpu_sim.stats();
        report.gpu_elapsed_seconds = gpu_stats.elapsed_seconds;
        report.gpu_transfer_seconds = gpu_stats.host_transfer_seconds;
        report.gpu_energy_drift_pct = (gpu_initial_energy != 0.0)
            ? std::abs((gpu_stats.total_energy - gpu_initial_energy) / gpu_initial_energy) * 100.0
            : 0.0;

        report.speedup = (report.gpu_elapsed_seconds > 0.0)
            ? (report.cpu_elapsed_seconds / report.gpu_elapsed_seconds)
            : 0.0;
        report.gpu_gflops = (report.gpu_elapsed_seconds > 0.0)
            ? (flops_total / report.gpu_elapsed_seconds) * 1e-9
            : 0.0;

        // 3. Particle-by-particle numerical comparison
        const auto& p_cpu = cpu_sim.particles();
        const auto& p_gpu = gpu_sim.particles();

        double sum_sq_pos_err = 0.0;
        for (std::size_t i = 0; i < p_cpu.size(); ++i) {
            const Vec3 dp = p_gpu[i].position.xyz() - p_cpu[i].position.xyz();
            const Vec3 dv = p_gpu[i].velocity.xyz() - p_cpu[i].velocity.xyz();

            const double pos_err = dp.norm();
            const double vel_err = dv.norm();
            const double ref_norm = std::max(1.0, p_cpu[i].position.xyz().norm());

            report.max_abs_position_error = std::max(report.max_abs_position_error, pos_err);
            report.max_rel_position_error = std::max(report.max_rel_position_error, pos_err / ref_norm);
            report.max_abs_velocity_error = std::max(report.max_abs_velocity_error, vel_err);
            sum_sq_pos_err += pos_err * pos_err;
        }

        report.rms_position_error = std::sqrt(sum_sq_pos_err / static_cast<double>(p_cpu.size()));
        report.relative_energy_diff_pct = (cpu_stats.total_energy != 0.0)
            ? std::abs((gpu_stats.total_energy - cpu_stats.total_energy) / cpu_stats.total_energy) * 100.0
            : 0.0;

        if (report.gpu_using_fp64) {
            report.passed =
                (report.max_rel_position_error <= 1e-7) &&
                (report.relative_energy_diff_pct <= 1e-6);
        } else {
            // In FP32, individual close-encounter pairs exhibit chaotic Lyapunov phase sensitivity,
            // so physical equivalence is governed by ensemble RMS error and total energy conservation.
            report.passed =
                (report.rms_position_error <= 2.5e-2) &&
                (report.relative_energy_diff_pct <= 0.1) &&
                (report.gpu_energy_drift_pct <= 1.0);
        }

    } catch (...) {
        report.gpu_available = false;
        report.passed = false;
    }

    return report;
}

void Validator::print_report(
    const ValidationReport& report,
    std::size_t particle_count,
    std::size_t steps,
    std::ostream& out
) {
    out << "\nCPU vs OpenCL Validation & Benchmark\n";
    out << "============================================================\n";
    out << "Particles        : " << particle_count << "\n";
    out << "Steps            : " << steps << "\n";

    if (!report.gpu_available) {
        out << "OpenCL GPU Status: Unavailable in current environment\n";
        out << std::fixed << std::setprecision(6);
        out << "CPU Reference    : " << report.cpu_elapsed_seconds << " s ("
            << report.cpu_gflops << " GFLOP/s, drift " << report.cpu_energy_drift_pct << " %)\n";
        return;
    }

    out << "OpenCL Device    : " << report.gpu_device_name
        << " (" << (report.gpu_using_fp64 ? "FP64" : "FP32") << ")\n";

    out << std::fixed << std::setprecision(6);
    out << "\nPerformance\n";
    out << "------------------------------------------------------------\n";
    out << "CPU Time         : " << report.cpu_elapsed_seconds << " s ("
        << report.cpu_gflops << " GFLOP/s)\n";
    out << "GPU Kernel Time  : " << report.gpu_elapsed_seconds << " s ("
        << report.gpu_gflops << " GFLOP/s)\n";
    out << "GPU H2D/D2H Time : " << report.gpu_transfer_seconds << " s\n";
    out << "Speedup (CPU/GPU): " << report.speedup << "x\n";

    out << std::scientific << std::setprecision(6);
    out << "\nNumerical Accuracy\n";
    out << "------------------------------------------------------------\n";
    out << "Max Abs Pos Err  : " << report.max_abs_position_error << "\n";
    out << "Max Rel Pos Err  : " << report.max_rel_position_error << "\n";
    out << "RMS Pos Err      : " << report.rms_position_error << "\n";
    out << "Max Abs Vel Err  : " << report.max_abs_velocity_error << "\n";

    out << std::fixed << std::setprecision(6);
    out << "CPU Energy Drift : " << report.cpu_energy_drift_pct << " %\n";
    out << "GPU Energy Drift : " << report.gpu_energy_drift_pct << " %\n";
    out << "Energy Diff (%)  : " << report.relative_energy_diff_pct << " %\n";
    out << "Validation       : " << (report.passed ? "PASS" : "FAIL") << "\n";
}

} // namespace astra