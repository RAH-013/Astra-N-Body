
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

    if (steps == 0) {
        report.error_message = "Validation requires at least one simulation step.";
        return report;
    }

    SimulationConfig cpu_cfg = base_config;
    cpu_cfg.requested_backend = BackendType::CPU;

    Simulation cpu_sim(cpu_cfg);
    cpu_sim.initialize_galaxy();

    const double cpu_initial_energy = cpu_sim.stats().total_energy;
    cpu_sim.run(steps);

    const auto& cpu_stats = cpu_sim.stats();

    report.cpu_elapsed_seconds = cpu_stats.elapsed_seconds;
    report.cpu_energy_drift_pct = cpu_initial_energy != 0.0
        ? std::abs((cpu_stats.total_energy - cpu_initial_energy) / cpu_initial_energy) * 100.0
        : 0.0;
    report.cpu_momentum = cpu_stats.momentum;

    const double n = static_cast<double>(base_config.particle_count);
    const double flops_total = 20.0 * n * n * static_cast<double>(steps);

    report.cpu_gflops = report.cpu_elapsed_seconds > 0.0
        ? flops_total / report.cpu_elapsed_seconds * 1e-9
        : 0.0;

    SimulationConfig gpu_cfg = base_config;
    gpu_cfg.requested_backend = BackendType::OpenCL;
    gpu_cfg.allow_cpu_fallback = false;

    try {
        if (run_warmup) {
            Simulation warmup_sim(gpu_cfg);
            warmup_sim.initialize_galaxy();
            warmup_sim.run(std::min<std::size_t>(steps, 2));
        }

        Simulation gpu_sim(gpu_cfg);

        if (gpu_sim.active_backend() != BackendType::OpenCL) {
            report.error_message = "OpenCL initialization did not activate an OpenCL backend.";
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
        report.gpu_total_seconds = report.gpu_elapsed_seconds + report.gpu_transfer_seconds;

        report.gpu_energy_drift_pct = gpu_initial_energy != 0.0
            ? std::abs((gpu_stats.total_energy - gpu_initial_energy) / gpu_initial_energy) * 100.0
            : 0.0;

        report.gpu_momentum = gpu_stats.momentum;

        report.kernel_speedup = report.gpu_elapsed_seconds > 0.0
            ? report.cpu_elapsed_seconds / report.gpu_elapsed_seconds
            : 0.0;

        report.speedup = report.gpu_total_seconds > 0.0
            ? report.cpu_elapsed_seconds / report.gpu_total_seconds
            : 0.0;

        report.gpu_gflops = report.gpu_elapsed_seconds > 0.0
            ? flops_total / report.gpu_elapsed_seconds * 1e-9
            : 0.0;

        const auto& cpu_particles = cpu_sim.particles();
        const auto& gpu_particles = gpu_sim.particles();

        if (cpu_particles.size() != gpu_particles.size()) {
            report.error_message = "CPU and OpenCL particle counts differ.";
            report.gpu_available = false;
            return report;
        }

        double sum_sq_pos_err = 0.0;
        double sum_sq_vel_err = 0.0;
        double sum_sq_ref_vel = 0.0;

        bool finite_results = true;

        for (std::size_t i = 0; i < cpu_particles.size(); ++i) {
            const Vec3 cpu_position = cpu_particles[i].position.xyz();
            const Vec3 gpu_position = gpu_particles[i].position.xyz();
            const Vec3 cpu_velocity = cpu_particles[i].velocity.xyz();
            const Vec3 gpu_velocity = gpu_particles[i].velocity.xyz();

            const Vec3 dp = gpu_position - cpu_position;
            const Vec3 dv = gpu_velocity - cpu_velocity;

            const double pos_err = dp.norm();
            const double vel_err = dv.norm();

            const double ref_position_norm = std::max(1.0, cpu_position.norm());
            const double ref_velocity_norm = std::max(1.0, cpu_velocity.norm());

            finite_results = finite_results
                && std::isfinite(pos_err)
                && std::isfinite(vel_err)
                && std::isfinite(gpu_particles[i].mass())
                && std::isfinite(gpu_position.x)
                && std::isfinite(gpu_position.y)
                && std::isfinite(gpu_position.z)
                && std::isfinite(gpu_velocity.x)
                && std::isfinite(gpu_velocity.y)
                && std::isfinite(gpu_velocity.z);

            report.max_abs_position_error = std::max(
                report.max_abs_position_error,
                pos_err
            );

            report.max_rel_position_error = std::max(
                report.max_rel_position_error,
                pos_err / ref_position_norm
            );

            report.max_abs_velocity_error = std::max(
                report.max_abs_velocity_error,
                vel_err
            );

            report.max_rel_velocity_error = std::max(
                report.max_rel_velocity_error,
                vel_err / ref_velocity_norm
            );

            sum_sq_pos_err += pos_err * pos_err;
            sum_sq_vel_err += vel_err * vel_err;
            sum_sq_ref_vel += cpu_velocity.norm_squared();
        }

        const double particle_count = static_cast<double>(cpu_particles.size());

        report.rms_position_error = particle_count > 0.0
            ? std::sqrt(sum_sq_pos_err / particle_count)
            : 0.0;

        report.rms_velocity_error = particle_count > 0.0
            ? std::sqrt(sum_sq_vel_err / particle_count)
            : 0.0;

        report.relative_rms_velocity_error = sum_sq_ref_vel > 0.0
            ? std::sqrt(sum_sq_vel_err / sum_sq_ref_vel)
            : (sum_sq_vel_err == 0.0 ? 0.0 : INFINITY);

        report.relative_energy_diff_pct = cpu_stats.total_energy != 0.0
            ? std::abs((gpu_stats.total_energy - cpu_stats.total_energy)
                / cpu_stats.total_energy) * 100.0
            : 0.0;

        const bool physical_stability =
            report.cpu_energy_drift_pct <= 0.25
            && report.gpu_energy_drift_pct <= (report.gpu_using_fp64 ? 0.25 : 1.0)
            && report.cpu_momentum <= 1e-8
            && report.gpu_momentum <= (report.gpu_using_fp64 ? 1e-7 : 1e-3);

        const bool numerical_equivalence = report.gpu_using_fp64
            ? report.max_rel_position_error <= 1e-6
                && report.max_rel_velocity_error <= 1e-5
                && report.relative_energy_diff_pct <= 1e-6
            : report.rms_position_error <= 2.5e-2
                && report.relative_rms_velocity_error <= 2.5e-2
                && report.relative_energy_diff_pct <= 0.1;

        report.passed = finite_results
            && physical_stability
            && numerical_equivalence;

        if (!report.passed) {
            report.error_message =
                "Numerical, conservation, or stability thresholds were not satisfied.";
        }

    } catch (const std::exception& ex) {
        report.gpu_available = false;
        report.passed = false;
        report.error_message = ex.what();

    } catch (...) {
        report.gpu_available = false;
        report.passed = false;
        report.error_message = "Unknown OpenCL validation error.";
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

    out << std::fixed << std::setprecision(6);
    out << "CPU Time         : " << report.cpu_elapsed_seconds << " s\n";
    out << "CPU Energy Drift : " << report.cpu_energy_drift_pct << " %\n";
    out << "CPU Momentum     : " << report.cpu_momentum << "\n";

    if (!report.gpu_available) {
        out << "OpenCL GPU Status: UNAVAILABLE\n";

        if (!report.error_message.empty()) {
            out << "Reason           : " << report.error_message << "\n";
        }

        return;
    }

    out << "OpenCL Device    : " << report.gpu_device_name
        << " (" << (report.gpu_using_fp64 ? "FP64" : "FP32") << ")\n";

    out << "\nPerformance\n";
    out << "------------------------------------------------------------\n";
    out << "GPU Kernel Time  : " << report.gpu_elapsed_seconds << " s\n";
    out << "GPU Transfer Time: " << report.gpu_transfer_seconds << " s\n";
    out << "GPU Total Time   : " << report.gpu_total_seconds << " s\n";
    out << "Kernel Speedup   : " << report.kernel_speedup << "x\n";
    out << "End-to-end Speedup: " << report.speedup << "x\n";
    out << "CPU GFLOP/s      : " << report.cpu_gflops << "\n";
    out << "GPU GFLOP/s      : " << report.gpu_gflops << "\n";

    out << "\nNumerical Accuracy\n";
    out << "------------------------------------------------------------\n";

    out << std::scientific << std::setprecision(6);
    out << "Max Abs Pos Err  : " << report.max_abs_position_error << "\n";
    out << "Max Rel Pos Err  : " << report.max_rel_position_error << "\n";
    out << "RMS Pos Err      : " << report.rms_position_error << "\n";
    out << "Max Abs Vel Err  : " << report.max_abs_velocity_error << "\n";
    out << "Max Rel Vel Err  : " << report.max_rel_velocity_error << "\n";
    out << "RMS Vel Err      : " << report.rms_velocity_error << "\n";
    out << "Relative RMS Vel : " << report.relative_rms_velocity_error << "\n";

    out << std::fixed << std::setprecision(6);
    out << "GPU Energy Drift : " << report.gpu_energy_drift_pct << " %\n";
    out << "Energy Diff (%)  : " << report.relative_energy_diff_pct << " %\n";
    out << "GPU Momentum     : " << report.gpu_momentum << "\n";
    out << "Validation       : " << (report.passed ? "PASS" : "FAIL") << "\n";

    if (!report.error_message.empty()) {
        out << "Reason           : " << report.error_message << "\n";
    }
}

}
