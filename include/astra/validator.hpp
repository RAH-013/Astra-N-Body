
#pragma once

#include <cstddef>
#include <ostream>
#include <string>

#include "astra/simulation.hpp"

namespace astra {

struct ValidationReport {
    bool gpu_available = false;
    bool gpu_using_fp64 = false;
    std::string gpu_device_name;
    std::string error_message;

    double max_abs_position_error = 0.0;
    double max_rel_position_error = 0.0;
    double rms_position_error = 0.0;
    double max_abs_velocity_error = 0.0;
    double max_rel_velocity_error = 0.0;
    double rms_velocity_error = 0.0;
    double relative_rms_velocity_error = 0.0;

    double cpu_energy_drift_pct = 0.0;
    double gpu_energy_drift_pct = 0.0;
    double relative_energy_diff_pct = 0.0;

    double cpu_elapsed_seconds = 0.0;
    double gpu_elapsed_seconds = 0.0;
    double gpu_transfer_seconds = 0.0;
    double speedup = 0.0;
    double kernel_speedup = 0.0;
    double gpu_total_seconds = 0.0;
    double cpu_momentum = 0.0;
    double gpu_momentum = 0.0;
    double cpu_gflops = 0.0;
    double gpu_gflops = 0.0;

    bool passed = false;
};

class Validator {
public:
    static ValidationReport compare_cpu_and_opencl(
        const SimulationConfig& base_config,
        std::size_t steps,
        bool run_warmup = true
    );

    static void print_report(
        const ValidationReport& report,
        std::size_t particle_count,
        std::size_t steps,
        std::ostream& out
    );
};

}
