#include "astra/opencl_device.hpp"
#include "astra/simulation.hpp"
#include "astra/validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* executable) {
    std::cout
        << "ASTRA N-BODY v" << ASTRA_VERSION_STRING << "\n\n"
        << "Usage:\n"
        << "  " << executable << " [options]\n\n"
        << "Simulation:\n"
        << "  --particles N        Exact number of simulated particles (default: 1024)\n"
        << "  --steps N            Simulation iterations (default: 100)\n"
        << "  --dt VALUE           Integration time step (default: 0.001)\n"
        << "  --softening VALUE    Gravitational softening (default: 0.01)\n"
        << "  --seed N             Random generator seed (default: 42)\n"
        << "  --no-central-body    Generate a self-gravitating disk without central mass\n\n"
        << "Backend:\n"
        << "  --backend MODE       auto | cpu | opencl (default: auto)\n"
        << "  --precision MODE     auto | fp64 | fp32 (default: auto)\n"
        << "  --platform N         OpenCL platform index\n"
        << "  --device N           OpenCL device index\n"
        << "  --kernel PATH        OpenCL kernel source path\n"
        << "  --no-fallback        Do not fall back to CPU\n"
        << "  --list-devices       List OpenCL devices\n\n"
        << "Output:\n"
        << "  --export-csv PATH    Export final particle positions and velocities\n"
        << "  --trajectory PATH    Export particle snapshots during simulation\n"
        << "  --snapshot-every N   Save a snapshot every N steps (default: 10)\n\n"
        << "Verification:\n"
        << "  --validate           Compare CPU and OpenCL numerical results\n"
        << "  --benchmark          Compare CPU and OpenCL performance\n"
        << "  --help               Show this message\n";
}

std::size_t parse_size(const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("Invalid non-negative integer: " + value);
    }
    std::size_t position = 0;
    const unsigned long long parsed = std::stoull(value, &position);
    if (position != value.size()) {
        throw std::invalid_argument("Invalid integer: " + value);
    }
    return static_cast<std::size_t>(parsed);
}

double parse_double(const std::string& value) {
    std::size_t position = 0;
    const double parsed = std::stod(value, &position);
    if (position != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("Invalid number: " + value);
    }
    return parsed;
}

astra::BackendType parse_backend(const std::string& value) {
    if (value == "auto") return astra::BackendType::Auto;
    if (value == "cpu") return astra::BackendType::CPU;
    if (value == "opencl" || value == "gpu") return astra::BackendType::OpenCL;
    throw std::invalid_argument("Invalid backend (expected auto|cpu|opencl): " + value);
}

astra::PrecisionMode parse_precision(const std::string& value) {
    if (value == "auto") return astra::PrecisionMode::Auto;
    if (value == "fp64" || value == "double") return astra::PrecisionMode::FP64;
    if (value == "fp32" || value == "float") return astra::PrecisionMode::FP32;
    throw std::invalid_argument("Invalid precision (expected auto|fp64|fp32): " + value);
}

void write_snapshot(
    const std::filesystem::path& path,
    const astra::Simulation& simulation,
    std::size_t frame,
    std::size_t step,
    bool append
) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(path, append ? std::ios::app : std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Unable to open CSV output: " + path.string());
    }

    if (!append) {
        output << "frame,step,particle_id,x,y,z,mass,vx,vy,vz\n";
    }

    output << std::setprecision(17);
    const auto& particles = simulation.particles();
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        output << frame << ',' << step << ',' << i << ','
               << particle.position.x << ','
               << particle.position.y << ','
               << particle.position.z << ','
               << particle.mass() << ','
               << particle.velocity.x << ','
               << particle.velocity.y << ','
               << particle.velocity.z << '\n';
    }
}

}

int main(int argc, char* argv[]) {
    try {
        astra::SimulationConfig config;
        std::size_t steps = 100;
        std::size_t snapshot_every = 10;
        bool list_devices = false;
        bool run_validate = false;
        bool run_benchmark = false;
        std::string export_csv;
        std::string trajectory_csv;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];

            if (argument == "--help") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (argument == "--list-devices") {
                list_devices = true;
                continue;
            }
            if (argument == "--no-fallback") {
                config.allow_cpu_fallback = false;
                continue;
            }
            if (argument == "--no-central-body") {
                config.include_central_body = false;
                continue;
            }
            if (argument == "--validate") {
                run_validate = true;
                continue;
            }
            if (argument == "--benchmark") {
                run_benchmark = true;
                continue;
            }

            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + argument);
            }
            const std::string value = argv[++i];

            if (argument == "--particles") {
                config.particle_count = parse_size(value);
            } else if (argument == "--steps") {
                steps = parse_size(value);
            } else if (argument == "--dt") {
                config.time_step = parse_double(value);
            } else if (argument == "--softening") {
                config.softening = parse_double(value);
            } else if (argument == "--seed") {
                config.seed = parse_size(value);
            } else if (argument == "--backend") {
                config.requested_backend = parse_backend(value);
            } else if (argument == "--precision") {
                config.requested_precision = parse_precision(value);
            } else if (argument == "--platform") {
                config.opencl_platform_index = parse_size(value);
            } else if (argument == "--device") {
                config.opencl_device_index = parse_size(value);
            } else if (argument == "--kernel") {
                config.kernel_path = value;
            } else if (argument == "--export-csv") {
                export_csv = value;
            } else if (argument == "--trajectory") {
                trajectory_csv = value;
            } else if (argument == "--snapshot-every") {
                snapshot_every = parse_size(value);
            } else {
                throw std::invalid_argument("Unknown argument: " + argument);
            }
        }

        if (list_devices) {
            const auto devices = astra::OpenCLDeviceManager::enumerate_devices();
            astra::OpenCLDeviceManager::print_device_list(devices, std::cout);
            return astra::OpenCLDeviceManager::is_opencl_compiled() ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        if (run_validate || run_benchmark) {
            const auto report = astra::Validator::compare_cpu_and_opencl(
                config,
                steps,
                run_benchmark
            );
            astra::Validator::print_report(report, config.particle_count, steps, std::cout);
            return report.gpu_available && report.passed ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        if (snapshot_every == 0) {
            throw std::invalid_argument("--snapshot-every must be greater than zero");
        }
        if (!trajectory_csv.empty() && !export_csv.empty() && trajectory_csv == export_csv) {
            throw std::invalid_argument("--trajectory and --export-csv must use different paths");
        }

        astra::Simulation simulation(config);
        std::cout << "\nASTRA N-BODY v" << ASTRA_VERSION_STRING << "\n";
        std::cout << "============================================================\n";
        std::cout << "Active Backend : " << simulation.active_backend_description() << "\n";
        if (simulation.active_opencl_device().has_value()) {
            astra::OpenCLDeviceManager::print_device_summary(*simulation.active_opencl_device(), std::cout);
        }
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "Particles      : " << config.particle_count << "\n";
        std::cout << "Steps          : " << steps << "\n";
        std::cout << "Time step      : " << config.time_step << "\n";
        std::cout << "Softening      : " << config.softening << "\n";
        std::cout << "Seed           : " << config.seed << "\n";

        simulation.initialize_galaxy();
        const double initial_energy = simulation.stats().total_energy;

        if (!trajectory_csv.empty()) {
            write_snapshot(trajectory_csv, simulation, 0, 0, false);
        }

        std::cout << "\nRunning simulation...\n";
        if (trajectory_csv.empty()) {
            simulation.run(steps);
        } else {
            std::size_t completed = 0;
            std::size_t frame = 1;
            while (completed < steps) {
                const std::size_t chunk = std::min(snapshot_every, steps - completed);
                simulation.run(chunk);
                completed += chunk;
                write_snapshot(trajectory_csv, simulation, frame++, completed, true);
            }
        }

        if (!export_csv.empty()) {
            write_snapshot(export_csv, simulation, 0, steps, false);
        }

        const auto& stats = simulation.stats();
        const double energy_drift = initial_energy != 0.0
            ? std::abs((stats.total_energy - initial_energy) / initial_energy) * 100.0
            : 0.0;

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "\nSimulation completed\n";
        std::cout << "============================================================\n";
        std::cout << "Actual particles  : " << simulation.particles().size() << "\n";
        std::cout << "Elapsed time      : " << stats.elapsed_seconds << " s\n";
        std::cout << "Average step time : "
                  << (steps > 0 ? stats.elapsed_seconds / static_cast<double>(steps) * 1000.0 : 0.0)
                  << " ms\n";
        if (simulation.active_backend() == astra::BackendType::OpenCL) {
            std::cout << "Host transfer time: " << stats.host_transfer_seconds << " s\n";
        }
        std::cout << "Completed steps   : " << stats.completed_steps << "\n";
        std::cout << "\nPhysical statistics\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "Kinetic energy    : " << stats.kinetic_energy << "\n";
        std::cout << "Potential energy  : " << stats.potential_energy << "\n";
        std::cout << "Total energy      : " << stats.total_energy << "\n";
        std::cout << "Initial energy    : " << initial_energy << "\n";
        std::cout << "Momentum magnitude: " << stats.momentum << "\n";
        std::cout << "Energy drift      : " << energy_drift << " %\n";
        if (!export_csv.empty()) {
            std::cout << "Final CSV         : " << export_csv << "\n";
        }
        if (!trajectory_csv.empty()) {
            std::cout << "Trajectory CSV    : " << trajectory_csv << "\n";
        }
        std::cout << "\nASTRA N-BODY finished successfully.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }
}
