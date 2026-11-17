#include <cmath>
#include "astra/simulation.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* executable) {
    std::cout
        << "ASTRA N-BODY v0.1.0\n\n"
        << "Usage:\n"
        << "  " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --particles N    Number of particles (default: 1024)\n"
        << "  --steps N        Simulation iterations (default: 100)\n"
        << "  --dt VALUE       Time step (default: 0.001)\n"
        << "  --softening X    Gravitational softening (default: 0.01)\n"
        << "  --seed N         Random seed (default: 42)\n"
        << "  --help           Show this message\n";
}

std::size_t parse_size(const std::string& value) {
    std::size_t position = 0;
    const unsigned long long parsed = std::stoull(value, &position);

    if (position != value.size() || value.empty() || value[0] == '-') {
        throw std::invalid_argument("Invalid positive integer: " + value);
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

}

int main(int argc, char* argv[]) {
    try {
        astra::SimulationConfig config;

        std::size_t steps = 100;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];

            if (argument == "--help") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }

            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value for " + argument
                );
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
            } else {
                throw std::invalid_argument(
                    "Unknown argument: " + argument
                );
            }
        }

        astra::Simulation simulation(config);

        std::cout << "\nASTRA N-BODY v0.1.0\n";
        std::cout << "==============================\n";
        std::cout << "Backend: CPU";
        
        #ifdef ASTRA_HAS_OPENMP
        std::cout << " (OpenMP)";
        #endif

        std::cout << "\n";
        std::cout << "Particles: " << config.particle_count << "\n";
        std::cout << "Steps: " << steps << "\n";
        std::cout << "Time step: " << config.time_step << "\n";
        std::cout << "Softening: " << config.softening << "\n";

        simulation.initialize_galaxy();

        const double initial_energy =
            simulation.stats().total_energy;

        std::cout << "\nRunning simulation...\n";

        simulation.run(steps);

        const auto& stats = simulation.stats();

        std::cout << std::fixed << std::setprecision(8);

        std::cout << "\nSimulation completed\n";
        std::cout << "==============================\n";

        std::cout << "Elapsed time: "
                  << stats.elapsed_seconds << " s\n";

        std::cout << "Average step time: "
                  << (steps > 0
                      ? stats.elapsed_seconds / steps * 1000.0
                      : 0.0)
                  << " ms\n";

        std::cout << "Completed steps: "
                  << stats.completed_steps << "\n";

        std::cout << "\nPhysical statistics\n";
        std::cout << "------------------------------\n";

        std::cout << "Kinetic energy: "
                  << stats.kinetic_energy << "\n";

        std::cout << "Potential energy: "
                  << stats.potential_energy << "\n";

        std::cout << "Total energy: "
                  << stats.total_energy << "\n";

        std::cout << "Initial energy: "
                  << initial_energy << "\n";

        std::cout << "Momentum magnitude: "
                  << stats.momentum << "\n";

        const double energy_drift =
            initial_energy != 0.0
                ? std::abs(
                    (stats.total_energy - initial_energy) /
                    initial_energy
                  ) * 100.0
                : 0.0;

        std::cout << "Energy drift: "
                  << energy_drift << " %\n";

        std::cout << "\nASTRA N-BODY finished successfully.\n";

        return EXIT_SUCCESS;

    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }
}
