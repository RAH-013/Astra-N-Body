
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "astra/particle.hpp"

namespace astra {

struct SimulationConfig {
    double gravitational_constant = 1.0;
    double softening = 0.01;
    double time_step = 0.001;
    std::size_t particle_count = 1024;
    std::uint64_t seed = 42;
};

struct SimulationStats {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    double total_energy = 0.0;
    double momentum = 0.0;
    double elapsed_seconds = 0.0;
    std::uint64_t completed_steps = 0;
};

class Simulation {
public:
    explicit Simulation(SimulationConfig config);

    void initialize_galaxy();
    void step();
    void run(std::size_t steps);

    const std::vector<Particle>& particles() const;
    const SimulationConfig& config() const;
    const SimulationStats& stats() const;

    void calculate_statistics();

private:
    void calculate_accelerations();

    SimulationConfig config_;
    std::vector<Particle> particles_;
    SimulationStats stats_;
};

}
