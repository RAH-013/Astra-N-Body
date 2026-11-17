
#pragma once

#include <vector>

#include "astra/particle.hpp"
#include "astra/simulation.hpp"

namespace astra {

class CPUBackend {
public:
    static void calculate_accelerations(
        std::vector<Particle>& particles,
        double gravitational_constant,
        double softening
    );

    static void integrate(
        std::vector<Particle>& particles,
        double time_step
    );

    static void finish_integration(
        std::vector<Particle>& particles,
        double time_step
    );
};

}
