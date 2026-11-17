#include "astra/cpu_backend.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#ifdef ASTRA_HAS_OPENMP
#include <omp.h>
#endif

namespace astra {

void CPUBackend::calculate_accelerations(
    std::vector<Particle>& particles,
    double gravitational_constant,
    double softening
) {
    const std::size_t count = particles.size();
    const double softening_squared = softening * softening;

    #ifdef ASTRA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
        Vec3 acceleration{};
        const Vec3 position_i = particles[i].position.xyz();

        for (std::size_t j = 0; j < count; ++j) {
            if (static_cast<std::size_t>(i) == j) {
                continue;
            }

            const Vec3 displacement = particles[j].position.xyz() - position_i;

            const double distance_squared =
                displacement.x * displacement.x +
                displacement.y * displacement.y +
                displacement.z * displacement.z +
                softening_squared;

            const double inverse_distance = 1.0 / std::sqrt(distance_squared);
            const double inverse_distance_cubed =
                inverse_distance * inverse_distance * inverse_distance;

            const double factor =
                gravitational_constant *
                particles[j].mass() *
                inverse_distance_cubed;

            acceleration += displacement * factor;
        }

        particles[i].acceleration.set_xyz(acceleration);
        particles[i].acceleration.w = 0.0;
    }
}

void CPUBackend::integrate(
    std::vector<Particle>& particles,
    double time_step
) {
    const double half_step = 0.5 * time_step;

    #ifdef ASTRA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(particles.size()); ++i) {
        Particle& particle = particles[i];

        const Vec3 half_velocity =
            particle.velocity.xyz() + particle.acceleration.xyz() * half_step;

        particle.velocity.set_xyz(half_velocity);
        particle.position.add_xyz(half_velocity * time_step);
    }
}

void CPUBackend::finish_integration(
    std::vector<Particle>& particles,
    double time_step
) {
    const double half_step = 0.5 * time_step;

    #ifdef ASTRA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(particles.size()); ++i) {
        Particle& particle = particles[i];
        particle.velocity.add_xyz(particle.acceleration.xyz() * half_step);
    }
}

} // namespace astra