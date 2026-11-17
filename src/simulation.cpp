
#include "astra/simulation.hpp"
#include "astra/cpu_backend.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>

namespace astra {

Simulation::Simulation(SimulationConfig config)
    : config_(config) {

    if (config_.particle_count < 2) {
        throw std::invalid_argument("At least two particles are required");
    }

    if (config_.time_step <= 0.0 ||
        config_.gravitational_constant <= 0.0 ||
        config_.softening < 0.0) {
        throw std::invalid_argument("Invalid simulation configuration");
    }

    particles_.reserve(config_.particle_count);
}


void Simulation::initialize_galaxy() {
    std::mt19937_64 generator(config_.seed);

    std::uniform_real_distribution<double> radius_distribution(0.15, 5.0);
    std::uniform_real_distribution<double> angle_distribution(
        0.0, 6.283185307179586
    );
    std::uniform_real_distribution<double> height_distribution(-0.08, 0.08);
    std::uniform_real_distribution<double> mass_distribution(0.5, 1.5);

    particles_.clear();

    Particle central_body;
    central_body.position = {0.0, 0.0, 0.0};
    central_body.velocity = {0.0, 0.0, 0.0};
    central_body.mass = 1000.0;

    particles_.push_back(central_body);

    for (std::size_t i = 0; i < config_.particle_count; ++i) {
        const double radius = radius_distribution(generator);
        const double angle = angle_distribution(generator);
        const double height = height_distribution(generator);

        const double x = radius * std::cos(angle);
        const double y = radius * std::sin(angle);

        const double orbital_speed =
            std::sqrt(config_.gravitational_constant * central_body.mass / radius);

        Particle particle;

        particle.position = {x, y, height};
        particle.velocity = {
            -std::sin(angle) * orbital_speed,
            std::cos(angle) * orbital_speed,
            0.0
        };

        particle.mass = mass_distribution(generator);

        particles_.push_back(particle);
    }

    calculate_accelerations();
    calculate_statistics();

    stats_.completed_steps = 0;
}

void Simulation::calculate_accelerations() {
    CPUBackend::calculate_accelerations(
        particles_,
        config_.gravitational_constant,
        config_.softening
    );
}

void Simulation::step() {
    CPUBackend::integrate(particles_, config_.time_step);

    calculate_accelerations();

    CPUBackend::finish_integration(particles_, config_.time_step);

    ++stats_.completed_steps;
}

void Simulation::run(std::size_t steps) {
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < steps; ++i) {
        step();
    }

    const auto end = std::chrono::steady_clock::now();

    stats_.elapsed_seconds =
        std::chrono::duration<double>(end - start).count();

    calculate_statistics();
}

void Simulation::calculate_statistics() {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;

    Vec3 total_momentum{};

    const std::size_t count = particles_.size();

    for (std::size_t i = 0; i < count; ++i) {
        const Particle& particle = particles_[i];

        const double velocity_squared =
            particle.velocity.x * particle.velocity.x +
            particle.velocity.y * particle.velocity.y +
            particle.velocity.z * particle.velocity.z;

        kinetic_energy +=
            0.5 * particle.mass * velocity_squared;

        total_momentum += particle.velocity * particle.mass;

        for (std::size_t j = i + 1; j < count; ++j) {
            const Vec3 displacement =
                particles_[j].position - particle.position;

            const double distance = std::sqrt(
                displacement.x * displacement.x +
                displacement.y * displacement.y +
                displacement.z * displacement.z +
                config_.softening * config_.softening
            );

            potential_energy -=
                config_.gravitational_constant *
                particle.mass *
                particles_[j].mass /
                distance;
        }
    }

    stats_.kinetic_energy = kinetic_energy;
    stats_.potential_energy = potential_energy;
    stats_.total_energy = kinetic_energy + potential_energy;

    stats_.momentum = std::sqrt(
        total_momentum.x * total_momentum.x +
        total_momentum.y * total_momentum.y +
        total_momentum.z * total_momentum.z
    );
}

const std::vector<Particle>& Simulation::particles() const {
    return particles_;
}

const SimulationConfig& Simulation::config() const {
    return config_;
}

const SimulationStats& Simulation::stats() const {
    return stats_;
}

}
