#include "astra/opencl_device.hpp"
#include "astra/particle.hpp"
#include "astra/simulation.hpp"
#include "astra/validator.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Assertion failed: ") + message);
    }
}

void test_shared_memory_layout() {
    require(sizeof(astra::Vec4) == 32, "Vec4 size must be 32 bytes");
    require(alignof(astra::Vec4) == 32, "Vec4 alignment must be 32 bytes");
    require(sizeof(astra::Particle) == 96, "Particle size must be 96 bytes");
    require(alignof(astra::Particle) == 32, "Particle alignment must be 32 bytes");

    astra::Particle particle{};
    particle.position = {1.0, 2.0, 3.0, 42.5};
    require(particle.mass() == 42.5, "Particle mass must use position.w");
    particle.set_mass(99.0);
    require(particle.position.w == 99.0, "Particle mass setter must update position.w");
}

void test_exact_particle_count_and_conservation() {
    astra::SimulationConfig config;
    config.particle_count = 4096;
    config.time_step = 0.001;
    config.softening = 0.01;
    config.seed = 42;
    config.requested_backend = astra::BackendType::CPU;

    astra::Simulation simulation(config);
    simulation.initialize_galaxy();
    require(simulation.particles().size() == config.particle_count, "Particle count must match configuration");
    require(simulation.stats().momentum < 1e-10, "Initial linear momentum must be near zero");

    const double initial_energy = simulation.stats().total_energy;
    simulation.run(100);
    const double final_energy = simulation.stats().total_energy;
    const double drift = std::abs((final_energy - initial_energy) / initial_energy) * 100.0;

    std::cout << "  [N=4096 CPU] Energy drift after 100 steps: " << drift << " %\n";
    require(drift < 0.25, "Energy drift must remain below 0.25 percent");
    require(simulation.stats().momentum < 1e-9, "Linear momentum must remain conserved");

    for (const auto& particle : simulation.particles()) {
        require(std::isfinite(particle.position.x), "Particle x position must be finite");
        require(std::isfinite(particle.position.y), "Particle y position must be finite");
        require(std::isfinite(particle.position.z), "Particle z position must be finite");
    }
}

void test_two_body_circular_orbit() {
    astra::SimulationConfig config;
    config.particle_count = 2;
    config.gravitational_constant = 1.0;
    config.softening = 0.0;
    config.time_step = 0.0005;
    config.requested_backend = astra::BackendType::CPU;

    astra::Simulation simulation(config);
    std::vector<astra::Particle> bodies(2);
    bodies[0].position = {-1.0, 0.0, 0.0, 1.0};
    bodies[0].velocity = {0.0, -0.5, 0.0, 0.0};
    bodies[1].position = {1.0, 0.0, 0.0, 1.0};
    bodies[1].velocity = {0.0, 0.5, 0.0, 0.0};

    simulation.set_particles(bodies);
    const double initial_energy = simulation.stats().total_energy;
    simulation.run(2000);
    const double final_energy = simulation.stats().total_energy;
    const double drift = std::abs((final_energy - initial_energy) / initial_energy) * 100.0;

    std::cout << "  [Two-body] Energy drift after 2000 steps: " << drift << " %\n";
    require(drift < 1e-4, "Two-body circular orbit energy drift must be below 1e-4 percent");
    require(simulation.stats().momentum < 1e-10, "Binary linear momentum must be conserved");
}

void test_seed_reproducibility() {
    astra::SimulationConfig config;
    config.particle_count = 128;
    config.seed = 731;
    config.requested_backend = astra::BackendType::CPU;

    astra::Simulation first(config);
    astra::Simulation second(config);
    first.initialize_galaxy();
    second.initialize_galaxy();

    require(first.particles().size() == second.particles().size(), "Seeded runs must have equal particle counts");
    for (std::size_t i = 0; i < first.particles().size(); ++i) {
        require(first.particles()[i].position.x == second.particles()[i].position.x, "Seed must reproduce x positions");
        require(first.particles()[i].position.y == second.particles()[i].position.y, "Seed must reproduce y positions");
        require(first.particles()[i].velocity.x == second.particles()[i].velocity.x, "Seed must reproduce velocities");
    }
}

void test_opencl_equivalence_if_gpu_available() {
    const auto devices = astra::OpenCLDeviceManager::enumerate_devices();
    bool gpu_found = false;
    for (const auto& device : devices) {
        gpu_found = gpu_found || device.category == astra::OpenCLDeviceCategory::GPU;
    }
    if (!gpu_found) {
        std::cout << "  [OpenCL] No OpenCL GPU detected; GPU equivalence test skipped.\n";
        return;
    }

    astra::SimulationConfig config;
    config.particle_count = 256;
    config.time_step = 0.001;
    config.softening = 0.01;
    config.seed = 42;

    for (const auto precision : {astra::PrecisionMode::FP32, astra::PrecisionMode::FP64}) {
        if (precision == astra::PrecisionMode::FP64) {
            bool fp64_available = false;
            for (const auto& device : devices) {
                fp64_available = fp64_available || (device.category == astra::OpenCLDeviceCategory::GPU && device.supports_fp64);
            }
            if (!fp64_available) {
                std::cout << "  [OpenCL FP64] Unsupported by detected GPU; test skipped.\n";
                continue;
            }
        }

        config.requested_precision = precision;
        const auto report = astra::Validator::compare_cpu_and_opencl(config, 25, false);
        require(report.gpu_available, "Detected OpenCL GPU must initialize successfully for requested precision");
        require(report.passed, "CPU/OpenCL numerical validation must pass for requested precision");
        std::cout << "  [OpenCL] Device: " << report.gpu_device_name
                  << " | Precision: " << (report.gpu_using_fp64 ? "FP64" : "FP32")
                  << " | Relative position error: " << report.max_rel_position_error
                  << " | Relative velocity error: " << report.max_rel_velocity_error << "\n";
    }
}

}

int main() {
    try {
        std::cout << "Running ASTRA N-BODY test suite...\n";
        test_shared_memory_layout();
        test_two_body_circular_orbit();
        test_exact_particle_count_and_conservation();
        test_seed_reproducibility();
        test_opencl_equivalence_if_gpu_available();
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }
}
