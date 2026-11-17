#include "astra/opencl_device.hpp"
#include "astra/particle.hpp"
#include "astra/simulation.hpp"
#include "astra/validator.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

    astra::Particle p{};
    p.position = {1.0, 2.0, 3.0, 42.5};
    require(p.mass() == 42.5, "Particle::mass() must read position.w");
    p.set_mass(99.0);
    require(p.position.w == 99.0, "Particle::set_mass() must write position.w");
}

void test_exact_particle_count_and_stability() {
    astra::SimulationConfig cfg;
    cfg.particle_count = 4096;
    cfg.time_step = 0.001;
    cfg.softening = 0.01;
    cfg.seed = 42;
    cfg.requested_backend = astra::BackendType::CPU;

    astra::Simulation sim(cfg);
    sim.initialize_galaxy();

    require(sim.particles().size() == 4096, "Actual particle count must equal requested count (4096)");
    require(sim.stats().momentum < 1e-10, "Initial center-of-mass momentum must be zero");

    const double e0 = sim.stats().total_energy;
    sim.run(100);
    const double e1 = sim.stats().total_energy;

    const double drift_pct = std::abs((e1 - e0) / e0) * 100.0;
    std::cout << "  [N=4096 CPU] Energy drift after 100 steps: " << drift_pct << " %\n";
    require(drift_pct < 0.25, "Energy drift at N=4096 must remain well below 0.25%");
    require(sim.stats().momentum < 1e-9, "Linear momentum must remain conserved");
}

void test_two_body_circular_orbit() {
    astra::SimulationConfig cfg;
    cfg.particle_count = 2;
    cfg.gravitational_constant = 1.0;
    cfg.softening = 0.0;
    cfg.time_step = 0.0005;
    cfg.requested_backend = astra::BackendType::CPU;

    astra::Simulation sim(cfg);

    // Equal-mass binary on circular orbit of separation d = 2.0 (r = 1.0 each)
    // v = sqrt(G * m / (4 * r)) = sqrt(1.0 / 4.0) = 0.5
    std::vector<astra::Particle> bodies(2);
    bodies[0].position = {-1.0, 0.0, 0.0, 1.0};
    bodies[0].velocity = {0.0, -0.5, 0.0, 0.0};
    bodies[1].position = {1.0, 0.0, 0.0, 1.0};
    bodies[1].velocity = {0.0, 0.5, 0.0, 0.0};

    sim.set_particles(bodies);
    const double e0 = sim.stats().total_energy;
    sim.run(2000);
    const double e1 = sim.stats().total_energy;

    const double drift_pct = std::abs((e1 - e0) / e0) * 100.0;
    std::cout << "  [Two-body] Energy drift after 2000 steps: " << drift_pct << " %\n";
    require(drift_pct < 1e-4, "Two-body circular orbit energy drift must be < 1e-4%");
}

void test_opencl_equivalence_if_available() {
    const auto devices = astra::OpenCLDeviceManager::enumerate_devices();
    if (devices.empty()) {
        std::cout << "  [OpenCL] No OpenCL device detected in current environment (skipping GPU comparison).\n";
        return;
    }

    astra::SimulationConfig cfg;
    cfg.particle_count = 256;
    cfg.time_step = 0.001;
    cfg.softening = 0.01;
    cfg.seed = 42;

    const auto report = astra::Validator::compare_cpu_and_opencl(cfg, 25, false);
    require(report.gpu_available, "OpenCL device was enumerated but failed to initialize");
    std::cout << "  [OpenCL] Device: " << report.gpu_device_name
              << " | FP64: " << (report.gpu_using_fp64 ? "YES" : "NO")
              << " | MaxRelPosErr: " << report.max_rel_position_error << "\n";
    require(report.passed, "CPU vs OpenCL numerical validation must pass");
}

} // namespace

int main() {
    try {
        std::cout << "Running ASTRA N-BODY v0.2.0 test suite...\n";
        test_shared_memory_layout();
        test_two_body_circular_orbit();
        test_exact_particle_count_and_stability();
        test_opencl_equivalence_if_available();
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Test failure: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}