#include "astra/simulation.hpp"
#include "astra/cpu_backend.hpp"
#include "astra/opencl_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef ASTRA_HAS_OPENMP
#include <omp.h>
#endif

namespace astra {

Simulation::Simulation(SimulationConfig config)
    : config_(std::move(config)) {
    if (config_.particle_count < 2) {
        throw std::invalid_argument("At least two particles are required");
    }

    if (config_.time_step <= 0.0 ||
        config_.gravitational_constant <= 0.0 ||
        config_.softening < 0.0) {
        throw std::invalid_argument("Invalid simulation configuration");
    }

    particles_.reserve(config_.particle_count);
    initialize_backend();
}

Simulation::~Simulation() = default;

void Simulation::initialize_backend() {
    if (config_.requested_backend == BackendType::CPU) {
        active_backend_ = BackendType::CPU;
        return;
    }

    const bool require_fp64 = (config_.requested_precision == PrecisionMode::FP64);

    try {
        if (!OpenCLDeviceManager::is_opencl_compiled()) {
            throw std::runtime_error("OpenCL support was not compiled into this binary.");
        }

        auto selected = OpenCLDeviceManager::select_best_device(
            config_.opencl_platform_index,
            config_.opencl_device_index,
            require_fp64
        );

        if (!selected.has_value()) {
            throw std::runtime_error(
                require_fp64
                    ? "No OpenCL device with cl_khr_fp64 support found."
                    : "No compatible OpenCL device found."
            );
        }

        opencl_backend_ = std::make_unique<OpenCLBackend>(
            *selected,
            config_.particle_count,
            config_.requested_precision,
            config_.kernel_path
        );

        active_device_ = *selected;
        active_backend_ = BackendType::OpenCL;
    } catch (const std::exception& ex) {
        opencl_backend_.reset();
        active_device_.reset();

        if (config_.requested_backend == BackendType::OpenCL && !config_.allow_cpu_fallback) {
            throw;
        }

        if (config_.requested_backend == BackendType::OpenCL) {
            std::cerr << "[Warning] OpenCL initialization failed (" << ex.what()
                      << "). Falling back to CPU backend.\n";
        }
        active_backend_ = BackendType::CPU;
    }
}

void Simulation::initialize_galaxy() {
    std::mt19937_64 generator(config_.seed);

    // Uniform-surface-density disk prevents central crowding and resolves softening timescales
    const double r_min = std::max(0.5, 5.0 * config_.softening);
    const double r_max = 5.0;
    const double r_min_sq = r_min * r_min;
    const double r_max_sq = r_max * r_max;
    const double half_thickness = std::max(0.15, 3.0 * config_.softening);

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> angle_distribution(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> height_distribution(-half_thickness, half_thickness);
    std::uniform_real_distribution<double> mass_distribution(0.5, 1.5);
    std::normal_distribution<double> normal01(0.0, 1.0);

    particles_.clear();
    particles_.reserve(config_.particle_count);

    const std::size_t disk_count = config_.include_central_body
        ? (config_.particle_count - 1)
        : config_.particle_count;

    constexpr double central_mass = 1000.0;
    constexpr double max_disk_mass = 500.0;

    if (config_.include_central_body) {
        Particle central_body{};
        central_body.position = {0.0, 0.0, 0.0, central_mass};
        central_body.velocity = {0.0, 0.0, 0.0, 0.0};
        central_body.acceleration = {0.0, 0.0, 0.0, 0.0};
        particles_.push_back(central_body);
    }

    std::vector<double> raw_masses(disk_count);
    double raw_mass_sum = 0.0;
    for (std::size_t i = 0; i < disk_count; ++i) {
        raw_masses[i] = mass_distribution(generator);
        raw_mass_sum += raw_masses[i];
    }
    const double mass_norm = (raw_mass_sum > 0.0)
        ? std::min(1.0, max_disk_mass / raw_mass_sum)
        : 1.0;

    const double min_sep = 3.0 * config_.softening;
    const double min_sep_sq = min_sep * min_sep;

    for (std::size_t i = 0; i < disk_count; ++i) {
        const double radius = std::sqrt(r_min_sq + u01(generator) * (r_max_sq - r_min_sq));
        double angle = angle_distribution(generator);
        double height = height_distribution(generator);
        const double mass = raw_masses[i] * mass_norm;

        double x = radius * std::cos(angle);
        double y = radius * std::sin(angle);

        // Enforce minimum initial separation to prevent sub-softening numerical shocks at t=0
        for (int attempt = 0; attempt < 64; ++attempt) {
            bool too_close = false;
            for (std::size_t j = 0; j < particles_.size(); ++j) {
                const double dx = particles_[j].position.x - x;
                const double dy = particles_[j].position.y - y;
                const double dz = particles_[j].position.z - height;
                if ((dx * dx + dy * dy + dz * dz) < min_sep_sq) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                break;
            }
            angle = angle_distribution(generator);
            height = height_distribution(generator);
            x = radius * std::cos(angle);
            y = radius * std::sin(angle);
        }

        Particle particle{};
        particle.position = {x, y, height, mass};
        particle.velocity = {0.0, 0.0, 0.0, 0.0};
        particle.acceleration = {0.0, 0.0, 0.0, 0.0};
        particles_.push_back(particle);
    }

    // Evaluate azimuthally averaged inward radial acceleration at each particle's radius R_i
    // so close-neighbor fluctuations never zero out orbital angular momentum
    const std::size_t start_idx = config_.include_central_body ? 1 : 0;
    const std::size_t total_count = particles_.size();
    const double eps2 = config_.softening * config_.softening;
    const double g_const = config_.gravitational_constant;
    constexpr int ring_samples = 16;

    std::vector<double> circular_speeds(total_count, 0.0);

    #ifdef ASTRA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(start_idx);
         idx < static_cast<std::ptrdiff_t>(total_count);
         ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx);
        const Particle& p = particles_[i];
        const double cyl_r = std::hypot(p.position.x, p.position.y);
        if (cyl_r <= 0.0) {
            continue;
        }

        double inward_accel_sum = 0.0;
        for (int s = 0; s < ring_samples; ++s) {
            const double phi = (6.283185307179586 * static_cast<double>(s)) / ring_samples;
            const double cos_p = std::cos(phi);
            const double sin_p = std::sin(phi);
            const double sx = cyl_r * cos_p;
            const double sy = cyl_r * sin_p;
            const double sz = p.position.z;

            double ax = 0.0;
            double ay = 0.0;
            for (std::size_t j = 0; j < total_count; ++j) {
                if (i == j) {
                    continue;
                }
                const double dx = particles_[j].position.x - sx;
                const double dy = particles_[j].position.y - sy;
                const double dz = particles_[j].position.z - sz;
                const double r2 = dx * dx + dy * dy + dz * dz + eps2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double factor = g_const * particles_[j].mass() * inv_r * inv_r * inv_r;
                ax += dx * factor;
                ay += dy * factor;
            }
            inward_accel_sum += -(ax * cos_p + ay * sin_p);
        }

        const double mean_inward_accel = inward_accel_sum / ring_samples;
        circular_speeds[i] = (mean_inward_accel > 0.0)
            ? std::sqrt(cyl_r * mean_inward_accel)
            : 0.0;
    }

    for (std::size_t i = start_idx; i < total_count; ++i) {
        Particle& p = particles_[i];
        const double cyl_r = std::hypot(p.position.x, p.position.y);
        if (cyl_r <= 0.0) {
            continue;
        }

        const double cos_theta = p.position.x / cyl_r;
        const double sin_theta = p.position.y / cyl_r;
        const double v_circ = circular_speeds[i];

        const double v_radial = 0.03 * v_circ * normal01(generator);
        const double v_z = 0.015 * v_circ * normal01(generator);

        p.velocity = {
            v_radial * cos_theta - v_circ * sin_theta,
            v_radial * sin_theta + v_circ * cos_theta,
            v_z,
            0.0
        };
    }

    // Transform into exact Center-of-Mass rest frame (zero net linear momentum)
    double total_mass = 0.0;
    Vec3 com_pos{};
    Vec3 com_vel{};
    for (const auto& p : particles_) {
        const double m = p.mass();
        total_mass += m;
        com_pos += p.position.xyz() * m;
        com_vel += p.velocity.xyz() * m;
    }
    if (total_mass > 0.0) {
        com_pos = com_pos * (1.0 / total_mass);
        com_vel = com_vel * (1.0 / total_mass);
        for (auto& p : particles_) {
            p.position.set_xyz(p.position.xyz() - com_pos);
            p.velocity.set_xyz(p.velocity.xyz() - com_vel);
        }
    }

    calculate_accelerations();

    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        const auto t0 = std::chrono::steady_clock::now();
        opencl_backend_->upload_particles(particles_);
        const auto t1 = std::chrono::steady_clock::now();
        stats_.host_transfer_seconds = std::chrono::duration<double>(t1 - t0).count();
        device_state_dirty_ = false;
    }

    calculate_statistics();
    stats_.completed_steps = 0;
}

void Simulation::set_particles(std::vector<Particle> particles) {
    if (particles.size() != config_.particle_count) {
        throw std::invalid_argument("Particle count does not match SimulationConfig");
    }

    particles_ = std::move(particles);
    calculate_accelerations();

    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        const auto t0 = std::chrono::steady_clock::now();
        opencl_backend_->upload_particles(particles_);
        const auto t1 = std::chrono::steady_clock::now();
        stats_.host_transfer_seconds = std::chrono::duration<double>(t1 - t0).count();
        device_state_dirty_ = false;
    }

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
    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        opencl_backend_->step(
            config_.gravitational_constant,
            config_.softening,
            config_.time_step
        );
        opencl_backend_->synchronize();
        device_state_dirty_ = true;
    } else {
        CPUBackend::integrate(particles_, config_.time_step);
        calculate_accelerations();
        CPUBackend::finish_integration(particles_, config_.time_step);
    }

    ++stats_.completed_steps;
}

void Simulation::run(std::size_t steps) {
    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        const auto start = std::chrono::steady_clock::now();
        opencl_backend_->run_steps(
            steps,
            config_.gravitational_constant,
            config_.softening,
            config_.time_step
        );
        const auto end = std::chrono::steady_clock::now();

        stats_.elapsed_seconds = std::chrono::duration<double>(end - start).count();
        stats_.completed_steps += steps;

        const auto dl_start = std::chrono::steady_clock::now();
        opencl_backend_->download_particles(particles_);
        const auto dl_end = std::chrono::steady_clock::now();
        stats_.host_transfer_seconds += std::chrono::duration<double>(dl_end - dl_start).count();
        device_state_dirty_ = false;
    } else {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < steps; ++i) {
            CPUBackend::integrate(particles_, config_.time_step);
            calculate_accelerations();
            CPUBackend::finish_integration(particles_, config_.time_step);
            ++stats_.completed_steps;
        }
        const auto end = std::chrono::steady_clock::now();
        stats_.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    }

    calculate_statistics();
}

void Simulation::calculate_statistics() {
    if (active_backend_ == BackendType::OpenCL && opencl_backend_ && device_state_dirty_) {
        opencl_backend_->download_particles(particles_);
        device_state_dirty_ = false;
    }

    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    Vec3 total_momentum{};

    const std::size_t count = particles_.size();
    const double softening_sq = config_.softening * config_.softening;

    for (std::size_t i = 0; i < count; ++i) {
        const Particle& particle = particles_[i];
        const double mass_i = particle.mass();

        kinetic_energy += 0.5 * mass_i * particle.velocity.xyz().norm_squared();
        total_momentum += particle.velocity.xyz() * mass_i;

        for (std::size_t j = i + 1; j < count; ++j) {
            const Vec3 displacement = particles_[j].position.xyz() - particle.position.xyz();
            const double distance = std::sqrt(displacement.norm_squared() + softening_sq);

            potential_energy -=
                config_.gravitational_constant *
                mass_i *
                particles_[j].mass() /
                distance;
        }
    }

    stats_.kinetic_energy = kinetic_energy;
    stats_.potential_energy = potential_energy;
    stats_.total_energy = kinetic_energy + potential_energy;
    stats_.momentum = total_momentum.norm();
}

const std::vector<Particle>& Simulation::particles() const noexcept {
    return particles_;
}

const SimulationConfig& Simulation::config() const noexcept {
    return config_;
}

const SimulationStats& Simulation::stats() const noexcept {
    return stats_;
}

BackendType Simulation::active_backend() const noexcept {
    return active_backend_;
}

std::string Simulation::active_backend_description() const {
    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        return std::string("OpenCL (") +
               (opencl_backend_->using_fp64() ? "FP64" : "FP32") +
               " | " + active_device_->device_name + ")";
    }
#ifdef ASTRA_HAS_OPENMP
    return "CPU (OpenMP)";
#else
    return "CPU (Single-thread)";
#endif
}

const std::optional<OpenCLDeviceInfo>& Simulation::active_opencl_device() const noexcept {
    return active_device_;
}

bool Simulation::is_using_fp64() const noexcept {
    if (active_backend_ == BackendType::OpenCL && opencl_backend_) {
        return opencl_backend_->using_fp64();
    }
    return true;
}

} // namespace astra