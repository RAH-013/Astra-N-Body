#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "astra/opencl_device.hpp"
#include "astra/particle.hpp"

namespace astra {

enum class BackendType {
    Auto,
    CPU,
    OpenCL
};

enum class PrecisionMode {
    Auto,
    FP64,
    FP32
};

struct SimulationConfig {
    double gravitational_constant = 1.0;
    double softening = 0.01;
    double time_step = 0.001;
    std::size_t particle_count = 1024;
    std::uint64_t seed = 42;

    BackendType requested_backend = BackendType::Auto;
    PrecisionMode requested_precision = PrecisionMode::Auto;
    bool allow_cpu_fallback = true;
    bool include_central_body = true;

    std::optional<std::size_t> opencl_platform_index;
    std::optional<std::size_t> opencl_device_index;
    std::string kernel_path;
};

struct SimulationStats {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    double total_energy = 0.0;
    double momentum = 0.0;
    double elapsed_seconds = 0.0;
    double host_transfer_seconds = 0.0;
    std::uint64_t completed_steps = 0;
};

class OpenCLBackend;

class Simulation {
public:
    explicit Simulation(SimulationConfig config);
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void initialize_galaxy();
    void set_particles(std::vector<Particle> particles);

    void step();
    void run(std::size_t steps);

    [[nodiscard]] const std::vector<Particle>& particles() const noexcept;
    [[nodiscard]] const SimulationConfig& config() const noexcept;
    [[nodiscard]] const SimulationStats& stats() const noexcept;

    [[nodiscard]] BackendType active_backend() const noexcept;
    [[nodiscard]] std::string active_backend_description() const;
    [[nodiscard]] const std::optional<OpenCLDeviceInfo>& active_opencl_device() const noexcept;
    [[nodiscard]] bool is_using_fp64() const noexcept;

    void calculate_statistics();

private:
    void initialize_backend();
    void calculate_accelerations();

    SimulationConfig config_;
    std::vector<Particle> particles_;
    SimulationStats stats_;

    BackendType active_backend_ = BackendType::CPU;
    std::optional<OpenCLDeviceInfo> active_device_;
    std::unique_ptr<OpenCLBackend> opencl_backend_;
    bool device_state_dirty_ = false;
};

} // namespace astra