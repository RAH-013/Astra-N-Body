#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "astra/opencl_device.hpp"
#include "astra/particle.hpp"
#include "astra/simulation.hpp"

namespace astra {

class OpenCLBackend {
public:
    OpenCLBackend(
        const OpenCLDeviceInfo& device_info,
        std::size_t particle_count,
        PrecisionMode requested_precision = PrecisionMode::Auto,
        const std::string& kernel_path = ""
    );

    ~OpenCLBackend();

    OpenCLBackend(const OpenCLBackend&) = delete;
    OpenCLBackend& operator=(const OpenCLBackend&) = delete;
    OpenCLBackend(OpenCLBackend&&) = delete;
    OpenCLBackend& operator=(OpenCLBackend&&) = delete;

    void upload_particles(const std::vector<Particle>& particles);
    void download_particles(std::vector<Particle>& particles) const;

    void calculate_accelerations(
        double gravitational_constant,
        double softening
    );

    void step(
        double gravitational_constant,
        double softening,
        double time_step
    );

    void run_steps(
        std::size_t steps,
        double gravitational_constant,
        double softening,
        double time_step
    );

    void synchronize() const;

    [[nodiscard]] const OpenCLDeviceInfo& device_info() const noexcept {
        return device_info_;
    }

    [[nodiscard]] bool using_fp64() const noexcept {
        return use_fp64_;
    }

    [[nodiscard]] std::size_t local_work_size() const noexcept {
        return local_work_size_;
    }

    [[nodiscard]] const std::string& resolved_kernel_path() const noexcept {
        return resolved_kernel_path_;
    }

private:
    static std::string resolve_kernel_file(const std::string& explicit_path);
    static std::string load_text_file(const std::string& path);

    OpenCLDeviceInfo device_info_;
    std::size_t particle_count_ = 0;
    bool use_fp64_ = false;
    std::size_t local_work_size_ = 64;
    std::size_t global_work_size_ = 64;
    std::string resolved_kernel_path_;

#ifdef ASTRA_HAS_OPENCL
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_forces_ = nullptr;
    cl_kernel kernel_positions_ = nullptr;
    cl_kernel kernel_velocities_ = nullptr;
    cl_mem buffer_particles_ = nullptr;
    cl_mem buffer_accelerations_ = nullptr;
#endif
};

} // namespace astra