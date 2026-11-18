#include "astra/opencl_backend.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace astra {

namespace {

struct alignas(16) Float4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct alignas(16) ParticleF32 {
    Float4 position{};
    Float4 velocity{};
    Float4 acceleration{};
};

static_assert(sizeof(ParticleF32) == 48, "ParticleF32 must match OpenCL float4 Particle (48 bytes)");

#ifdef ASTRA_HAS_OPENCL
void check_cl(cl_int err, const char* context_msg) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << "OpenCL error (" << err << ") during " << context_msg;
        throw std::runtime_error(oss.str());
    }
}
#endif

} // namespace

std::string OpenCLBackend::resolve_kernel_file(const std::string& explicit_path) {
    namespace fs = std::filesystem;

    std::vector<fs::path> candidates;
    if (!explicit_path.empty()) {
        candidates.emplace_back(explicit_path);
    }
    candidates.emplace_back("kernels/nbody.cl");
    candidates.emplace_back("../kernels/nbody.cl");

#ifdef ASTRA_DEFAULT_KERNEL_DIR
    candidates.emplace_back(fs::path(ASTRA_DEFAULT_KERNEL_DIR) / "nbody.cl");
#endif
#ifdef ASTRA_SOURCE_KERNEL_DIR
    candidates.emplace_back(fs::path(ASTRA_SOURCE_KERNEL_DIR) / "nbody.cl");
#endif

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return candidate.string();
        }
    }

    throw std::runtime_error(
        "Unable to locate OpenCL kernel file 'kernels/nbody.cl'. "
        "Specify path via --kernel."
    );
}

std::string OpenCLBackend::load_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OpenCL kernel file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

OpenCLBackend::OpenCLBackend(
    const OpenCLDeviceInfo& device_info,
    std::size_t particle_count,
    PrecisionMode requested_precision,
    const std::string& kernel_path
)
    : device_info_(device_info),
      particle_count_(particle_count) {
#ifndef ASTRA_HAS_OPENCL
    (void)requested_precision;
    (void)kernel_path;
    throw std::runtime_error("ASTRA N-BODY was built without OpenCL support.");
#else
    if (particle_count_ == 0 || particle_count_ > static_cast<std::size_t>(UINT32_MAX)) {
        throw std::invalid_argument("OpenCLBackend particle count must fit in an unsigned 32-bit integer");
    }

    if (requested_precision == PrecisionMode::FP64) {
        if (!device_info_.supports_fp64) {
            throw std::runtime_error(
                "Selected OpenCL device '" + device_info_.device_name +
                "' does not support cl_khr_fp64 (FP64 required)."
            );
        }
        use_fp64_ = true;
    } else if (requested_precision == PrecisionMode::FP32) {
        use_fp64_ = false;
    } else {
        use_fp64_ = device_info_.supports_fp64;
    }

    // Tile size in kernels/nbody.cl is fixed at 64 (AMD Vega wavefront width).
    // Ensure device supports at least 64 work-items per work-group.
    if (device_info_.max_work_group_size > 0 && device_info_.max_work_group_size < 64) {
        throw std::runtime_error(
            "OpenCL device max work-group size (<64) is below tiled kernel requirement."
        );
    }
    local_work_size_ = 64;
    global_work_size_ =
        ((particle_count_ + local_work_size_ - 1) / local_work_size_) * local_work_size_;

    resolved_kernel_path_ = resolve_kernel_file(kernel_path);
    const std::string source_code = load_text_file(resolved_kernel_path_);

    cl_int err = CL_SUCCESS;
    const cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(device_info_.platform_id),
        0
    };

    context_ = clCreateContext(props, 1, &device_info_.device_id, nullptr, nullptr, &err);
    check_cl(err, "clCreateContext");

    try {
        queue_ = clCreateCommandQueue(context_, device_info_.device_id, 0, &err);
        check_cl(err, "clCreateCommandQueue");

        const char* src_ptr = source_code.c_str();
        const std::size_t src_len = source_code.size();
        program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
        check_cl(err, "clCreateProgramWithSource");

        std::string build_options = "-cl-opt-disable";  
        if (use_fp64_) {
            build_options += " -DASTRA_USE_FP64=1";
        }

        err = clBuildProgram(
            program_,
            1,
            &device_info_.device_id,
            build_options.c_str(),
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(
                program_,
                device_info_.device_id,
                CL_PROGRAM_BUILD_LOG,
                0,
                nullptr,
                &log_size
            );
            std::string build_log(log_size, '\0');
            if (log_size > 0) {
                clGetProgramBuildInfo(
                    program_,
                    device_info_.device_id,
                    CL_PROGRAM_BUILD_LOG,
                    log_size,
                    build_log.data(),
                    nullptr
                );
            }
            throw std::runtime_error("OpenCL kernel build failed:\n" + build_log);
        }

        kernel_forces_ = clCreateKernel(program_, "calculate_forces", &err);
        check_cl(err, "clCreateKernel(calculate_forces)");

        kernel_positions_ = clCreateKernel(program_, "integrate_positions", &err);
        check_cl(err, "clCreateKernel(integrate_positions)");

        kernel_velocities_ = clCreateKernel(program_, "integrate_velocities", &err);
        check_cl(err, "clCreateKernel(integrate_velocities)");

        const std::size_t particle_bytes = particle_count_ * (use_fp64_ ? sizeof(Particle) : sizeof(ParticleF32));
        const std::size_t accel_bytes = particle_count_ * (use_fp64_ ? sizeof(Vec4) : sizeof(Float4));

        if (device_info_.global_mem_bytes > 0 &&
            (particle_bytes + accel_bytes) > device_info_.global_mem_bytes) {
            throw std::runtime_error("Requested particle buffers exceed OpenCL device global memory.");
        }

        buffer_particles_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, particle_bytes, nullptr, &err);
        check_cl(err, "clCreateBuffer(particles)");

        buffer_accelerations_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, accel_bytes, nullptr, &err);
        check_cl(err, "clCreateBuffer(accelerations)");
    } catch (...) {
        if (buffer_accelerations_) clReleaseMemObject(buffer_accelerations_);
        if (buffer_particles_) clReleaseMemObject(buffer_particles_);
        if (kernel_velocities_) clReleaseKernel(kernel_velocities_);
        if (kernel_positions_) clReleaseKernel(kernel_positions_);
        if (kernel_forces_) clReleaseKernel(kernel_forces_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
        throw;
    }
#endif
}

OpenCLBackend::~OpenCLBackend() {
#ifdef ASTRA_HAS_OPENCL
    if (queue_) {
        clFinish(queue_);
    }
    if (buffer_accelerations_) clReleaseMemObject(buffer_accelerations_);
    if (buffer_particles_) clReleaseMemObject(buffer_particles_);
    if (kernel_velocities_) clReleaseKernel(kernel_velocities_);
    if (kernel_positions_) clReleaseKernel(kernel_positions_);
    if (kernel_forces_) clReleaseKernel(kernel_forces_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
#endif
}

void OpenCLBackend::upload_particles(const std::vector<Particle>& particles) {
#ifndef ASTRA_HAS_OPENCL
    (void)particles;
#else
    if (particles.size() != particle_count_) {
        throw std::invalid_argument("Particle vector size mismatch during OpenCL upload");
    }

    if (use_fp64_) {
        const cl_int err = clEnqueueWriteBuffer(
            queue_,
            buffer_particles_,
            CL_TRUE,
            0,
            particles.size() * sizeof(Particle),
            particles.data(),
            0,
            nullptr,
            nullptr
        );
        check_cl(err, "clEnqueueWriteBuffer(FP64)");
    } else {
        std::vector<ParticleF32> converted(particles.size());
        for (std::size_t i = 0; i < particles.size(); ++i) {
            converted[i].position = {
                static_cast<float>(particles[i].position.x),
                static_cast<float>(particles[i].position.y),
                static_cast<float>(particles[i].position.z),
                static_cast<float>(particles[i].position.w)
            };
            converted[i].velocity = {
                static_cast<float>(particles[i].velocity.x),
                static_cast<float>(particles[i].velocity.y),
                static_cast<float>(particles[i].velocity.z),
                0.0f
            };
            converted[i].acceleration = {
                static_cast<float>(particles[i].acceleration.x),
                static_cast<float>(particles[i].acceleration.y),
                static_cast<float>(particles[i].acceleration.z),
                0.0f
            };
        }

        const cl_int err = clEnqueueWriteBuffer(
            queue_,
            buffer_particles_,
            CL_TRUE,
            0,
            converted.size() * sizeof(ParticleF32),
            converted.data(),
            0,
            nullptr,
            nullptr
        );
        check_cl(err, "clEnqueueWriteBuffer(FP32)");
    }
#endif
}

void OpenCLBackend::download_particles(std::vector<Particle>& particles) const {
#ifndef ASTRA_HAS_OPENCL
    (void)particles;
#else
    if (particles.size() != particle_count_) {
        throw std::invalid_argument("Particle vector size mismatch during OpenCL download");
    }

    if (use_fp64_) {
        const cl_int err = clEnqueueReadBuffer(
            queue_,
            buffer_particles_,
            CL_TRUE,
            0,
            particles.size() * sizeof(Particle),
            particles.data(),
            0,
            nullptr,
            nullptr
        );
        check_cl(err, "clEnqueueReadBuffer(FP64)");
    } else {
        std::vector<ParticleF32> converted(particles.size());
        const cl_int err = clEnqueueReadBuffer(
            queue_,
            buffer_particles_,
            CL_TRUE,
            0,
            converted.size() * sizeof(ParticleF32),
            converted.data(),
            0,
            nullptr,
            nullptr
        );
        check_cl(err, "clEnqueueReadBuffer(FP32)");

        for (std::size_t i = 0; i < particles.size(); ++i) {
            particles[i].position = {
                static_cast<double>(converted[i].position.x),
                static_cast<double>(converted[i].position.y),
                static_cast<double>(converted[i].position.z),
                static_cast<double>(converted[i].position.w)
            };
            particles[i].velocity = {
                static_cast<double>(converted[i].velocity.x),
                static_cast<double>(converted[i].velocity.y),
                static_cast<double>(converted[i].velocity.z),
                0.0
            };
            particles[i].acceleration = {
                static_cast<double>(converted[i].acceleration.x),
                static_cast<double>(converted[i].acceleration.y),
                static_cast<double>(converted[i].acceleration.z),
                0.0
            };
        }
    }
#endif
}

void OpenCLBackend::calculate_accelerations(
    double gravitational_constant,
    double softening
) {
#ifndef ASTRA_HAS_OPENCL
    (void)gravitational_constant;
    (void)softening;
#else
    const cl_uint count_u32 = static_cast<cl_uint>(particle_count_);

    check_cl(clSetKernelArg(kernel_forces_, 0, sizeof(cl_mem), &buffer_particles_), "setArg(forces,0)");
    check_cl(clSetKernelArg(kernel_forces_, 1, sizeof(cl_mem), &buffer_accelerations_), "setArg(forces,1)");
    check_cl(clSetKernelArg(kernel_forces_, 2, sizeof(cl_uint), &count_u32), "setArg(forces,2)");

    if (use_fp64_) {
        const double g_val = gravitational_constant;
        const double eps_val = softening;
        check_cl(clSetKernelArg(kernel_forces_, 3, sizeof(double), &g_val), "setArg(forces,3)");
        check_cl(clSetKernelArg(kernel_forces_, 4, sizeof(double), &eps_val), "setArg(forces,4)");
    } else {
        const float g_val = static_cast<float>(gravitational_constant);
        const float eps_val = static_cast<float>(softening);
        check_cl(clSetKernelArg(kernel_forces_, 3, sizeof(float), &g_val), "setArg(forces,3)");
        check_cl(clSetKernelArg(kernel_forces_, 4, sizeof(float), &eps_val), "setArg(forces,4)");
    }

    const cl_int err = clEnqueueNDRangeKernel(
        queue_,
        kernel_forces_,
        1,
        nullptr,
        &global_work_size_,
        &local_work_size_,
        0,
        nullptr,
        nullptr
    );
    check_cl(err, "clEnqueueNDRangeKernel(calculate_forces)");
#endif
}

void OpenCLBackend::step(
    double gravitational_constant,
    double softening,
    double time_step
) {
#ifndef ASTRA_HAS_OPENCL
    (void)gravitational_constant;
    (void)softening;
    (void)time_step;
#else
    const cl_uint count_u32 = static_cast<cl_uint>(particle_count_);

    // 1. Half-kick velocity & drift position
    check_cl(clSetKernelArg(kernel_positions_, 0, sizeof(cl_mem), &buffer_particles_), "setArg(pos,0)");
    check_cl(clSetKernelArg(kernel_positions_, 1, sizeof(cl_uint), &count_u32), "setArg(pos,1)");
    if (use_fp64_) {
        const double dt_val = time_step;
        check_cl(clSetKernelArg(kernel_positions_, 2, sizeof(double), &dt_val), "setArg(pos,2)");
    } else {
        const float dt_val = static_cast<float>(time_step);
        check_cl(clSetKernelArg(kernel_positions_, 2, sizeof(float), &dt_val), "setArg(pos,2)");
    }

    check_cl(
        clEnqueueNDRangeKernel(
            queue_,
            kernel_positions_,
            1,
            nullptr,
            &global_work_size_,
            &local_work_size_,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueNDRangeKernel(integrate_positions)"
    );

    // 2. Compute new gravitational accelerations
    calculate_accelerations(gravitational_constant, softening);

    // 3. Second half-kick velocity & update stored acceleration
    check_cl(clSetKernelArg(kernel_velocities_, 0, sizeof(cl_mem), &buffer_particles_), "setArg(vel,0)");
    check_cl(clSetKernelArg(kernel_velocities_, 1, sizeof(cl_mem), &buffer_accelerations_), "setArg(vel,1)");
    check_cl(clSetKernelArg(kernel_velocities_, 2, sizeof(cl_uint), &count_u32), "setArg(vel,2)");
    if (use_fp64_) {
        const double dt_val = time_step;
        check_cl(clSetKernelArg(kernel_velocities_, 3, sizeof(double), &dt_val), "setArg(vel,3)");
    } else {
        const float dt_val = static_cast<float>(time_step);
        check_cl(clSetKernelArg(kernel_velocities_, 3, sizeof(float), &dt_val), "setArg(vel,3)");
    }

    check_cl(
        clEnqueueNDRangeKernel(
            queue_,
            kernel_velocities_,
            1,
            nullptr,
            &global_work_size_,
            &local_work_size_,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueNDRangeKernel(integrate_velocities)"
    );
#endif
}

void OpenCLBackend::run_steps(
    std::size_t steps,
    double gravitational_constant,
    double softening,
    double time_step
) {
    for (std::size_t s = 0; s < steps; ++s) {
        step(gravitational_constant, softening, time_step);
    }
    synchronize();
}

void OpenCLBackend::synchronize() const {
#ifdef ASTRA_HAS_OPENCL
    if (queue_) {
        check_cl(clFinish(queue_), "clFinish");
    }
#endif
}

} // namespace astra