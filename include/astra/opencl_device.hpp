#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#ifdef ASTRA_HAS_OPENCL
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace astra {

enum class OpenCLDeviceCategory {
    GPU,
    CPU,
    Accelerator,
    Other
};

struct OpenCLDeviceInfo {
    std::size_t platform_index = 0;
    std::size_t device_index = 0;

    std::string platform_name;
    std::string platform_vendor;
    std::string platform_version;

    std::string device_name;
    std::string device_vendor;
    std::uint32_t vendor_id = 0;
    std::string device_version;
    std::string opencl_c_version;
    std::string driver_version;
    std::string extensions;

    OpenCLDeviceCategory category = OpenCLDeviceCategory::Other;
    bool is_amd = false;
    bool is_amd_radeon = false;
    bool supports_fp64 = false;
    std::uint64_t double_fp_config = 0;

    std::uint32_t compute_units = 0;
    std::uint32_t max_clock_mhz = 0;
    std::uint64_t global_mem_bytes = 0;
    std::uint64_t local_mem_bytes = 0;
    std::size_t max_work_group_size = 0;

    int selection_score = 0;

#ifdef ASTRA_HAS_OPENCL
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
#endif
};

class OpenCLDeviceManager {
public:
    static bool is_opencl_compiled() noexcept;

    static std::vector<OpenCLDeviceInfo> enumerate_devices();

    static std::optional<OpenCLDeviceInfo> select_best_device(
        std::optional<std::size_t> preferred_platform = std::nullopt,
        std::optional<std::size_t> preferred_device = std::nullopt,
        bool require_fp64 = false
    );

    static void print_device_list(
        const std::vector<OpenCLDeviceInfo>& devices,
        std::ostream& out
    );

    static void print_device_summary(
        const OpenCLDeviceInfo& device,
        std::ostream& out
    );
};

} // namespace astra