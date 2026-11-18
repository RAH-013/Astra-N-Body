#include "astra/opencl_device.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace astra {

namespace {

#ifdef ASTRA_HAS_OPENCL
std::string to_lower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

bool contains_icase(const std::string& haystack, const std::string& needle) {
    return to_lower(haystack).find(to_lower(needle)) != std::string::npos;
}

std::string get_platform_info_string(cl_platform_id platform, cl_platform_info param) {
    std::size_t size = 0;
    if (clGetPlatformInfo(platform, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string buffer(size, '\0');
    if (clGetPlatformInfo(platform, param, size, buffer.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    while (!buffer.empty() && buffer.back() == '\0') {
        buffer.pop_back();
    }
    return buffer;
}

std::string get_device_info_string(cl_device_id device, cl_device_info param) {
    std::size_t size = 0;
    if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string buffer(size, '\0');
    if (clGetDeviceInfo(device, param, size, buffer.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    while (!buffer.empty() && buffer.back() == '\0') {
        buffer.pop_back();
    }
    return buffer;
}

int compute_device_score(const OpenCLDeviceInfo& info) {
    int score = 0;

    if (info.category == OpenCLDeviceCategory::GPU) {
        score += 1000;
    } else if (info.category == OpenCLDeviceCategory::Accelerator) {
        score += 500;
    } else if (info.category == OpenCLDeviceCategory::CPU) {
        score += 100;
    }

    if (info.is_amd_radeon) {
        score += 400;
    } else if (info.is_amd) {
        score += 250;
    }

    if (info.supports_fp64) {
        score += 200;
    }

    score += static_cast<int>(std::min<std::uint32_t>(info.compute_units, 128));
    return score;
}
#endif

const char* category_label(OpenCLDeviceCategory category) {
    switch (category) {
        case OpenCLDeviceCategory::GPU:
            return "GPU";
        case OpenCLDeviceCategory::CPU:
            return "CPU";
        case OpenCLDeviceCategory::Accelerator:
            return "Accelerator";
        default:
            return "Other";
    }
}

} // namespace

bool OpenCLDeviceManager::is_opencl_compiled() noexcept {
#ifdef ASTRA_HAS_OPENCL
    return true;
#else
    return false;
#endif
}

std::vector<OpenCLDeviceInfo> OpenCLDeviceManager::enumerate_devices() {
    std::vector<OpenCLDeviceInfo> result;

#ifdef ASTRA_HAS_OPENCL
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        return result;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return result;
    }

    for (cl_uint p = 0; p < platform_count; ++p) {
        const cl_platform_id platform = platforms[p];
        const std::string plat_name = get_platform_info_string(platform, CL_PLATFORM_NAME);
        const std::string plat_vendor = get_platform_info_string(platform, CL_PLATFORM_VENDOR);
        const std::string plat_version = get_platform_info_string(platform, CL_PLATFORM_VERSION);

        cl_uint device_count = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (err != CL_SUCCESS || device_count == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        for (cl_uint d = 0; d < device_count; ++d) {
            const cl_device_id device = devices[d];

            OpenCLDeviceInfo info;
            info.platform_index = p;
            info.device_index = d;
            info.platform_id = platform;
            info.device_id = device;
            info.platform_name = plat_name;
            info.platform_vendor = plat_vendor;
            info.platform_version = plat_version;

            info.device_name = get_device_info_string(device, CL_DEVICE_NAME);
            info.device_vendor = get_device_info_string(device, CL_DEVICE_VENDOR);
            info.device_version = get_device_info_string(device, CL_DEVICE_VERSION);
            info.opencl_c_version = get_device_info_string(device, CL_DEVICE_OPENCL_C_VERSION);
            info.driver_version = get_device_info_string(device, CL_DRIVER_VERSION);
            info.extensions = get_device_info_string(device, CL_DEVICE_EXTENSIONS);

            cl_uint vendor_id = 0;
            clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, nullptr);
            info.vendor_id = vendor_id;

            cl_device_type dev_type = 0;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr);
            if (dev_type & CL_DEVICE_TYPE_GPU) {
                info.category = OpenCLDeviceCategory::GPU;
            } else if (dev_type & CL_DEVICE_TYPE_CPU) {
                info.category = OpenCLDeviceCategory::CPU;
            } else if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) {
                info.category = OpenCLDeviceCategory::Accelerator;
            } else {
                info.category = OpenCLDeviceCategory::Other;
            }

            cl_uint compute_units = 0;
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
            info.compute_units = compute_units;

            cl_uint clock_mhz = 0;
            clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_mhz), &clock_mhz, nullptr);
            info.max_clock_mhz = clock_mhz;

            cl_ulong global_mem = 0;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
            info.global_mem_bytes = global_mem;

            cl_ulong local_mem = 0;
            clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, nullptr);
            info.local_mem_bytes = local_mem;

            std::size_t max_wg = 0;
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
            info.max_work_group_size = max_wg;

            cl_device_fp_config double_cfg = 0;
            const cl_int fp64_err = clGetDeviceInfo(
                device,
                CL_DEVICE_DOUBLE_FP_CONFIG,
                sizeof(double_cfg),
                &double_cfg,
                nullptr
            );
            info.double_fp_config = (fp64_err == CL_SUCCESS) ? static_cast<std::uint64_t>(double_cfg) : 0ULL;

            const bool has_fp64_ext = info.extensions.find("cl_khr_fp64") != std::string::npos;
            info.supports_fp64 = has_fp64_ext && (info.double_fp_config != 0);

            info.is_amd =
                (info.vendor_id == 0x1002) ||
                contains_icase(info.device_vendor, "amd") ||
                contains_icase(info.device_vendor, "advanced micro devices") ||
                contains_icase(info.platform_vendor, "amd");

            info.is_amd_radeon =
                info.is_amd &&
                (info.category == OpenCLDeviceCategory::GPU) &&
                (contains_icase(info.device_name, "radeon") ||
                 contains_icase(info.device_name, "vega") ||
                 contains_icase(info.device_name, "gfx"));

            info.selection_score = compute_device_score(info);
            result.push_back(std::move(info));
        }
    }
#endif

    return result;
}

std::optional<OpenCLDeviceInfo> OpenCLDeviceManager::select_best_device(
    std::optional<std::size_t> preferred_platform,
    std::optional<std::size_t> preferred_device,
    bool require_fp64,
    bool require_gpu
) {
    const auto devices = enumerate_devices();
    if (devices.empty()) {
        return std::nullopt;
    }

    if (preferred_platform.has_value() || preferred_device.has_value()) {
        const std::size_t target_p = preferred_platform.value_or(0);
        const std::size_t target_d = preferred_device.value_or(0);
        for (const auto& dev : devices) {
            if (dev.platform_index == target_p && dev.device_index == target_d) {
                if ((require_fp64 && !dev.supports_fp64) ||
                    (require_gpu && dev.category != OpenCLDeviceCategory::GPU)) {
                    return std::nullopt;
                }
                return dev;
            }
        }
        return std::nullopt;
    }

    const OpenCLDeviceInfo* best = nullptr;
    for (const auto& dev : devices) {
        if ((require_fp64 && !dev.supports_fp64) ||
            (require_gpu && dev.category != OpenCLDeviceCategory::GPU)) {
            continue;
        }
        if (best == nullptr || dev.selection_score > best->selection_score) {
            best = &dev;
        }
    }

    if (best == nullptr) {
        return std::nullopt;
    }
    return *best;
}

void OpenCLDeviceManager::print_device_list(
    const std::vector<OpenCLDeviceInfo>& devices,
    std::ostream& out
) {
    if (!is_opencl_compiled()) {
        out << "OpenCL support was not compiled into this binary.\n";
        return;
    }
    if (devices.empty()) {
        out << "No OpenCL platforms or devices detected.\n";
        return;
    }

    out << "Detected OpenCL Devices (" << devices.size() << "):\n";
    out << "------------------------------------------------------------\n";
    for (const auto& dev : devices) {
        out << "[" << dev.platform_index << ":" << dev.device_index << "] "
            << dev.device_name
            << " (" << category_label(dev.category) << ")"
            << (dev.is_amd_radeon ? " [AMD Radeon]" : "")
            << "\n";
        out << "      Platform : " << dev.platform_name << " (" << dev.platform_version << ")\n";
        out << "      Vendor   : " << dev.device_vendor << " (0x" << std::hex << dev.vendor_id << std::dec << ")\n";
        out << "      Hardware : " << dev.compute_units << " CUs @ " << dev.max_clock_mhz << " MHz, "
            << std::fixed << std::setprecision(1)
            << (static_cast<double>(dev.global_mem_bytes) / (1024.0 * 1024.0)) << " MiB Global, "
            << (static_cast<double>(dev.local_mem_bytes) / 1024.0) << " KiB Local\n";
        out << "      FP64     : " << (dev.supports_fp64 ? "Supported (cl_khr_fp64)" : "Not supported (FP32 only)")
            << "\n";
    }
}

void OpenCLDeviceManager::print_device_summary(
    const OpenCLDeviceInfo& device,
    std::ostream& out
) {
    out << "OpenCL Device: [" << device.platform_index << ":" << device.device_index << "] "
        << device.device_name
        << " (" << category_label(device.category)
        << (device.is_amd_radeon ? ", AMD Radeon" : "") << ")\n";
    out << "OpenCL Platform: " << device.platform_name << " | Driver: " << device.driver_version << "\n";
    out << "Device Specs: " << device.compute_units << " CUs @ " << device.max_clock_mhz << " MHz | "
        << std::fixed << std::setprecision(1)
        << (static_cast<double>(device.global_mem_bytes) / (1024.0 * 1024.0)) << " MiB VRAM | "
        << "Max WG: " << device.max_work_group_size << "\n";
    out << "FP64 Capability: "
        << (device.supports_fp64 ? "YES (cl_khr_fp64)" : "NO (using FP32 fallback kernel)")
        << "\n";
}

} // namespace astra