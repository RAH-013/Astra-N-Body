cmake_minimum_required(VERSION 3.20)

project(astra_nbody
    VERSION 0.2.2
    DESCRIPTION "3D gravitational N-body simulation with OpenMP and OpenCL"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASTRA_ENABLE_OPENMP "Enable OpenMP CPU parallelism" ON)
option(ASTRA_ENABLE_OPENCL "Enable OpenCL GPU acceleration" ON)
option(ASTRA_BUILD_TESTS "Build automated test suite" ON)

set(ASTRA_KERNEL_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/astra-nbody/kernels")

add_library(astra_core STATIC
    src/simulation.cpp
    src/cpu_backend.cpp
    src/opencl_device.cpp
    src/opencl_backend.cpp
    src/validator.cpp
)

target_include_directories(astra_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_definitions(astra_core
    PUBLIC
        ASTRA_VERSION_STRING="${PROJECT_VERSION}"
        ASTRA_DEFAULT_KERNEL_DIR="${ASTRA_KERNEL_INSTALL_DIR}"
        ASTRA_SOURCE_KERNEL_DIR="${PROJECT_SOURCE_DIR}/kernels"
)

if(ASTRA_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(astra_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(astra_core PUBLIC ASTRA_HAS_OPENMP)
    endif()
endif()

if(ASTRA_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        target_link_libraries(astra_core PUBLIC OpenCL::OpenCL)
        target_compile_definitions(astra_core PUBLIC ASTRA_HAS_OPENCL CL_TARGET_OPENCL_VERSION=120)
    else()
        message(WARNING "OpenCL development headers/library not found. Building with CPU fallback only.")
    endif()
endif()

if(MSVC)
    target_compile_options(astra_core PRIVATE /W4 /O2)
else()
    target_compile_options(astra_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

add_executable(astra-nbody
    src/main.cpp
)

target_link_libraries(astra-nbody PRIVATE astra_core)

if(MSVC)
    target_compile_options(astra-nbody PRIVATE /W4 /O2)
else()
    target_compile_options(astra-nbody PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

install(TARGETS astra-nbody RUNTIME DESTINATION bin)
install(FILES kernels/nbody.cl DESTINATION share/astra-nbody/kernels)
install(DIRECTORY include/astra DESTINATION include)

if(ASTRA_BUILD_TESTS)
    enable_testing()
    add_executable(astra_tests tests/test_nbody.cpp)
    target_link_libraries(astra_tests PRIVATE astra_core)
    if(MSVC)
        target_compile_options(astra_tests PRIVATE /W4 /O2)
    else()
        target_compile_options(astra_tests PRIVATE -Wall -Wextra -Wpedantic -O3)
    endif()
    add_test(NAME astra_unit_and_integration_tests COMMAND astra_tests)
    set_tests_properties(astra_unit_and_integration_tests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4")
endif()