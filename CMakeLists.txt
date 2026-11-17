
cmake_minimum_required(VERSION 3.20)

project(astra_nbody
    VERSION 0.1.0
    DESCRIPTION "3D gravitational N-body simulation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASTRA_ENABLE_OPENMP "Enable OpenMP CPU parallelism" ON)

add_executable(astra-nbody
    src/main.cpp
    src/simulation.cpp
    src/cpu_backend.cpp
)

target_include_directories(astra-nbody
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

if(ASTRA_ENABLE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(astra-nbody PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(astra-nbody PRIVATE ASTRA_HAS_OPENMP)
    endif()
endif()

if(MSVC)
    target_compile_options(astra-nbody PRIVATE /W4 /O2)
else()
    target_compile_options(astra-nbody PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

install(TARGETS astra-nbody RUNTIME DESTINATION bin)
