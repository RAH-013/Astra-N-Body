#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace astra {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    constexpr Vec3 operator+(const Vec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    constexpr Vec3 operator-(const Vec3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    constexpr Vec3 operator*(double scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }

    constexpr Vec3& operator+=(const Vec3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    [[nodiscard]] double norm_squared() const noexcept {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(norm_squared());
    }
};

// ABI-compatible with OpenCL double4 (32 bytes, 32-byte aligned)
struct alignas(32) Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;

    [[nodiscard]] constexpr Vec3 xyz() const noexcept {
        return {x, y, z};
    }

    constexpr void set_xyz(const Vec3& v) noexcept {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    constexpr void add_xyz(const Vec3& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
    }
};

// ABI-compatible with OpenCL Particle struct:
//   position.xyz = spatial coordinates, position.w = mass
//   velocity.xyz = velocity components, velocity.w = reserved (0.0)
//   acceleration.xyz = acceleration components, acceleration.w = reserved (0.0)
struct alignas(32) Particle {
    Vec4 position{};
    Vec4 velocity{};
    Vec4 acceleration{};

    [[nodiscard]] constexpr double mass() const noexcept {
        return position.w;
    }

    constexpr void set_mass(double m) noexcept {
        position.w = m;
    }
};

static_assert(sizeof(Vec4) == 32, "Vec4 must match OpenCL double4 size (32 bytes)");
static_assert(alignof(Vec4) == 32, "Vec4 must match OpenCL double4 alignment (32 bytes)");
static_assert(sizeof(Particle) == 96, "Particle must match OpenCL Particle size (96 bytes)");
static_assert(alignof(Particle) == 32, "Particle must match OpenCL Particle alignment (32 bytes)");
static_assert(offsetof(Particle, position) == 0, "Particle::position offset mismatch");
static_assert(offsetof(Particle, velocity) == 32, "Particle::velocity offset mismatch");
static_assert(offsetof(Particle, acceleration) == 64, "Particle::acceleration offset mismatch");
static_assert(std::is_standard_layout_v<Particle>, "Particle must be standard-layout for OpenCL buffer transfer");

} 