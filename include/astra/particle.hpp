
#pragma once

namespace astra {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3& other) const {
        return {
            x + other.x,
            y + other.y,
            z + other.z
        };
    }

    Vec3 operator-(const Vec3& other) const {
        return {
            x - other.x,
            y - other.y,
            z - other.z
        };
    }

    Vec3 operator*(double scalar) const {
        return {
            x * scalar,
            y * scalar,
            z * scalar
        };
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
};

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    double mass = 1.0;
};

}
