
typedef struct {
    double4 position;
    double4 velocity;
    double4 acceleration;
} Particle;

__kernel void calculate_forces(
    __global const Particle* particles,
    __global double4* accelerations,
    const uint particle_count,
    const double gravitational_constant,
    const double softening
) {
    const uint i = get_global_id(0);

    if (i >= particle_count) {
        return;
    }

    const double4 position_i = particles[i].position;
    double4 acceleration = (double4)(0.0);

    const double softening_squared = softening * softening;

    for (uint j = 0; j < particle_count; ++j) {
        if (i == j) {
            continue;
        }

        const double4 displacement =
            particles[j].position - position_i;

        const double distance_squared =
            dot(displacement.xyz, displacement.xyz) +
            softening_squared;

        const double inverse_distance =
            rsqrt(distance_squared);

        const double inverse_distance_cubed =
            inverse_distance * inverse_distance * inverse_distance;

        const double factor =
            gravitational_constant *
            particles[j].position.w *
            inverse_distance_cubed;

        acceleration.xyz += displacement.xyz * factor;
    }

    accelerations[i] = acceleration;
}

__kernel void integrate_positions(
    __global Particle* particles,
    const uint particle_count,
    const double time_step
) {
    const uint i = get_global_id(0);

    if (i >= particle_count) {
        return;
    }

    const double half_step = 0.5 * time_step;

    Particle particle = particles[i];

    particle.velocity.xyz +=
        particle.acceleration.xyz * half_step;

    particle.position.xyz +=
        particle.velocity.xyz * time_step;

    particles[i] = particle;
}

__kernel void integrate_velocities(
    __global Particle* particles,
    __global const double4* accelerations,
    const uint particle_count,
    const double time_step
) {
    const uint i = get_global_id(0);

    if (i >= particle_count) {
        return;
    }

    Particle particle = particles[i];

    particle.velocity.xyz +=
        accelerations[i].xyz * (0.5 * time_step);

    particle.acceleration = accelerations[i];

    particles[i] = particle;
}
