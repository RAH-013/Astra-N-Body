
#ifdef ASTRA_USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
typedef double4 vec4_t;
#define ZERO_REAL 0.0
#define HALF_REAL 0.5
#else
typedef float real_t;
typedef float4 vec4_t;
#define ZERO_REAL 0.0f
#define HALF_REAL 0.5f
#endif

#define TILE_SIZE 64

typedef struct {
    vec4_t position;
    vec4_t velocity;
    vec4_t acceleration;
} Particle;

__kernel void calculate_forces(
    __global const Particle* particles,
    __global vec4_t* accelerations,
    const uint particle_count,
    const real_t gravitational_constant,
    const real_t softening
) {
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);

    __local vec4_t tile_pos[TILE_SIZE];

    const vec4_t position_i =
        i < particle_count ? particles[i].position : (vec4_t)(ZERO_REAL);

    real_t ax = ZERO_REAL;
    real_t ay = ZERO_REAL;
    real_t az = ZERO_REAL;

    const real_t softening_squared = softening * softening;

    for (uint tile_start = 0; tile_start < particle_count; tile_start += TILE_SIZE) {
        const uint tile_idx = tile_start + lid;

        tile_pos[lid] = tile_idx < particle_count
            ? particles[tile_idx].position
            : (vec4_t)(ZERO_REAL);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < particle_count) {
            const uint tile_end = min((uint)TILE_SIZE, particle_count - tile_start);

            for (uint k = 0; k < tile_end; ++k) {
                const uint j = tile_start + k;

                if (i == j) {
                    continue;
                }

                const vec4_t pos_j = tile_pos[k];

                const real_t dx = pos_j.x - position_i.x;
                const real_t dy = pos_j.y - position_i.y;
                const real_t dz = pos_j.z - position_i.z;

                const real_t distance_squared =
                    dx * dx + dy * dy + dz * dz + softening_squared;

                const real_t inverse_distance =
                    (real_t)1.0 / sqrt(distance_squared);

                const real_t inverse_distance_cubed =
                    inverse_distance * inverse_distance * inverse_distance;

                const real_t factor =
                    gravitational_constant * pos_j.w * inverse_distance_cubed;

                ax += dx * factor;
                ay += dy * factor;
                az += dz * factor;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < particle_count) {
        accelerations[i] = (vec4_t)(ax, ay, az, ZERO_REAL);
    }
}

__kernel void integrate_positions(
    __global Particle* particles,
    const uint particle_count,
    const real_t time_step
) {
    const uint i = get_global_id(0);

    if (i >= particle_count) {
        return;
    }

    const real_t half_step = HALF_REAL * time_step;

    Particle particle = particles[i];

    particle.velocity.x += particle.acceleration.x * half_step;
    particle.velocity.y += particle.acceleration.y * half_step;
    particle.velocity.z += particle.acceleration.z * half_step;

    particle.position.x += particle.velocity.x * time_step;
    particle.position.y += particle.velocity.y * time_step;
    particle.position.z += particle.velocity.z * time_step;

    particles[i] = particle;
}

__kernel void integrate_velocities(
    __global Particle* particles,
    __global const vec4_t* accelerations,
    const uint particle_count,
    const real_t time_step
) {
    const uint i = get_global_id(0);

    if (i >= particle_count) {
        return;
    }

    const real_t half_step = HALF_REAL * time_step;

    Particle particle = particles[i];
    const vec4_t acceleration = accelerations[i];

    particle.velocity.x += acceleration.x * half_step;
    particle.velocity.y += acceleration.y * half_step;
    particle.velocity.z += acceleration.z * half_step;

    particle.acceleration = (vec4_t)(
        acceleration.x,
        acceleration.y,
        acceleration.z,
        ZERO_REAL
    );

    particles[i] = particle;
}
