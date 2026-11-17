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
    vec4_t position;     // xyz = position, w = mass
    vec4_t velocity;     // xyz = velocity, w = 0
    vec4_t acceleration; // xyz = acceleration, w = 0
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

    const vec4_t position_i = (i < particle_count) ? particles[i].position : (vec4_t)(ZERO_REAL);
    vec4_t acceleration = (vec4_t)(ZERO_REAL);
    const real_t softening_squared = softening * softening;

    for (uint tile_start = 0; tile_start < particle_count; tile_start += TILE_SIZE) {
        const uint tile_idx = tile_start + lid;
        tile_pos[lid] = (tile_idx < particle_count)
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
                const vec4_t displacement = pos_j - position_i;

                const real_t distance_squared =
                    dot(displacement.xyz, displacement.xyz) + softening_squared;

                const real_t inverse_distance = rsqrt(distance_squared);
                const real_t inverse_distance_cubed =
                    inverse_distance * inverse_distance * inverse_distance;

                const real_t factor =
                    gravitational_constant * pos_j.w * inverse_distance_cubed;

                acceleration.xyz += displacement.xyz * factor;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < particle_count) {
        acceleration.w = ZERO_REAL;
        accelerations[i] = acceleration;
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

    particle.velocity.xyz += particle.acceleration.xyz * half_step;
    particle.position.xyz += particle.velocity.xyz * time_step;

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
    const vec4_t accel = accelerations[i];

    particle.velocity.xyz += accel.xyz * half_step;
    particle.acceleration = (vec4_t)(accel.xyz, ZERO_REAL);

    particles[i] = particle;
}