#ifndef RNG_H
#define RNG_H

#include <curand_kernel.h>
#include <util/vec3.h>

__device__ inline float rng_next(curandState_t* state)
{
    // curand_uniform returns between the range (0.0, 1] but we
    // actually prefer the range [0.0, 1.0)
    return 1.0 - curand_uniform(state);
}

__device__ inline float rng_next_in_range(curandState_t* state, float min, float max)
{
    return min + rng_next(state) * (max - min);
}

__device__ vec3 rng_next_in_unit_sphere(curandState_t*);

#endif