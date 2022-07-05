#ifndef RNG_H
#define RNG_H

#include <curand_kernel.h>
#include <util/vec3.h>

__device__ float rng_next(curandState_t*);
__device__ float rng_next_in_range(curandState_t*, float min, float max);
__device__ vec3 rng_next_in_unit_sphere(curandState_t*);

#endif