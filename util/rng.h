#ifndef RNG_H
#define RNG_H

#include <curand_kernel.h>
#include <util/vec3.h>

__device__ double rng_next(curandState_t*);
__device__ double rng_next_in_range(curandState_t*, double min, double max);
__device__ vec3 rng_next_in_unit_sphere(curandState_t*);

#endif