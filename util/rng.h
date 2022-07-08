#ifndef RNG_H
#define RNG_H

#include <curand_kernel.h>
#include <util/vec3.h>
/**
 * @brief Returns a random value in the range (0.0, 1]
 *
 * @param state Initialized state of curand
 * @return A random value in the range
 */
__device__ inline float rng_next(curandState_t* state)
{
    return 1.0 - curand_uniform(state);
}

/**
 * @brief Returns a random value in the range (min, max]
 *
 * @param state Initialized state of curand
 * @param min Minimum value of the range (excluded)
 * @param max Maximum value of the range (included)
 * @return A random value in the specified range
 */
__device__ inline float rng_next_in_range(curandState_t* state, float min, float max)
{
    return min + rng_next(state) * (max - min);
}

/**
 * @brief Returns a random point in 3d space that is inside a sphere of unit radius.
 * All points returned by this function have guaranteed distance from the origin less
 * than 1
 *
 * @return A random point
 */
__device__ vec3 rng_next_in_unit_sphere(curandState_t*);

#endif