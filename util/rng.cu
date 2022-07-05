#include <math.h>
#include <util/rng.h>

__device__ float rng_next(curandState_t* state)
{
    // curand_uniform returns between the range (0.0, 1] but we
    // actually prefer the range [0.0, 1.0)
    return 1.0 - curand_uniform(state);
}

__device__ float rng_next_in_range(curandState_t* state, float min, float max)
{
    return min + rng_next(state) * (max - min);
}

__device__ vec3 rng_next_in_unit_sphere(curandState_t* state)
{
    // See: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    float u = rng_next(state);
    float v = rng_next(state);
    float theta = u * 2.0 * M_PI;
    float phi = cos(2.0 * v - 1.0);
    float r = cbrt(rng_next(state));
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float x = r * sin_phi * cos_theta;
    float y = r * sin_phi * sin_theta;
    float z = r * cos_phi;

    return make_vec3(x, y, z);
}
