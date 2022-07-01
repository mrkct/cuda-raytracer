#include <math.h>
#include <util/rng.h>

__device__ double rng_next(curandState_t* state)
{
    // curand_uniform returns between the range (0.0, 1] but we
    // actually prefer the range [0.0, 1.0)
    return 1.0 - curand_uniform(state);
}

__device__ double rng_next_in_range(curandState_t* state, double min, double max)
{
    return min + rng_next(state) * (max - min);
}

__device__ vec3 rng_next_in_unit_sphere(curandState_t* state)
{
    // See: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    double u = rng_next(state);
    double v = rng_next(state);
    double theta = u * 2.0 * M_PI;
    double phi = cos(2.0 * v - 1.0);
    double r = cbrt(rng_next(state));
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double sin_phi = sin(phi);
    double cos_phi = cos(phi);
    double x = r * sin_phi * cos_theta;
    double y = r * sin_phi * sin_theta;
    double z = r * cos_phi;

    return make_vec3(x, y, z);
}
