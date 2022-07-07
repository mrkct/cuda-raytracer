#include <math.h>
#include <util/rng.h>

__device__ vec3 rng_next_in_unit_sphere(curandState_t* state)
{
    // See: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    float const u = rng_next(state);
    float const v = rng_next(state);
    float const theta = u * 2.0 * M_PI;
    float const phi = cos(2.0 * v - 1.0);
    float const r = cbrt(rng_next(state));
    float const sin_theta = sin(theta);
    float const cos_theta = cos(theta);
    float const sin_phi = sin(phi);
    float const cos_phi = cos(phi);
    float const x = r * sin_phi * cos_theta;
    float const y = r * sin_phi * sin_theta;
    float const z = r * cos_phi;

    return make_vec3(x, y, z);
}
