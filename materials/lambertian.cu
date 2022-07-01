#include <materials/lambertian.h>
#include <util/rng.h>

#include <stdio.h>

struct LambertianData make_lambertian_material_data(color albedo)
{
    return (struct LambertianData) { .albedo = albedo };
}

struct Material make_lambertian_material(LambertianData* data)
{
    return (struct Material) { .material_type = LAMBERTIAN, .data = data };
}

__device__ bool lambertian_scatter(void* vdata, curandState_t* rng_state, struct Ray ray, HitRecord* rec,
    color* out_attenuation, struct Ray* out_scattered)
{
    struct LambertianData* data = (struct LambertianData*)vdata;

    vec3 scatter_direction = rec->normal + rng_next_in_unit_sphere(rng_state);

    // Catch degenerate scatter direction
    if (is_near_zero(scatter_direction))
        scatter_direction = rec->normal;

    *out_scattered = make_ray(rec->p, scatter_direction);
    *out_attenuation = data->albedo;

    return true;
}