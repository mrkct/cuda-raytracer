#include <materials/metal.h>
#include <util/rng.h>

struct MetalData make_metal_material_data(color albedo, double fuzz)
{
    return (struct MetalData) { .albedo = albedo, .fuzz = fuzz };
}

struct Material make_metal_material(MetalData* data)
{
    return (struct Material) { .material_type = METAL, .data = data };
}

__device__ bool metal_scatter(void* vdata, curandState_t* rng_state, struct Ray ray, HitRecord* rec,
    color* out_attenuation, struct Ray* out_scattered)
{
    MetalData* data = (MetalData*)vdata;

    vec3 reflected = reflect_vector(unit_vector(ray.direction), rec->normal);
    *out_scattered = make_ray(rec->p, reflected + data->fuzz * rng_next_in_unit_sphere(rng_state));
    *out_attenuation = data->albedo;
    return (vec3_dot(out_scattered->direction, rec->normal) > 0);
}