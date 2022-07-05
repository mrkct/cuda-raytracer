#include <materials/dielectric.h>
#include <util/more_math.h>
#include <util/rng.h>

struct DielectricData make_dielectric_material_data(float refraction_level)
{
    return (struct DielectricData) { .refraction_level = refraction_level };
}

struct Material make_dielectric_material(DielectricData* data)
{
    return (struct Material) { .material_type = DIELECTRIC, .data = data };
}

static __device__ float dielectric_reflectance(float cosine, float ref_idx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

static __device__ vec3 refract_vector(vec3 uv, vec3 n, float etai_over_etat)
{
    float cos_theta = my_fmin(vec3_dot(vec3_negate(uv), n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - vec3_length_squared(r_out_perp))) * n;

    return r_out_perp + r_out_parallel;
}

__device__ bool dielectric_scatter(void* vdata, curandState_t* rng_state, struct Ray ray, HitRecord* rec,
    color* out_attenuation, struct Ray* out_scattered)
{
    struct DielectricData* data = (struct DielectricData*)vdata;

    *out_attenuation = make_color(1.0, 1.0, 1.0);
    float refraction_ratio = rec->front_face ? (1.0 / data->refraction_level) : data->refraction_level;

    vec3 unit_direction = unit_vector(ray.direction);

    float cos_theta = my_fmin(vec3_dot(vec3_negate(unit_direction), rec->normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || dielectric_reflectance(cos_theta, refraction_ratio) > rng_next(rng_state))
        direction = reflect_vector(unit_direction, rec->normal);
    else
        direction = refract_vector(unit_direction, rec->normal, refraction_ratio);

    *out_scattered = make_ray(rec->p, direction);

    return true;
}