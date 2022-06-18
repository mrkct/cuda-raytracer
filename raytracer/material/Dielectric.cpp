#include <raytracer/material/Dielectric.h>

__device__ Vec3 Dielectric::refract_vector(Vec3 const& uv, Vec3 const& n, double etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ bool Dielectric::scatter(
    size_t id,
    Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const
{
    attenuation = Color(1.0, 1.0, 1.0);
    float refraction_ratio = rec.front_face ? (1.0 / m_refraction_level) : m_refraction_level;

    Vec3 unit_direction = unit_vector(r_in.direction());

    float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3 direction;

    if (cannot_refract)
        direction = reflect_vector(unit_direction, rec.normal);
    else
        direction = refract_vector(unit_direction, rec.normal, refraction_ratio);

    scattered = Ray(rec.p, direction);

    return true;
}
