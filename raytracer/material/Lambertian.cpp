#include <raytracer/material/Lambertian.h>

__device__ bool Lambertian::scatter(
    DeviceRNG& rng,
    Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const
{
    auto scatter_direction = rec.normal + rng.next_in_unit_sphere();

    // Catch degenerate scatter direction
    if (scatter_direction.is_near_zero())
        scatter_direction = rec.normal;

    scattered = { rec.p, scatter_direction };
    attenuation = m_albedo;
    return true;
}