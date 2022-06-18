#include <raytracer/material/Lambertian.h>

__device__ bool Lambertian::scatter(
    size_t id,
    Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const
{
    auto scatter_direction = rec.normal + m_rng.next_in_unit_sphere(id);

    // Catch degenerate scatter direction
    if (scatter_direction.is_near_zero())
        scatter_direction = rec.normal;
    __syncthreads();

    scattered = { rec.p, scatter_direction };
    attenuation = m_albedo;
    return true;
}