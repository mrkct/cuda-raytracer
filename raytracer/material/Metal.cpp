#include <raytracer/material/Metal.h>

__device__ bool Metal::scatter(
    size_t id,
    Ray const& ray,
    HitRecord const& rec,
    Color& attenuation,
    Ray& scattered) const
{
    auto reflected = reflect_vector(unit_vector(ray.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + m_fuzz * m_rng.next_in_unit_sphere(id));
    attenuation = m_albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}