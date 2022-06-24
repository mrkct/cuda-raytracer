#include <raytracer/material/Metal.h>

__device__ bool Metal::scatter(
    DeviceRNG& rng,
    Ray const& ray,
    HitRecord const& rec,
    Color& attenuation,
    Ray& scattered) const
{
    auto reflected = reflect_vector(unit_vector(ray.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + m_fuzz * rng.next_in_unit_sphere());
    attenuation = m_albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}