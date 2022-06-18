#include <raytracer/material/Metal.h>

__device__ bool Metal::scatter(
    size_t id,
    Ray const& ray,
    HitRecord const& rec,
    Color& attenuation,
    Ray& scattered) const
{
    auto reflected = reflect_vector(unit_vector(ray.direction()), rec.normal);
    scattered = Ray(rec.p, reflected);
    attenuation = m_albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ Vec3 Metal::reflect_vector(Vec3 vec, Vec3 normal)
{
    // dot(vec, normal) = scale to multiply the normal because 'vec' is not 'unit length'
    // but 'normal' is and we want the reflected vector to be the same length as the original one
    return vec - 2 * dot(vec, normal) * normal;
}