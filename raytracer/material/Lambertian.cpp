#include <raytracer/material/Lambertian.h>

#define tprintf       \
    if (id == 206273) \
    printf

__device__ bool Lambertian::scatter(
    DeviceRNG& rng,
    Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const
{
    Vec3 r = rng.next_in_unit_sphere();
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int id = (480 - row - 1) * 640 + col;

    tprintf("id: %u\trand_in_sphere=(%.6f %.6f %.6f)\n", id, r.x(), r.y(), r.z());
    auto scatter_direction = rec.normal + r;

    // Catch degenerate scatter direction
    if (scatter_direction.is_near_zero()) {
        scatter_direction = rec.normal;
        tprintf("id: %u\tdegenerate ray. scatter_dir=%.6f %.6f %.6f\n", id, scatter_direction.x(), scatter_direction.y(), scatter_direction.z());
    }

    scattered = { rec.p, scatter_direction };
    attenuation = m_albedo;
    return true;
}