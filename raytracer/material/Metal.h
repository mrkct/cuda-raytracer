#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>

class Metal : public Material {
public:
    __device__ Metal(Color albedo)
        : m_albedo(albedo)
    {
    }

    __device__ virtual bool scatter(
        size_t id,
        Ray const& ray,
        HitRecord const& rec,
        Color& attenuation,
        Ray& scattered) const override;

private:
    static __device__ Vec3 reflect_vector(Vec3, Vec3);

    Color m_albedo;
};