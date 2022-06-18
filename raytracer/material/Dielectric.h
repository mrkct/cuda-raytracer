#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>

class Dielectric : public Material {
public:
    __device__ Dielectric(float refraction_level)
        : m_refraction_level(refraction_level)
    {
    }

    __device__ virtual bool scatter(
        size_t id,
        Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const override;

private:
    static __device__ Vec3 refract_vector(Vec3 const& uv, Vec3 const& n, double etai_over_etat);

    float m_refraction_level;
};