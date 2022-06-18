#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/DeviceRNG.h>

class Dielectric : public Material {
public:
    __device__ Dielectric(DeviceRNG& rng, float refraction_level)
        : m_rng(rng)
        , m_refraction_level(refraction_level)
    {
    }

    __device__ virtual bool scatter(
        size_t id,
        Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const override;

private:
    static __device__ Vec3 refract_vector(Vec3 const& uv, Vec3 const& n, double etai_over_etat);
    static __device__ float reflectance(float cosine, float ref_idx);

    DeviceRNG& m_rng;
    float m_refraction_level;
};