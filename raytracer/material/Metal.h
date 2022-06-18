#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/DeviceRNG.h>

class Metal : public Material {
public:
    __device__ Metal(DeviceRNG& rng, Color albedo, float fuzz)
        : m_rng(rng)
        , m_albedo(albedo)
        , m_fuzz(fuzz)
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

    DeviceRNG& m_rng;
    Color m_albedo;
    float m_fuzz;
};