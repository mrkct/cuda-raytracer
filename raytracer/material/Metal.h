#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/DeviceRNG.h>

class Metal : public Material {
public:
    __device__ Metal(Color albedo, float fuzz)
        : m_albedo(albedo)
        , m_fuzz(fuzz)
    {
    }

    __device__ virtual bool scatter(
        DeviceRNG& rng,
        Ray const& ray,
        HitRecord const& rec,
        Color& attenuation,
        Ray& scattered) const override;

private:
    Color m_albedo;
    float m_fuzz;
};