#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/DeviceRNG.h>

class Lambertian : public Material {
public:
    __device__ Lambertian(DeviceRNG& rng, Color const& color)
        : m_albedo(color)
        , m_rng(rng)
    {
    }

    __device__ virtual bool scatter(
        size_t id,
        Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const override;

public:
    Color m_albedo;
    DeviceRNG& m_rng;
};