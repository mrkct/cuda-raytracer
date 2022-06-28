#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/DeviceRNG.h>

class Lambertian : public Material {
public:
    __device__ Lambertian(Color const& color)
        : m_albedo(color)
    {
    }

    __device__ virtual bool scatter(
        DeviceRNG&,
        Ray const& r_in, HitRecord const& rec, Color& attenuation, Ray& scattered) const override;

    __device__ virtual int id() const override { return 0; }

public:
    Color m_albedo;
    int x = 1;
};