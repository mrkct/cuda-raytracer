#pragma once

#include <raytracer/util/DeviceRNG.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

struct HitRecord;

class Material {
public:
    __device__ virtual bool scatter(
        DeviceRNG&,
        Ray const& ray,
        HitRecord const& rec,
        Color& attenuation,
        Ray& scattered) const = 0;
};