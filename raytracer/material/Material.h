#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

struct HitRecord;

class Material {
public:
    __device__ virtual bool scatter(
        size_t id,
        Ray const& ray,
        HitRecord const& rec,
        Color& attenuation,
        Ray& scattered) const = 0;
};