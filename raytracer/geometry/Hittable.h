#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

struct HitRecord {
    Point3 p;
    Vec3 normal;
    float t;
};

class Hittable {
public:
    __device__ virtual bool hit(Ray const& r, double t_min, double t_max, HitRecord& rec) const = 0;
};
