#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

class Material;

struct HitRecord {
    Point3 p;
    Vec3 normal;
    float t;
    Material const* material;

    bool front_face;

    __device__ inline void set_face_normal(Ray const& ray, Vec3 const& outward_normal)
    {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    __device__ virtual bool hit(Ray const& r, double t_min, double t_max, HitRecord& rec) const = 0;
};
