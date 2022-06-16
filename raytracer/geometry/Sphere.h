#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/Ray.h>

class Sphere : public Hittable {
public:
    __device__ Sphere() { }
    __device__ Sphere(Point3 center, double radius)
        : m_center(center)
        , m_radius(radius) {};

    __device__ virtual bool hit(
        Ray const& r, double t_min, double t_max, HitRecord& rec) const override;

public:
    Point3 m_center;
    double m_radius;
};
