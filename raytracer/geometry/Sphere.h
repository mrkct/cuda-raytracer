#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/Ray.h>

class Sphere : public Hittable {
public:
    __device__ Sphere(Point3 center, double radius, Material const& material)
        : m_center(center)
        , m_radius(radius)
        , m_material(material) {};

    __device__ virtual bool hit(
        Ray const& r, double t_min, double t_max, HitRecord& rec) const override;

public:
    Point3 m_center;
    double m_radius;
    Material const& m_material;
};
