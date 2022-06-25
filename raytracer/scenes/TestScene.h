#pragma once

#include <raytracer/HittableList.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Dielectric.h>
#include <raytracer/material/Lambertian.h>
#include <raytracer/material/Metal.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>

class TestScene : public Hittable {
public:
    static Hittable& init(DeviceRNG::Builder) { return new_on_device<TestScene>(); }

    __device__ TestScene()
        : m_world(*new HittableList)
    {
        auto& material_center = *new Lambertian(Color(1, 0, 0));

        m_world.reserve(1);

        m_world.append(new Sphere({ 0, 0, 2 }, 1, material_center));
        // m_world.append(new Sphere({ 1, 0, 1 }, 0.5, material_right));
        // m_world.append(new Sphere({ -1, 0, 1 }, 0.5, material_left));
        //
        // m_world.append(new Sphere({ 0, -100.5, 1 }, 100, material_ground));
    }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

private:
    HittableList& m_world;
};