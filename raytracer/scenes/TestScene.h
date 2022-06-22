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
    static Hittable& init(DeviceRNG& rng) { return new_on_device<TestScene>(rng); }

    __device__ TestScene(DeviceRNG& rng)
        : m_world(*new HittableList)
    {
        auto& material_ground = *new Lambertian(rng, Color(0.8, 0.8, 0.0));
        auto& material_center = *new Lambertian(rng, Color(0.1, 0.2, 0.5));
        auto& material_left = *new Dielectric(rng, 0.5);
        auto& material_right = *new Metal(rng, Color(0.8, 0.6, 0.2), 0.0);

        m_world.reserve(4);

        m_world.append(new Sphere({ 0, 0, -1 }, 0.5, material_center));
        m_world.append(new Sphere({ 1, 0, -1 }, 0.5, material_right));
        m_world.append(new Sphere({ -1, 0, -1 }, 0.5, material_left));

        m_world.append(new Sphere({ 0, -100.5, -1 }, 100, material_ground));
    }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

private:
    HittableList& m_world;
};