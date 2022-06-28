#pragma once

#include <raytracer/HittableList.h>
#include <raytracer/geometry/Hittable.h>

class TestScene : public Hittable {
public:
    static Hittable& init();

    __device__ TestScene();

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

    __device__ virtual int id() const override { return -2; };

private:
    HittableList& m_world;
};