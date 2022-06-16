#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/DeviceArray.h>

class HittableList : public Hittable {
public:
    __device__ HittableList()
    {
    }

    __device__ void reserve(size_t elements);

    __device__ void append(Hittable* h) { m_objects.append(h); }

    __device__ virtual bool hit(Ray const& r, double t_min, double t_max, HitRecord& rec) const override;

private:
    DeviceArray<Hittable*> m_objects = { .elements = nullptr, .length = 0 };
};