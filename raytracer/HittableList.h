#pragma once

#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/DeviceArray.h>

class HittableList : public Hittable {
public:
    __device__ HittableList()
    {
    }

    __device__ HittableList(DeviceArray<Hittable*> d)
        : m_objects(d)
    {
    }

    __device__ void reserve(size_t elements);

    __device__ void append(Hittable* h) { m_objects.append(h); }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override;

    __device__ virtual int id() const override { return -1; };

public:
    DeviceArray<Hittable*> m_objects = { .elements = nullptr, .length = 0 };
};