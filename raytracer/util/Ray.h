#pragma once

#include <raytracer/util/Vec3.h>

class Ray {
public:
    __device__ Ray() { }
    __device__ Ray(Point3 const& origin, Vec3 const& direction)
        : orig(origin)
        , dir(direction)
    {
    }

    __device__ Point3 origin() const { return orig; }
    __device__ Vec3 direction() const { return dir; }

    __device__ Point3 at(float t) const
    {
        return orig + t * dir;
    }

public:
    Point3 orig;
    Vec3 dir;
};
