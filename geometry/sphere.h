#ifndef SPHERE_H
#define SPHERE_H

#include <geometry/hittable.h>
#include <materials/material.h>
#include <textures/texture.h>
#include <util/vec3.h>

struct Sphere {
    point3 origin;
    float radius;
    struct Material* material;
};

inline struct Sphere make_sphere(point3 origin, float radius, struct Material* material)
{
    return (struct Sphere) { .origin = origin, .radius = radius, .material = material };
}

__device__ bool ray_sphere_hit(struct Sphere, struct Ray, float t_min, float t_max, HitRecord* out);

#endif