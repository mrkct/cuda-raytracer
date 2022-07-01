#ifndef HITTABLE_H
#define HITTABLE_H

#include <ray.h>
#include <util/vec3.h>

struct Material;

struct HitRecord {
    point3 p;
    vec3 normal;
    double t;
    Material const* material;

    bool front_face;
};

__device__ inline void set_face_normal(HitRecord* record, struct Ray ray, vec3 outward_normal)
{
    record->front_face = vec3_dot(ray.direction, outward_normal) < 0;
    record->normal = record->front_face ? outward_normal : vec3_negate(outward_normal);
}

#endif