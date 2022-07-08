#ifndef HITTABLE_H
#define HITTABLE_H

#include <ray.h>
#include <util/vec3.h>

struct Material;

struct HitRecord {
    point3 p;
    vec3 normal;
    float t;

    float u, v;

    Material const* material;
    bool front_face;
};

/**
 * @brief Given a normal pointing outwards and a ray,
 * decide whether the normal should be inverted or not.
 * This sets values inside the 'record' argument
 *
 * @param record Output parameter, will have some of its fields changed
 * @param ray The ray from which the outward normal is derived
 * @param outward_normal A normalized vector pointing outwards from the object
 */
__device__ inline void set_face_normal(HitRecord* record, struct Ray const& ray, vec3 const& outward_normal)
{
    record->front_face = vec3_dot(ray.direction, outward_normal) < 0;
    record->normal = record->front_face ? outward_normal : vec3_negate(outward_normal);
}

#endif