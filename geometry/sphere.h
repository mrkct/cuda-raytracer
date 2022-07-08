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

/**
 * @brief Creates a sphere to be added to a scene
 *
 * @param origin
 * @param radius
 * @param material
 * @return struct Sphere
 */
inline struct Sphere make_sphere(point3 origin, float radius, struct Material* material)
{
    return (struct Sphere) { .origin = origin, .radius = radius, .material = material };
}

/**
 * @brief Returns whether a ray has hit a particular sphere and fills the HitRecord
 * with data about the intersection in case. Only intersections with a distance between
 * (t_min, t_max) are considered
 *
 * @param t_min Minimum distance to consider an intersection valid
 * @param t_max Maximum distance to consider an intersection valid
 * @param out Output parameter, will be filled with intersection data if the function returns true
 * @return true if the ray has hit the sphere, false otherwise
 */
__device__ bool ray_sphere_hit(struct Sphere const&, struct Ray const&, float const t_min, float const t_max, HitRecord* out);

#endif