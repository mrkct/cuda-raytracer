#ifndef SCENE_H
#define SCENE_H

#include <geometry/sphere.h>
#include <ray.h>

struct Scene {
    Sphere* spheres;
    unsigned spheres_length;
};

/**
 * @brief Calculates wheter the argument ray has hit any object in the scene.
 * In that case, returns true and fill the passed HitRecord with the data
 * about the intersection
 *
 * @param t_min Minimum distance to consider in the intersection
 * @param t_max Maximum distance to consider for the intersection
 * @param out An output argument where the data for the intersection will be stored
 * @return true if the ray has hit an object, false otherwise
 */
__device__ bool ray_scene_hit(struct Scene const&, struct Ray const&, float const t_min, float const t_max, HitRecord* out);

#endif