#ifndef SCENE_H
#define SCENE_H

#include <geometry/sphere.h>
#include <ray.h>

struct Scene {
    Sphere* spheres;
    unsigned spheres_length;
};

__device__ bool ray_scene_hit(struct Scene const&, struct Ray const&, float const t_min, float const t_max, HitRecord* out);

#endif