#ifndef SCENE_H
#define SCENE_H

#include <geometry/sphere.h>
#include <ray.h>

struct Scene {
    Sphere* spheres;
    unsigned spheres_length;
};

__device__ bool ray_scene_hit(struct Scene, struct Ray, float t_min, float t_max, HitRecord* out);

#endif