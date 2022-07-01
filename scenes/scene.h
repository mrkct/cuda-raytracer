#ifndef SCENE_H
#define SCENE_H

#include <geometry/sphere.h>
#include <ray.h>

struct Scene {
    Sphere* spheres;
    unsigned spheres_length;
};

__device__ bool ray_scene_hit(struct Scene, struct Ray, double t_min, double t_max, HitRecord* out);

#endif