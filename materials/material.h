#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>
#include <geometry/hittable.h>
#include <ray.h>

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

struct Material {
    enum MaterialType material_type;
    void* data;
};

/**
 * @brief Calculates whether a ray is scattered after colliding with a particular material,
 * in that case it also calculates the attenuation on the ray's color and the reflected ray.
 *
 * @return true if the ray is scattered, false if the ray is absorbed
 */
__device__ bool material_scatter(struct Material const*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif