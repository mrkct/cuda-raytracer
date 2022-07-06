#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>
#include <geometry/hittable.h>
#include <ray.h>

typedef bool (*ScatterFunction)(void*, curandState_t*, struct Ray, HitRecord*, color*, struct Ray*);

enum MaterialType { LAMBERTIAN,
    METAL,
    DIELECTRIC };

struct Material {
    enum MaterialType material_type;
    void* data;
};

__device__ bool material_scatter(struct Material const*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif