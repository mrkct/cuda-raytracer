#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include <materials/material.h>
#include <textures/texture.h>

struct LambertianData {
    struct Texture* texture;
};

struct LambertianData make_lambertian_material_data(struct Texture* texture);
struct Material make_lambertian_material(struct LambertianData*);
__device__ bool lambertian_scatter(void*, curandState_t*, struct Ray, HitRecord*, color*, struct Ray*);

#endif