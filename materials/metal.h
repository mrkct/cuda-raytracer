#ifndef METAL_H
#define METAL_H

#include <materials/material.h>

struct MetalData {
    color albedo;
    float fuzz;
};

struct MetalData make_metal_material_data(color albedo, float fuzz);
struct Material make_metal_material(struct MetalData*);
__device__ bool metal_scatter(void*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif