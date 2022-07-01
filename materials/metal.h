#ifndef METAL_H
#define METAL_H

#include <materials/material.h>

struct MetalData {
    color albedo;
    double fuzz;
};

struct MetalData make_metal_material_data(color albedo, double fuzz);
struct Material make_metal_material(struct MetalData*);
__device__ bool metal_scatter(void*, curandState_t*, struct Ray, HitRecord*, color*, struct Ray*);

#endif