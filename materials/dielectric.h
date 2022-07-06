#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include <materials/material.h>

struct DielectricData {
    float refraction_level;
};

struct DielectricData make_dielectric_material_data(float refraction_level);
struct Material make_dielectric_material(struct DielectricData*);
__device__ bool dielectric_scatter(void*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif