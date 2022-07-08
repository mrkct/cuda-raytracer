#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include <materials/material.h>

struct DielectricData {
    float refraction_level;
};

/**
 * @brief Creates the data for a dielectric material with a given refraction level
 *
 * @param refraction_level A value between (0, 1] indicating how much light is refracted,
 * the rest of the light is reflected
 * @return struct DielectricData
 */
struct DielectricData make_dielectric_material_data(float refraction_level);

/**
 * @brief Wraps a lambertian material data into a Material object
 *
 * @return struct Material
 */
struct Material make_dielectric_material(struct DielectricData*);

/**
 * @brief Calculates whether a ray is scattered after colliding with a lambertian material,
 * in that case it also calculates the attenuation on the ray's color and the reflected ray.
 *
 * @return true if the ray is scattered, false if the ray is absorbed
 */
__device__ bool dielectric_scatter(void*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif