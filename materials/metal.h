#ifndef METAL_H
#define METAL_H

#include <materials/material.h>

struct MetalData {
    color albedo;
    float fuzz;
};

/**
 * @brief Creates the data for a metallic material with
 *
 * @param albedo The innate color of the metal
 * @param fuzz How much the metal reflects clearly, this is a value in [0, 1]
 * @return struct MetalData
 */
struct MetalData make_metal_material_data(color albedo, float fuzz);

/**
 * @brief Wraps a metal material data into a Material object
 *
 * @return struct Material
 */
struct Material make_metal_material(struct MetalData*);

/**
 * @brief Calculates whether a ray is scattered after colliding with a metallic material,
 * in that case it also calculates the attenuation on the ray's color and the reflected ray.
 *
 * @return true if the ray is scattered, false if the ray is absorbed
 */
__device__ bool metal_scatter(void*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif