#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include <materials/material.h>
#include <textures/texture.h>

struct LambertianData {
    struct Texture* texture;
};

/**
 * @brief Creates the data for a lambertian material with a given texture
 *
 * @param texture The texture of the material
 * @return struct LambertianData
 */
struct LambertianData make_lambertian_material_data(struct Texture* texture);

/**
 * @brief Wraps a lambertian material data into a Material object
 *
 * @return struct Material
 */
struct Material make_lambertian_material(struct LambertianData*);

/**
 * @brief Calculates whether a ray is scattered after colliding with a lambertian material,
 * in that case it also calculates the attenuation on the ray's color and the reflected ray.
 *
 * @return true if the ray is scattered, false if the ray is absorbed
 */
__device__ bool lambertian_scatter(void*, curandState_t*, struct Ray const&, HitRecord*, color*, struct Ray*);

#endif