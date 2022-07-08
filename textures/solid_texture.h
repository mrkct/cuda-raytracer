#ifndef SOLID_TEXTURE_H
#define SOLID_TEXTURE_H

#include <textures/texture.h>

struct SolidTextureData {
    color c;
};

/**
 * @brief Creates the necessary texture data to create a solid color texture
 *
 * @param c Color of the texture
 * @return struct SolidTextureData
 */
struct SolidTextureData make_solid_texture_data(color c);

/**
 * @brief Wraps a SolidTextureData into a Texture object
 *
 * @param d
 * @return struct Texture
 */
struct Texture make_solid_texture(struct SolidTextureData* d);

/**
 * @brief Returns the color at a given point on the texture. In this case
 * as the texture is made of a solid color, this function simply returns
 * a constant and all of its arguments are useless and exists only to
 * have a consistent interface with other textures
 *
 * @param d
 * @return color
 */
__device__ color solid_texture_color_at(struct SolidTextureData* d, float const, float const, point3 const&);

#endif