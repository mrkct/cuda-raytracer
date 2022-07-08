#ifndef CHECKERED_TEXTURE_H
#define CHECKERED_TEXTURE_H

#include <textures/texture.h>

struct CheckeredTextureData {
    float size;
    color c1, c2;
};

/**
 * @brief Creates a CheckeredTextureData object representing a checkered
 * texture with squares alternating between the 2 argument colors and
 * with sides of 'size'
 *
 * @param size A value between 0 and 1 representing the size of the sides of the squares
 * @param c1 Color of the first type of squares
 * @param c2 Color of the second type of squares
 * @return struct CheckeredTextureData
 */
struct CheckeredTextureData make_checkered_texture_data(float size, color c1, color c2);

/**
 * @brief Wraps a CheckeredTextureData into a Texture object
 *
 * @param d
 * @return struct Texture
 */
struct Texture make_checkered_texture(struct CheckeredTextureData* d);

/**
 * @brief Returns the color of the checkered texture at a given point
 *
 * @param u A value between [0, 1] representing the horizontal relative position on the texture
 * @param v A value between [0, 1] representing the vertical relative position on the tecture
 * @param point Collision point at which the ray and geometry intersected
 * @return color
 */
__device__ color checkered_texture_color_at(struct CheckeredTextureData* d, float const, float const, point3 const&);

#endif