#ifndef TEXTURE_H
#define TEXTURE_H

#include <util/vec3.h>

enum TextureType {
    SOLID_COLOR,
    CHECKERED,
    IMAGE
};

struct Texture {
    TextureType texture_type;
    void* data;
};

/**
 * @brief Returns the color of a texture at a given collision point
 *
 * @param u A value between [0, 1] representing the horizontal relative position on the texture
 * @param v A value between [0, 1] representing the vertical relative position on the tecture
 * @param point Collision point at which the ray and geometry intersected
 * @return __device__
 */
__device__ color texture_color_at(struct Texture*, float u, float v, vec3& point);

#endif