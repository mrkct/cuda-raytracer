#ifndef IMAGE_TEXTURE_H
#define IMAGE_TEXTURE_H

#include <stdint.h>
#include <textures/texture.h>

struct ImageTextureData {
    uint32_t* data;
    unsigned width, height;
};

/**
 * @brief Loads an image from a path on the disk and creates
 * the texture data. This is a shortcut from loading the file
 * by yourself and the calling 'make_image_texture_data'
 *
 * @return struct ImageTextureData
 */
struct ImageTextureData make_image_texture_data_from_file(char const*);

/**
 * @brief Creates an ImageTextureData using an ARGB framebuffer you have
 * loaded yourself. The framebuffer MUST be accessible from the GPU side.
 *
 * @return struct ImageTextureData
 */
struct ImageTextureData make_image_texture_data(uint32_t*, unsigned, unsigned);

/**
 * @brief Wraps an ImageTextureData into a Texture object
 *
 * @param d
 * @return struct Texture
 */
struct Texture make_image_texture(struct ImageTextureData* d);

/**
 * @brief Returns the color of the image texture at a given collision point
 *
 * @param u A value between [0, 1] representing the horizontal relative position on the texture
 * @param v A value between [0, 1] representing the vertical relative position on the tecture
 * @param point Collision point at which the ray and geometry intersected
 * @return color
 */
__device__ color image_texture_color_at(struct ImageTextureData* d, float const, float const, point3 const&);

#endif