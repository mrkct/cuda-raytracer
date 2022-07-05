#ifndef IMAGE_TEXTURE_H
#define IMAGE_TEXTURE_H

#include <stdint.h>
#include <textures/texture.h>

struct ImageTextureData {
    uint32_t* data;
    unsigned width, height;
};

struct ImageTextureData make_image_texture_data_from_file(char const*);
struct ImageTextureData make_image_texture_data(uint32_t*, unsigned, unsigned);
struct Texture make_image_texture(struct ImageTextureData* d);
__device__ color image_texture_color_at(struct ImageTextureData* d, float, float, point3);

#endif