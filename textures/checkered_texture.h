#ifndef CHECKERED_TEXTURE_H
#define CHECKERED_TEXTURE_H

#include <textures/texture.h>

struct CheckeredTextureData {
    float size;
    color c1, c2;
};

struct CheckeredTextureData make_checkered_texture_data(float size, color c1, color c2);
struct Texture make_checkered_texture(struct CheckeredTextureData* d);
__device__ color checkered_texture_color_at(struct CheckeredTextureData* d, float const, float const, point3 const&);

#endif