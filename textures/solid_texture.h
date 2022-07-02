#ifndef SOLID_TEXTURE_H
#define SOLID_TEXTURE_H

#include <textures/texture.h>

struct SolidTextureData {
    color c;
};

struct SolidTextureData make_solid_texture_data(color c);
struct Texture make_solid_texture(struct SolidTextureData* d);
__device__ color solid_texture_color_at(struct SolidTextureData* d, double, double, point3);

#endif