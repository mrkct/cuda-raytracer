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

__device__ color texture_color_at(struct Texture*, float u, float v, vec3& point);

#endif