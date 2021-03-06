#include <textures/checkered_texture.h>

struct CheckeredTextureData make_checkered_texture_data(float size, color c1, color c2)
{
    return (struct CheckeredTextureData) {
        .size = size,
        .c1 = c1,
        .c2 = c2
    };
}

struct Texture make_checkered_texture(struct CheckeredTextureData* d)
{
    return (struct Texture) { .texture_type = CHECKERED, .data = d };
}

__device__ color checkered_texture_color_at(struct CheckeredTextureData* d, float const u, float const v, point3 const& p)
{
    int const x = (int)(u / d->size) + (int)(v / d->size);
    return x % 2 == 0 ? d->c1 : d->c2;
}
