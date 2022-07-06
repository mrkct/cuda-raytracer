#include <textures/solid_texture.h>

struct SolidTextureData make_solid_texture_data(color c)
{
    return (struct SolidTextureData) { .c = c };
}

struct Texture make_solid_texture(struct SolidTextureData* d)
{
    return (struct Texture) { .texture_type = SOLID_COLOR, .data = d };
}

__device__ color solid_texture_color_at(struct SolidTextureData* d, float const, float const, point3 const&)
{
    return d->c;
}
