#include <textures/checkered_texture.h>

struct CheckeredTextureData make_checkered_texture_data(double size, color c1, color c2)
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

__device__ color checkered_texture_color_at(struct CheckeredTextureData* d, double u, double v, point3 p)
{
    int x = (int)(u / d->size) + (int)(v / d->size);
    return x % 2 == 0 ? d->c1 : d->c2;
}
