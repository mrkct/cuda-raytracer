#include <stdio.h>
#include <textures/checkered_texture.h>
#include <textures/image_texture.h>
#include <textures/solid_texture.h>
#include <textures/texture.h>

__device__ color texture_color_at(struct Texture* texture, double u, double v, vec3& point)
{
    switch (texture->texture_type) {
    case SOLID_COLOR:
        return solid_texture_color_at((struct SolidTextureData*)texture->data, u, v, point);
    case CHECKERED:
        return checkered_texture_color_at((struct CheckeredTextureData*)texture->data, u, v, point);
    case IMAGE:
        return image_texture_color_at((struct ImageTextureData*)texture->data, u, v, point);
    default:
        printf("texture not implemented!!!\n");
        return make_color(0, 0, 0);
    };
}
