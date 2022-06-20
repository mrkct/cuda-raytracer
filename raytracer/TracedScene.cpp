#include <raytracer/TracedScene.h>
#include <stb_image_write.h>

int TracedScene::write_to_file(char const* output_path)
{
    return stbi_write_png(
        output_path,
        width(),
        height(),
        4, // 4 bytes per pixel, RGBA
        pixel_data(),
        width() * 4);
}