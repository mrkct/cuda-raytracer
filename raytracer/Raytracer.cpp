#include <raytracer/Raytracer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>


void helloFromGPU (void) {
    printf("hello world from the GPU\n");
}

TracedScene Raytracer::trace_scene(Scene const&)
{
    auto *data = new uint8_t[TracedScene::bytes_per_pixel() * m_image.width * m_image.height];
    // helloFromGPU<<<1, 10>>>();
    return TracedScene { m_image.width, m_image.height, data };
}