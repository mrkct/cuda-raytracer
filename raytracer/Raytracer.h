#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <raytracer/TracedScene.h>
#include <raytracer/Scene.h>

class Raytracer {
public:
    Raytracer(int image_width, int image_height)
        : m_image({image_width, image_height})
    {}

    TracedScene trace_scene(Scene const&);

    private:
    static constexpr int bytes_per_pixel = 4;
    struct {
        int width, height;
    } m_image;
};

#endif