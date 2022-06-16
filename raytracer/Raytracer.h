#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <raytracer/TracedScene.h>

class Raytracer {
public:
    Raytracer(unsigned int image_width, unsigned int image_height)
        : m_image({ image_width, image_height })
    {
    }

    TracedScene trace_scene();

private:
    struct {
        unsigned int width, height;
    } m_image;
};

#endif