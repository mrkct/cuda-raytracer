#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <raytracer/TracedScene.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/DeviceRNG.h>

class Raytracer {
public:
    Raytracer(unsigned int image_width, unsigned int image_height)
        : m_image({ image_width, image_height })
    {
    }

    TracedScene trace_scene(Point3 camera_position, Point3 look_at, Hittable& (*init)(DeviceRNG&));

private:
    struct {
        unsigned int width, height;
    } m_image;
};

#endif