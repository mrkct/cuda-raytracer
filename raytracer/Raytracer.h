#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <raytracer/DeviceCanvas.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/DeviceRNG.h>

class Raytracer {
public:
    Raytracer(unsigned int image_width, unsigned int image_height)
        : m_image({ image_width, image_height })
    {
        static constexpr int blockSize = 8;

        m_grid = { (m_image.width + blockSize - 1) / blockSize, (m_image.height + blockSize - 1) / blockSize };
        m_blocks = { blockSize, blockSize };
        m_rng = DeviceRNG::init(m_grid, m_blocks, m_image.width, m_image.height);
    }

    Hittable& prepare_scene(Hittable& (*init)(DeviceRNG&)) const { return init(*m_rng); }
    DeviceCanvas create_canvas() { return DeviceCanvas(m_image.width, m_image.height); }

    void trace_scene(DeviceCanvas&, Point3 camera_position, Point3 look_at, Hittable& world);

private:
    struct {
        unsigned int width, height;
    } m_image;
    dim3 m_grid, m_blocks;
    DeviceRNG* m_rng;
};

#endif