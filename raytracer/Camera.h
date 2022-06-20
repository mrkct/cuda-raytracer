#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

class Camera {
public:
    static Camera& create_on_device(
        Point3 const& look_from,
        Point3 const& look_at,
        double vertical_fov,
        size_t image_width, size_t image_height);

    __device__ Camera(
        Point3 look_from,
        Point3 look_at,
        Vec3 view_up,
        double vertical_fov,
        size_t image_width,
        size_t image_height);

    __device__ Ray get_ray(double u, double v) const
    {
        return { origin, lower_left_corner + u * horizontal + v * vertical - origin };
    }

private:
    __device__ static float degrees_to_radians(float deg)
    {
        return deg * M_PI / 180.0f;
    }

    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
};