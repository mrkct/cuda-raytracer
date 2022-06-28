#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

class Camera {
public:
    struct Builder {
        Point3 look_from, look_at;
        float vertical_fov;
        size_t image_width, image_height;

        __device__ Camera build()
        {
            printf("Camera::Builder.build()\n");
            return Camera(look_from, look_at, { 0, 1, 0 }, vertical_fov, image_width, image_height);
        }
    };

    __device__ Camera(
        Point3 look_from,
        Point3 look_at,
        Vec3 view_up,
        float vertical_fov,
        size_t image_width,
        size_t image_height);

    __device__ Ray get_ray(float u, float v) const
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