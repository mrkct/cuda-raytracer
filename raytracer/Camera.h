#pragma once

#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>

class Camera {
public:
    __device__ Camera(
        Point3 look_from,
        Point3 look_at,
        Vec3 view_up,
        double vertical_fov,
        size_t image_width,
        size_t image_height)
    {
        auto const aspect_ratio = (float)image_width / image_height;

        auto theta = degrees_to_radians(vertical_fov);
        auto h = tanf(theta / 2.0f);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        auto w = unit_vector(look_from - look_at);
        auto u = unit_vector(cross(view_up, w));
        auto v = cross(w, u);

        origin = look_from;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
    }

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