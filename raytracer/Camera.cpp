#include <raytracer/Camera.h>
#include <raytracer/util/CudaHelpers.h>

__device__ Camera::Camera(
    Point3 look_from,
    Point3 look_at,
    Vec3 view_up,
    float vertical_fov,
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
